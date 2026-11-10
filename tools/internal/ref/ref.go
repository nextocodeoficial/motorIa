// Package ref is a deliberately naive Qwen3-MoE forward pass in pure Go.
// It is the golden reference for the C engine: simple loops, float64
// accumulation, no shared code with src/. Speed is irrelevant here.
package ref

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"

	"motoria/tools/internal/mia"
)

type Layer struct {
	AttnNorm, Wq, Wk, Wv, Wo, QNorm, KNorm, FfnNorm, Router []float32
}

type Expert struct{ Gate, Up, Down []float32 }

type Model struct {
	Cfg                   mia.Config
	Embd, OutNorm, Output []float32
	Layers                []Layer
	Experts               [][]Expert
}

// ---- mia.Source ------------------------------------------------------------

func (m *Model) Config() mia.Config { return m.Cfg }

func (m *Model) Dense(name string, n int) ([]float32, error) {
	switch name {
	case "token_embd":
		return m.Embd, nil
	case "output_norm":
		return m.OutNorm, nil
	case "output":
		return m.Output, nil
	}
	var l int
	var field string
	if _, err := fmt.Sscanf(name, "blk.%d.%s", &l, &field); err != nil {
		return nil, fmt.Errorf("unknown tensor %q", name)
	}
	L := m.Layers[l]
	t := map[string][]float32{
		"attn_norm": L.AttnNorm, "wq": L.Wq, "wk": L.Wk, "wv": L.Wv, "wo": L.Wo,
		"q_norm": L.QNorm, "k_norm": L.KNorm, "ffn_norm": L.FfnNorm, "router": L.Router,
	}[field]
	if t == nil {
		return nil, fmt.Errorf("unknown tensor %q", name)
	}
	return t, nil
}

func (m *Model) Expert(l, e int) (gate, up, down []float32, err error) {
	x := m.Experts[l][e]
	return x.Gate, x.Up, x.Down, nil
}

// ---- construction ---------------------------------------------------------

// Random builds a reproducible random model with sane activation scales.
func Random(c mia.Config, seed uint64) *Model {
	r := rand.New(rand.NewPCG(seed, seed^0x5eed))
	mat := func(rows, cols int) []float32 {
		x := make([]float32, rows*cols)
		s := 1 / math.Sqrt(float64(cols))
		for i := range x {
			x[i] = float32(r.NormFloat64() * s)
		}
		return x
	}
	vec := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(1 + 0.1*r.NormFloat64())
		}
		return x
	}
	qd, kvd := c.NHeads*c.HeadDim, c.NKVHeads*c.HeadDim
	m := &Model{Cfg: c, Embd: mat(c.Vocab, c.Dim), OutNorm: vec(c.Dim), Output: mat(c.Vocab, c.Dim)}
	for i := range m.Embd {
		m.Embd[i] *= 4 // embeddings are not projected, give them unit-ish scale
	}
	for l := 0; l < c.NLayers; l++ {
		m.Layers = append(m.Layers, Layer{
			AttnNorm: vec(c.Dim), Wq: mat(qd, c.Dim), Wk: mat(kvd, c.Dim), Wv: mat(kvd, c.Dim),
			Wo: mat(c.Dim, qd), QNorm: vec(c.HeadDim), KNorm: vec(c.HeadDim),
			FfnNorm: vec(c.Dim), Router: mat(c.NExperts, c.Dim),
		})
		ex := make([]Expert, c.NExperts)
		for e := range ex {
			ex[e] = Expert{
				Gate: mat(c.ExpertHidden, c.Dim), Up: mat(c.ExpertHidden, c.Dim),
				Down: mat(c.Dim, c.ExpertHidden),
			}
		}
		m.Experts = append(m.Experts, ex)
	}
	return m
}

// QuantizeRoundtrip returns a copy whose quantized tensors went through Q8_0
// and back, i.e. exactly the weights the C engine sees in a q8_0 file.
func QuantizeRoundtrip(m *Model) *Model {
	c := m.Cfg
	q := func(x []float32, cols int) []float32 {
		out := make([]float32, 0, len(x))
		for r := 0; r < len(x)/cols; r++ {
			out = append(out, mia.DequantizeQ8(mia.QuantizeRowQ8(x[r*cols:(r+1)*cols]))...)
		}
		return out
	}
	qd := c.NHeads * c.HeadDim
	n := &Model{Cfg: c, Embd: q(m.Embd, c.Dim), OutNorm: m.OutNorm, Output: q(m.Output, c.Dim)}
	for l, L := range m.Layers {
		L.Wq, L.Wk, L.Wv, L.Wo = q(L.Wq, c.Dim), q(L.Wk, c.Dim), q(L.Wv, c.Dim), q(L.Wo, qd)
		n.Layers = append(n.Layers, L)
		ex := make([]Expert, c.NExperts)
		for e, E := range m.Experts[l] {
			ex[e] = Expert{q(E.Gate, c.Dim), q(E.Up, c.Dim), q(E.Down, c.ExpertHidden)}
		}
		n.Experts = append(n.Experts, ex)
	}
	return n
}

// ---- math -----------------------------------------------------------------

func matvec(w, x []float32, rows int) []float32 {
	cols := len(x)
	out := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var s float64
		for c := 0; c < cols; c++ {
			s += float64(w[r*cols+c]) * float64(x[c])
		}
		out[r] = float32(s)
	}
	return out
}

func rmsnorm(x, w []float32, eps float32) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(ss/float64(len(x))+float64(eps))
	out := make([]float32, len(x))
	for i := range x {
		out[i] = float32(float64(x[i]) * inv * float64(w[i]))
	}
	return out
}

// rope applies NeoX-style rotary embedding (first half paired with second).
func rope(x []float32, headDim, pos int, theta float32) {
	half := headDim / 2
	for i := 0; i < half; i++ {
		freq := math.Pow(float64(theta), -2*float64(i)/float64(headDim))
		s, c := math.Sincos(float64(pos) * freq)
		a, b := float64(x[i]), float64(x[i+half])
		x[i] = float32(a*c - b*s)
		x[i+half] = float32(a*s + b*c)
	}
}

func softmax(x []float64) {
	mx := math.Inf(-1)
	for _, v := range x {
		mx = math.Max(mx, v)
	}
	var sum float64
	for i := range x {
		x[i] = math.Exp(x[i] - mx)
		sum += x[i]
	}
	for i := range x {
		x[i] /= sum
	}
}

// route returns the top-k experts (descending probability) and their weights.
func route(logits []float32, k int, normTopK bool) ([]int, []float32) {
	p := make([]float64, len(logits))
	for i, v := range logits {
		p[i] = float64(v)
	}
	softmax(p)
	idx := make([]int, len(p))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return p[idx[a]] > p[idx[b]] })
	idx = idx[:k]
	var sum float64
	for _, i := range idx {
		sum += p[i]
	}
	w := make([]float32, k)
	for j, i := range idx {
		if normTopK {
			w[j] = float32(p[i] / sum)
		} else {
			w[j] = float32(p[i])
		}
	}
	return idx, w
}

func silu(x float64) float64 { return x / (1 + math.Exp(-x)) }

// ---- session (KV cache + one-token step) ---------------------------------

type Session struct {
	m      *Model
	kc, vc [][][]float32 // [layer][pos][kvd]
}

func NewSession(m *Model) *Session {
	return &Session{m: m, kc: make([][][]float32, m.Cfg.NLayers), vc: make([][][]float32, m.Cfg.NLayers)}
}

// Step feeds `tok` at position `pos` (must be sequential) and returns logits.
func (s *Session) Step(tok, pos int) []float32 {
	c, m := s.m.Cfg, s.m
	hd := c.HeadDim
	x := append([]float32(nil), m.Embd[tok*c.Dim:(tok+1)*c.Dim]...)
	for l, L := range m.Layers {
		xn := rmsnorm(x, L.AttnNorm, c.NormEps)
		q := matvec(L.Wq, xn, c.NHeads*hd)
		k := matvec(L.Wk, xn, c.NKVHeads*hd)
		v := matvec(L.Wv, xn, c.NKVHeads*hd)
		for h := 0; h < c.NHeads; h++ {
			copy(q[h*hd:], rmsnorm(q[h*hd:(h+1)*hd], L.QNorm, c.NormEps))
			rope(q[h*hd:(h+1)*hd], hd, pos, c.RopeTheta)
		}
		for h := 0; h < c.NKVHeads; h++ {
			copy(k[h*hd:], rmsnorm(k[h*hd:(h+1)*hd], L.KNorm, c.NormEps))
			rope(k[h*hd:(h+1)*hd], hd, pos, c.RopeTheta)
		}
		s.kc[l] = append(s.kc[l][:pos], k)
		s.vc[l] = append(s.vc[l][:pos], v)

		att := make([]float32, c.NHeads*hd)
		group := c.NHeads / c.NKVHeads
		for h := 0; h < c.NHeads; h++ {
			kvh := h / group
			sc := make([]float64, pos+1)
			for t := 0; t <= pos; t++ {
				var d float64
				for i := 0; i < hd; i++ {
					d += float64(q[h*hd+i]) * float64(s.kc[l][t][kvh*hd+i])
				}
				sc[t] = d / math.Sqrt(float64(hd))
			}
			softmax(sc)
			for i := 0; i < hd; i++ {
				var o float64
				for t := 0; t <= pos; t++ {
					o += sc[t] * float64(s.vc[l][t][kvh*hd+i])
				}
				att[h*hd+i] = float32(o)
			}
		}
		for i, v := range matvec(L.Wo, att, c.Dim) {
			x[i] += v
		}

		xn = rmsnorm(x, L.FfnNorm, c.NormEps)
		idx, w := route(matvec(L.Router, xn, c.NExperts), c.NActive, c.NormTopK)
		for j, e := range idx {
			E := m.Experts[l][e]
			g := matvec(E.Gate, xn, c.ExpertHidden)
			u := matvec(E.Up, xn, c.ExpertHidden)
			for i := range g {
				g[i] = float32(silu(float64(g[i])) * float64(u[i]))
			}
			for i, v := range matvec(E.Down, g, c.Dim) {
				x[i] += w[j] * v
			}
		}
	}
	return matvec(m.Output, rmsnorm(x, m.OutNorm, c.NormEps), c.Vocab)
}
