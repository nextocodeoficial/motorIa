// Package mia defines the .mia on-disk format shared with the C engine
// (include/mia/mia_file.h). Any change here must be mirrored there.
//
// Layout:
//
//	[0, HeaderRegion)        Header (112 bytes, zero padded)
//	[DenseOffset, +DenseSize) dense tensors, always resident in RAM
//	[ExpertsOffset, ...)      one fixed-size blob per (layer, expert), streamed
//	[TokOffset, +TokSize)     tokenizer blob (optional)
package mia

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

const (
	QK           = 32
	HeaderRegion = 16384
	TensorAlign  = 64
	ExpertAlign  = 16384 // matches the SSD page so each miss is one aligned read
	Version      = 1
)

type WType uint32

const (
	F32 WType = 0
	Q80 WType = 1
)

func (w WType) String() string {
	if w == Q80 {
		return "q8_0"
	}
	return "f32"
}

// Config is the model hyper-parameters (Qwen3-MoE family).
type Config struct {
	Dim, NLayers, NHeads, NKVHeads, HeadDim, Vocab int
	NExperts, NActive, ExpertHidden                int
	NormTopK                                       bool
	RopeTheta, NormEps                             float32
}

// Header mirrors `mia_header` in C byte for byte.
type Header struct {
	Magic                                                   [4]byte
	Version, WType, Dim, NLayers, NHeads, NKVHeads, HeadDim uint32
	Vocab, NExperts, NActive, ExpertHidden, NormTopK        uint32
	RopeTheta, NormEps                                      float32
	Pad                                                     uint32
	DenseOffset, DenseSize                                  uint64
	ExpertsOffset, ExpertBlobSize                           uint64
	TokOffset, TokSize                                      uint64
}

type BlockQ8 struct {
	D float32
	Q [QK]int8
}

// QuantizeRowQ8 must stay bit-identical to mia_quantize_row_q8 in C.
func QuantizeRowQ8(x []float32) []BlockQ8 {
	out := make([]BlockQ8, len(x)/QK)
	for b := range out {
		xb := x[b*QK : (b+1)*QK]
		var amax float32
		for _, v := range xb {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		d := amax / 127
		var id float32
		if d != 0 {
			id = 1 / d
		}
		out[b].D = d
		for i, v := range xb {
			out[b].Q[i] = int8(math.Round(float64(v * id)))
		}
	}
	return out
}

func DequantizeQ8(b []BlockQ8) []float32 {
	out := make([]float32, len(b)*QK)
	for i, blk := range b {
		for j, q := range blk.Q {
			out[i*QK+j] = blk.D * float32(q)
		}
	}
	return out
}

// TensorEntry is one dense tensor. Matrices use the file's weight type;
// vectors and routers stay F32 (they are tiny and precision sensitive).
type TensorEntry struct {
	Name       string
	Rows, Cols int
	Quantized  bool
	Offset     int // relative to DenseOffset
	Bytes      int
}

type Layout struct {
	Cfg            Config
	WT             WType
	Dense          []TensorEntry
	DenseSize      int
	Gate, Up, Down int // offsets inside an expert blob
	ExpertBlobSize int
}

func alignUp(n, a int) int { return (n + a - 1) / a * a }

func (l Layout) MatBytes(rows, cols int) int {
	if l.WT == Q80 {
		return rows * (cols / QK) * 36
	}
	return rows * cols * 4
}

func NewLayout(c Config, wt WType) Layout {
	l := Layout{Cfg: c, WT: wt}
	off := 0
	add := func(name string, rows, cols int, quant bool) {
		e := TensorEntry{Name: name, Rows: rows, Cols: cols, Quantized: quant, Offset: off}
		if quant {
			e.Bytes = l.MatBytes(rows, cols)
		} else {
			e.Bytes = rows * cols * 4
		}
		l.Dense = append(l.Dense, e)
		off = alignUp(off+e.Bytes, TensorAlign)
	}
	qd, kvd := c.NHeads*c.HeadDim, c.NKVHeads*c.HeadDim
	add("token_embd", c.Vocab, c.Dim, true)
	for i := 0; i < c.NLayers; i++ {
		p := fmt.Sprintf("blk.%d.", i)
		add(p+"attn_norm", 1, c.Dim, false)
		add(p+"wq", qd, c.Dim, true)
		add(p+"wk", kvd, c.Dim, true)
		add(p+"wv", kvd, c.Dim, true)
		add(p+"wo", c.Dim, qd, true)
		add(p+"q_norm", 1, c.HeadDim, false)
		add(p+"k_norm", 1, c.HeadDim, false)
		add(p+"ffn_norm", 1, c.Dim, false)
		add(p+"router", c.NExperts, c.Dim, false)
	}
	add("output_norm", 1, c.Dim, false)
	add("output", c.Vocab, c.Dim, true)
	l.DenseSize = off

	l.Gate = 0
	l.Up = alignUp(l.MatBytes(c.ExpertHidden, c.Dim), TensorAlign)
	l.Down = l.Up + alignUp(l.MatBytes(c.ExpertHidden, c.Dim), TensorAlign)
	l.ExpertBlobSize = alignUp(l.Down+l.MatBytes(c.Dim, c.ExpertHidden), ExpertAlign)
	return l
}

// Source supplies float32 weights by canonical name. Dense returns n values
// (rows*cols, row-major). Expert returns gate/up [hidden x dim] and down
// [dim x hidden].
type Source interface {
	Config() Config
	Dense(name string, n int) ([]float32, error)
	Expert(layer, e int) (gate, up, down []float32, err error)
}

func encodeMatrix(x []float32, quant bool, wt WType, rows, cols int) []byte {
	if quant && wt == Q80 {
		out := make([]byte, 0, rows*(cols/QK)*36)
		for r := 0; r < rows; r++ {
			for _, b := range QuantizeRowQ8(x[r*cols : (r+1)*cols]) {
				out = binary.LittleEndian.AppendUint32(out, math.Float32bits(b.D))
				for _, q := range b.Q {
					out = append(out, byte(q))
				}
			}
		}
		return out
	}
	out := make([]byte, 0, len(x)*4)
	for _, v := range x {
		out = binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
	}
	return out
}

func float32bits(v float32) uint32 { return math.Float32bits(v) }

// WriteFile streams src into a .mia file. Only one tensor or expert is held
// in memory at a time, so it works for checkpoints larger than RAM.
func WriteFile(path string, src Source, wt WType, tokenizer []byte) error {
	c := src.Config()
	l := NewLayout(c, wt)
	nExperts := c.NLayers * c.NExperts
	h := Header{
		Magic: [4]byte{'M', 'I', 'A', '1'}, Version: Version, WType: uint32(wt),
		Dim: u32(c.Dim), NLayers: u32(c.NLayers), NHeads: u32(c.NHeads),
		NKVHeads: u32(c.NKVHeads), HeadDim: u32(c.HeadDim), Vocab: u32(c.Vocab),
		NExperts: u32(c.NExperts), NActive: u32(c.NActive),
		ExpertHidden: u32(c.ExpertHidden), RopeTheta: c.RopeTheta, NormEps: c.NormEps,
		DenseOffset: HeaderRegion, DenseSize: uint64(l.DenseSize),
		ExpertBlobSize: uint64(l.ExpertBlobSize),
	}
	if c.NormTopK {
		h.NormTopK = 1
	}
	h.ExpertsOffset = uint64(alignUp(HeaderRegion+l.DenseSize, ExpertAlign))
	h.TokOffset = h.ExpertsOffset + uint64(nExperts)*h.ExpertBlobSize
	h.TokSize = uint64(len(tokenizer))

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := &offsetWriter{w: bufio.NewWriterSize(f, 1<<22)}

	if err := binary.Write(w, binary.LittleEndian, h); err != nil {
		return err
	}
	w.padTo(int64(h.DenseOffset))
	for _, e := range l.Dense {
		x, err := src.Dense(e.Name, e.Rows*e.Cols)
		if err != nil {
			return fmt.Errorf("dense %s: %w", e.Name, err)
		}
		if len(x) != e.Rows*e.Cols {
			return fmt.Errorf("dense %s: got %d values, want %d", e.Name, len(x), e.Rows*e.Cols)
		}
		w.padTo(int64(h.DenseOffset) + int64(e.Offset))
		w.Write(encodeMatrix(x, e.Quantized, wt, e.Rows, e.Cols))
	}
	for i := 0; i < c.NLayers; i++ {
		for e := 0; e < c.NExperts; e++ {
			gate, up, down, err := src.Expert(i, e)
			if err != nil {
				return fmt.Errorf("expert %d/%d: %w", i, e, err)
			}
			base := int64(h.ExpertsOffset) + int64(i*c.NExperts+e)*int64(l.ExpertBlobSize)
			w.padTo(base + int64(l.Gate))
			w.Write(encodeMatrix(gate, true, wt, c.ExpertHidden, c.Dim))
			w.padTo(base + int64(l.Up))
			w.Write(encodeMatrix(up, true, wt, c.ExpertHidden, c.Dim))
			w.padTo(base + int64(l.Down))
			w.Write(encodeMatrix(down, true, wt, c.Dim, c.ExpertHidden))
		}
	}
	w.padTo(int64(h.TokOffset))
	w.Write(tokenizer)
	if w.err != nil {
		return w.err
	}
	return w.w.Flush()
}

func u32(v int) uint32 { return uint32(v) }

type offsetWriter struct {
	w   *bufio.Writer
	n   int64
	err error
}

func (o *offsetWriter) Write(p []byte) (int, error) {
	if o.err != nil {
		return 0, o.err
	}
	n, err := o.w.Write(p)
	o.n += int64(n)
	o.err = err
	return n, err
}

func (o *offsetWriter) padTo(off int64) {
	if off < o.n {
		o.err = fmt.Errorf("layout overlap: at %d, want %d", o.n, off)
		return
	}
	o.Write(make([]byte, off-o.n))
}
