// Package hf reads a Hugging Face Qwen3-MoE checkpoint (config.json +
// safetensors shards) and exposes it as a mia.Source. Tensors are read
// on demand with ReadAt, so memory stays at one tensor at a time.
package hf

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"motoria/tools/internal/mia"
)

type hfConfig struct {
	ModelType         string  `json:"model_type"`
	HiddenSize        int     `json:"hidden_size"`
	NumLayers         int     `json:"num_hidden_layers"`
	NumHeads          int     `json:"num_attention_heads"`
	NumKVHeads        int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	VocabSize         int     `json:"vocab_size"`
	NumExperts        int     `json:"num_experts"`
	NumExpertsPerTok  int     `json:"num_experts_per_tok"`
	MoeIntermediate   int     `json:"moe_intermediate_size"`
	NormTopKProb      bool    `json:"norm_topk_prob"`
	RopeTheta         float64 `json:"rope_theta"`
	RMSNormEps        float64 `json:"rms_norm_eps"`
	TieWordEmbeddings bool    `json:"tie_word_embeddings"`
	DecoderSparseStep int     `json:"decoder_sparse_step"`
	MLPOnlyLayers     []int   `json:"mlp_only_layers"`
}

type tensorInfo struct {
	DType   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

type shard struct {
	f       *os.File
	dataOff int64
	tensors map[string]tensorInfo
}

type Checkpoint struct {
	cfg    mia.Config
	tied   bool
	shards map[string]*shard // file name -> shard
	where  map[string]string // tensor name -> file name
}

func Open(dir string) (*Checkpoint, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var hc hfConfig
	if err := json.Unmarshal(raw, &hc); err != nil {
		return nil, fmt.Errorf("config.json: %w", err)
	}
	if hc.ModelType != "qwen3_moe" {
		return nil, fmt.Errorf("model_type %q not supported (want qwen3_moe)", hc.ModelType)
	}
	if len(hc.MLPOnlyLayers) > 0 || hc.DecoderSparseStep > 1 {
		return nil, fmt.Errorf("dense MLP layers are not supported (mlp_only_layers/decoder_sparse_step)")
	}
	if hc.HeadDim == 0 {
		hc.HeadDim = hc.HiddenSize / hc.NumHeads
	}
	ck := &Checkpoint{
		cfg: mia.Config{
			Dim: hc.HiddenSize, NLayers: hc.NumLayers, NHeads: hc.NumHeads,
			NKVHeads: hc.NumKVHeads, HeadDim: hc.HeadDim, Vocab: hc.VocabSize,
			NExperts: hc.NumExperts, NActive: hc.NumExpertsPerTok,
			ExpertHidden: hc.MoeIntermediate, NormTopK: hc.NormTopKProb,
			RopeTheta: float32(hc.RopeTheta), NormEps: float32(hc.RMSNormEps),
		},
		tied:   hc.TieWordEmbeddings,
		shards: map[string]*shard{},
		where:  map[string]string{},
	}

	files := []string{"model.safetensors"}
	if raw, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json")); err == nil {
		var idx struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(raw, &idx); err != nil {
			return nil, fmt.Errorf("index: %w", err)
		}
		seen := map[string]bool{}
		files = files[:0]
		for _, f := range idx.WeightMap {
			if !seen[f] {
				seen[f] = true
				files = append(files, f)
			}
		}
	}
	for _, name := range files {
		s, err := openShard(filepath.Join(dir, name))
		if err != nil {
			ck.Close()
			return nil, err
		}
		ck.shards[name] = s
		for t := range s.tensors {
			ck.where[t] = name
		}
	}
	return ck, nil
}

func openShard(path string) (*shard, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var n [8]byte
	if _, err := f.ReadAt(n[:], 0); err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	hlen := int64(binary.LittleEndian.Uint64(n[:]))
	if hlen <= 0 || hlen > 100<<20 {
		f.Close()
		return nil, fmt.Errorf("%s: bad safetensors header length %d", path, hlen)
	}
	hj := make([]byte, hlen)
	if _, err := f.ReadAt(hj, 8); err != nil {
		f.Close()
		return nil, err
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(hj, &hdr); err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	s := &shard{f: f, dataOff: 8 + hlen, tensors: map[string]tensorInfo{}}
	for name, v := range hdr {
		if name == "__metadata__" {
			continue
		}
		var ti tensorInfo
		if err := json.Unmarshal(v, &ti); err != nil {
			f.Close()
			return nil, fmt.Errorf("%s: tensor %s: %w", path, name, err)
		}
		s.tensors[name] = ti
	}
	return s, nil
}

func (c *Checkpoint) Close() {
	for _, s := range c.shards {
		s.f.Close()
	}
}

func (c *Checkpoint) Config() mia.Config { return c.cfg }

// read loads tensor `name` as float32, checking it has n elements.
func (c *Checkpoint) read(name string, n int) ([]float32, error) {
	file, ok := c.where[name]
	if !ok {
		return nil, fmt.Errorf("tensor %s not found", name)
	}
	s := c.shards[file]
	ti := s.tensors[name]
	size := ti.Offsets[1] - ti.Offsets[0]
	buf := make([]byte, size)
	if _, err := s.f.ReadAt(buf, s.dataOff+ti.Offsets[0]); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	var out []float32
	switch ti.DType {
	case "BF16":
		out = make([]float32, size/2)
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(buf[2*i:])) << 16)
		}
	case "F16":
		out = make([]float32, size/2)
		for i := range out {
			out[i] = f16(binary.LittleEndian.Uint16(buf[2*i:]))
		}
	case "F32":
		out = make([]float32, size/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
		}
	default:
		return nil, fmt.Errorf("%s: dtype %s not supported", name, ti.DType)
	}
	if len(out) != n {
		return nil, fmt.Errorf("%s: %d elements, want %d", name, len(out), n)
	}
	return out, nil
}

func f16(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := int32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign)
	case exp == 0: // subnormal
		v := float32(mant) / 1024 * float32(math.Pow(2, -14))
		if sign != 0 {
			return -v
		}
		return v
	case exp == 31:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	}
	return math.Float32frombits(sign | uint32(exp-15+127)<<23 | mant<<13)
}

var layerFields = map[string]string{
	"attn_norm": "input_layernorm.weight",
	"wq":        "self_attn.q_proj.weight",
	"wk":        "self_attn.k_proj.weight",
	"wv":        "self_attn.v_proj.weight",
	"wo":        "self_attn.o_proj.weight",
	"q_norm":    "self_attn.q_norm.weight",
	"k_norm":    "self_attn.k_norm.weight",
	"ffn_norm":  "post_attention_layernorm.weight",
	"router":    "mlp.gate.weight",
}

func (c *Checkpoint) Dense(name string, n int) ([]float32, error) {
	switch name {
	case "token_embd":
		return c.read("model.embed_tokens.weight", n)
	case "output_norm":
		return c.read("model.norm.weight", n)
	case "output":
		if c.tied {
			return c.read("model.embed_tokens.weight", n)
		}
		return c.read("lm_head.weight", n)
	}
	var l int
	var field string
	if _, err := fmt.Sscanf(name, "blk.%d.%s", &l, &field); err != nil || layerFields[field] == "" {
		return nil, fmt.Errorf("unknown tensor %q", name)
	}
	return c.read(fmt.Sprintf("model.layers.%d.%s", l, layerFields[field]), n)
}

func (c *Checkpoint) Expert(l, e int) (gate, up, down []float32, err error) {
	p := fmt.Sprintf("model.layers.%d.mlp.experts.%d.", l, e)
	n := c.cfg.ExpertHidden * c.cfg.Dim
	if gate, err = c.read(p+"gate_proj.weight", n); err != nil {
		return
	}
	if up, err = c.read(p+"up_proj.weight", n); err != nil {
		return
	}
	down, err = c.read(p+"down_proj.weight", n)
	return
}
