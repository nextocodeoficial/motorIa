package hf

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"motoria/tools/internal/mia"
	"motoria/tools/internal/ref"
)

func tinyCfg() mia.Config {
	return mia.Config{
		Dim: 64, NLayers: 2, NHeads: 4, NKVHeads: 2, HeadDim: 16, Vocab: 96,
		NExperts: 8, NActive: 2, ExpertHidden: 32, NormTopK: true,
		RopeTheta: 1e6, NormEps: 1e-6,
	}
}

func bf16(v float32) uint16 { return uint16(math.Float32bits(v) >> 16) }

// writeShard writes a safetensors file with BF16 tensors.
func writeShard(t *testing.T, path string, tensors map[string][]float32, shapes map[string][]int) {
	t.Helper()
	header := map[string]any{}
	var data []byte
	for name, x := range tensors {
		begin := len(data)
		for _, v := range x {
			data = binary.LittleEndian.AppendUint16(data, bf16(v))
		}
		header[name] = map[string]any{"dtype": "BF16", "shape": shapes[name], "data_offsets": []int{begin, len(data)}}
	}
	hj, _ := json.Marshal(header)
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(hj)))
	out = append(append(out, hj...), data...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}

// writeCheckpoint lays out m as a 2-shard HF Qwen3-MoE checkpoint.
func writeCheckpoint(t *testing.T, dir string, m *ref.Model) {
	c := m.Cfg
	qd, kvd := c.NHeads*c.HeadDim, c.NKVHeads*c.HeadDim
	shards := []map[string][]float32{{}, {}}
	shapes := map[string][]int{}
	index := map[string]string{}
	put := func(shard int, name string, x []float32, shape ...int) {
		shards[shard][name] = x
		shapes[name] = shape
		index[name] = fmt.Sprintf("model-%05d-of-00002.safetensors", shard+1)
	}
	put(0, "model.embed_tokens.weight", m.Embd, c.Vocab, c.Dim)
	put(1, "model.norm.weight", m.OutNorm, c.Dim)
	put(1, "lm_head.weight", m.Output, c.Vocab, c.Dim)
	for l, L := range m.Layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		s := l % 2
		put(s, p+"input_layernorm.weight", L.AttnNorm, c.Dim)
		put(s, p+"self_attn.q_proj.weight", L.Wq, qd, c.Dim)
		put(s, p+"self_attn.k_proj.weight", L.Wk, kvd, c.Dim)
		put(s, p+"self_attn.v_proj.weight", L.Wv, kvd, c.Dim)
		put(s, p+"self_attn.o_proj.weight", L.Wo, c.Dim, qd)
		put(s, p+"self_attn.q_norm.weight", L.QNorm, c.HeadDim)
		put(s, p+"self_attn.k_norm.weight", L.KNorm, c.HeadDim)
		put(s, p+"post_attention_layernorm.weight", L.FfnNorm, c.Dim)
		put(s, p+"mlp.gate.weight", L.Router, c.NExperts, c.Dim)
		for e, E := range m.Experts[l] {
			q := fmt.Sprintf("%smlp.experts.%d.", p, e)
			put(1-s, q+"gate_proj.weight", E.Gate, c.ExpertHidden, c.Dim)
			put(1-s, q+"up_proj.weight", E.Up, c.ExpertHidden, c.Dim)
			put(1-s, q+"down_proj.weight", E.Down, c.Dim, c.ExpertHidden)
		}
	}
	for i, sh := range shards {
		writeShard(t, filepath.Join(dir, fmt.Sprintf("model-%05d-of-00002.safetensors", i+1)), sh, shapes)
	}
	idx, _ := json.Marshal(map[string]any{"weight_map": index})
	os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), idx, 0o644)
	cfg, _ := json.Marshal(map[string]any{
		"model_type": "qwen3_moe", "hidden_size": c.Dim, "num_hidden_layers": c.NLayers,
		"num_attention_heads": c.NHeads, "num_key_value_heads": c.NKVHeads, "head_dim": c.HeadDim,
		"vocab_size": c.Vocab, "num_experts": c.NExperts, "num_experts_per_tok": c.NActive,
		"moe_intermediate_size": c.ExpertHidden, "norm_topk_prob": c.NormTopK,
		"rope_theta": c.RopeTheta, "rms_norm_eps": c.NormEps, "tie_word_embeddings": false,
		"decoder_sparse_step": 1, "mlp_only_layers": []int{},
	})
	os.WriteFile(filepath.Join(dir, "config.json"), cfg, 0o644)
}

func roundBF16(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = math.Float32frombits(uint32(bf16(v)) << 16)
	}
	return out
}

func equal(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", what, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %v, want %v", what, i, got[i], want[i])
		}
	}
}

func TestCheckpointReadsQwen3MoE(t *testing.T) {
	dir := t.TempDir()
	m := ref.Random(tinyCfg(), 7)
	writeCheckpoint(t, dir, m)

	ck, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ck.Close()

	if got := ck.Config(); got != tinyCfg() {
		t.Fatalf("config = %+v\nwant %+v", got, tinyCfg())
	}
	tests := []struct {
		name string
		want []float32
	}{
		{"token_embd", m.Embd},
		{"output", m.Output},
		{"blk.1.wo", m.Layers[1].Wo},
		{"blk.0.router", m.Layers[0].Router},
		{"blk.1.k_norm", m.Layers[1].KNorm},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ck.Dense(tt.name, len(tt.want))
			if err != nil {
				t.Fatal(err)
			}
			equal(t, tt.name, got, roundBF16(tt.want))
		})
	}
	g, u, d, err := ck.Expert(1, 6)
	if err != nil {
		t.Fatal(err)
	}
	E := m.Experts[1][6]
	equal(t, "gate", g, roundBF16(E.Gate))
	equal(t, "up", u, roundBF16(E.Up))
	equal(t, "down", d, roundBF16(E.Down))
}

func TestOpenRejectsDenseMLPLayers(t *testing.T) {
	dir := t.TempDir()
	writeCheckpoint(t, dir, ref.Random(tinyCfg(), 7))
	raw, _ := os.ReadFile(filepath.Join(dir, "config.json"))
	var cfg map[string]any
	json.Unmarshal(raw, &cfg)
	cfg["mlp_only_layers"] = []int{0}
	raw, _ = json.Marshal(cfg)
	os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o644)
	if _, err := Open(dir); err == nil {
		t.Fatal("expected error for dense MLP layers")
	}
}
