// Package e2e checks the C engine (build/mia-dump) against the Go reference.
// Run via `make e2e` (sets MIA_DUMP). Skipped in -short or without MIA_DUMP.
package e2e

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"motoria/tools/internal/mia"
	"motoria/tools/internal/ref"
	"motoria/tools/internal/tok"
)

var tokens = []int{1, 5, 9, 33, 2, 77, 64, 5, 5, 90, 12, 0}

func tinyCfg() mia.Config {
	return mia.Config{
		Dim: 64, NLayers: 2, NHeads: 4, NKVHeads: 2, HeadDim: 16, Vocab: 96,
		NExperts: 8, NActive: 2, ExpertHidden: 32, NormTopK: true,
		RopeTheta: 1e6, NormEps: 1e-6,
	}
}

func runDump(t *testing.T, bin, model string, slots, vocab int) [][]float32 {
	t.Helper()
	out := filepath.Join(t.TempDir(), "logits.bin")
	args := []string{model, out, strconv.Itoa(slots)}
	for _, tk := range tokens {
		args = append(args, strconv.Itoa(tk))
	}
	cmd := exec.Command(bin, args...)
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("mia-dump failed: %v\n%s", err, b)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 4*vocab*len(tokens) {
		t.Fatalf("logits file has %d bytes, want %d", len(raw), 4*vocab*len(tokens))
	}
	res := make([][]float32, len(tokens))
	for p := range res {
		res[p] = make([]float32, vocab)
		for i := range res[p] {
			res[p][i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*(p*vocab+i):]))
		}
	}
	return res
}

func argmax(x []float32) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}

func TestEngineMatchesReference(t *testing.T) {
	bin := os.Getenv("MIA_DUMP")
	if testing.Short() || bin == "" {
		t.Skip("set MIA_DUMP to the mia-dump binary (make e2e)")
	}
	cfg := tinyCfg()
	base := ref.Random(cfg, 42)

	tests := []struct {
		name   string
		wt     mia.WType
		refM   *ref.Model
		relTol float64 // max |c - ref| / max |ref|
	}{
		{"f32 weights match closely", mia.F32, base, 1e-4},
		// Q8 path also quantizes activations, so allow a little drift.
		{"q8_0 weights stay within activation-quantization error", mia.Q80, ref.QuantizeRoundtrip(base), 3e-2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tiny.mia")
			if err := mia.WriteFile(path, base, tt.wt, nil); err != nil {
				t.Fatal(err)
			}
			s := ref.NewSession(tt.refM)
			want := make([][]float32, len(tokens))
			for p, tk := range tokens {
				want[p] = s.Step(tk, p)
			}

			full := runDump(t, bin, path, 0, cfg.Vocab)
			for p := range tokens {
				var maxDiff, maxRef float64
				for i := range want[p] {
					maxDiff = math.Max(maxDiff, math.Abs(float64(full[p][i]-want[p][i])))
					maxRef = math.Max(maxRef, math.Abs(float64(want[p][i])))
				}
				if maxDiff/maxRef > tt.relTol {
					t.Errorf("pos %d: rel err %.2e > %.0e", p, maxDiff/maxRef, tt.relTol)
				}
				if tt.wt == mia.F32 && argmax(full[p]) != argmax(want[p]) {
					t.Errorf("pos %d: argmax %d, want %d", p, argmax(full[p]), argmax(want[p]))
				}
			}

			// Streaming through a 2-slot cache must not change a single bit.
			tight := runDump(t, bin, path, 2, cfg.Vocab)
			for p := range tokens {
				for i := range full[p] {
					if full[p][i] != tight[p][i] {
						t.Fatalf("pos %d logit %d: cached %v vs streamed %v", p, i, full[p][i], tight[p][i])
					}
				}
			}
		})
	}
}

// tinyTokenizerJSON maps ids 0..95 to printable ASCII bytes 32..127 so the
// tiny model's 96-token vocabulary is a valid byte-level BPE vocabulary.
func tinyTokenizerJSON(t *testing.T) string {
	t.Helper()
	vocab := map[string]int{}
	for i := 0; i < 93; i++ {
		b := byte(33 + i) // '!'..'}' map to themselves in GPT-2 byte encoding
		vocab[string(rune(b))] = i
	}
	vocab["Ġ"] = 93 // space
	vocab["Ċ"] = 94 // newline
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	js := `{"added_tokens":[{"id":95,"content":"<|im_end|>","special":true}],` +
		`"model":{"type":"BPE","vocab":` + mustJSON(vocab) + `,"merges":[]}}`
	if err := os.WriteFile(path, []byte(js), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

func TestCLIGeneratesWithinRAMBudget(t *testing.T) {
	bin := os.Getenv("MIA_BIN")
	if testing.Short() || bin == "" {
		t.Skip("set MIA_BIN to the mia binary (make e2e)")
	}
	blob, err := tok.Export(tinyTokenizerJSON(t))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tiny.mia")
	if err := mia.WriteFile(path, ref.Random(tinyCfg(), 3), mia.Q80, blob); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name    string
		args    []string
		wantErr bool
		wantOut string
	}{
		{"greedy raw prompt prints stats", []string{"-m", path, "-p", "hola mundo", "-n", "8", "-t", "0", "--raw", "--ram", "64M"}, false, "tok/s"},
		{"chat template needs im_start", []string{"-m", path, "-p", "hola", "-n", "4"}, true, "im_start"},
		{"budget below dense weights is rejected", []string{"-m", path, "-p", "x", "--raw", "--ram", "1K"}, true, "RAM budget"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := exec.Command(bin, tt.args...).CombinedOutput()
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %v\n%s", err, tt.wantErr, out)
			}
			if !strings.Contains(string(out), tt.wantOut) {
				t.Fatalf("output missing %q:\n%s", tt.wantOut, out)
			}
		})
	}
}
