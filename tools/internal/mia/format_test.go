package mia

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func tinyConfig() Config {
	return Config{
		Dim: 64, NLayers: 2, NHeads: 4, NKVHeads: 2, HeadDim: 16, Vocab: 96,
		NExperts: 8, NActive: 2, ExpertHidden: 32, NormTopK: true,
		RopeTheta: 1e6, NormEps: 1e-6,
	}
}

func TestHeaderIs112BytesLittleEndian(t *testing.T) {
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, Header{}); err != nil {
		t.Fatal(err)
	}
	if buf.Len() != 112 {
		t.Fatalf("header size = %d, want 112 (must match C struct)", buf.Len())
	}
}

func TestQuantizeRowQ8(t *testing.T) {
	tests := []struct {
		name  string
		in    func() []float32
		check func(t *testing.T, b []BlockQ8)
	}{
		{
			name: "max magnitude maps to 127 and halves round away from zero",
			in: func() []float32 {
				x := make([]float32, QK)
				x[3], x[7] = -2, 1
				return x
			},
			check: func(t *testing.T, b []BlockQ8) {
				if b[0].Q[3] != -127 || b[0].Q[7] != 64 {
					t.Fatalf("got q[3]=%d q[7]=%d, want -127 64", b[0].Q[3], b[0].Q[7])
				}
			},
		},
		{
			name: "zero block has zero scale",
			in:   func() []float32 { return make([]float32, QK) },
			check: func(t *testing.T, b []BlockQ8) {
				if b[0].D != 0 {
					t.Fatalf("scale = %v, want 0", b[0].D)
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) { tt.check(t, QuantizeRowQ8(tt.in())) })
	}
}

func TestLayoutAlignment(t *testing.T) {
	for _, wt := range []WType{F32, Q80} {
		l := NewLayout(tinyConfig(), wt)
		for _, e := range l.Dense {
			if e.Offset%64 != 0 {
				t.Errorf("%v: tensor %s offset %d not 64-aligned", wt, e.Name, e.Offset)
			}
		}
		if l.ExpertBlobSize%ExpertAlign != 0 {
			t.Errorf("%v: expert blob size %d not %d-aligned", wt, l.ExpertBlobSize, ExpertAlign)
		}
		if l.Down+l.MatBytes(32, 64) > l.ExpertBlobSize {
			t.Errorf("%v: expert down matrix overflows blob", wt)
		}
	}
}

// fakeSource returns constant tensors so the written file is easy to check.
type fakeSource struct{ cfg Config }

func (f fakeSource) Config() Config { return f.cfg }
func (f fakeSource) Dense(name string, n int) ([]float32, error) {
	return fill(n, 0.5), nil
}
func (f fakeSource) Expert(layer, e int) (gate, up, down []float32, err error) {
	c := f.cfg
	v := float32(layer*100 + e)
	return fill(c.ExpertHidden*c.Dim, v), fill(c.ExpertHidden*c.Dim, v), fill(c.Dim*c.ExpertHidden, v), nil
}

func fill(n int, v float32) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = v
	}
	return x
}

func TestWriteProducesExpertsAtComputedOffsets(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.mia")
	cfg := tinyConfig()
	if err := WriteFile(path, fakeSource{cfg}, F32, nil); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var h Header
	if err := binary.Read(bytes.NewReader(data), binary.LittleEndian, &h); err != nil {
		t.Fatal(err)
	}
	if string(h.Magic[:]) != "MIA1" || h.NExperts != 8 {
		t.Fatalf("bad header: %+v", h)
	}
	off := h.ExpertsOffset + uint64(1*8+5)*h.ExpertBlobSize
	got := binary.LittleEndian.Uint32(data[off:])
	if want := float32bits(105); got != want {
		t.Fatalf("expert (1,5) first value bits = %x, want %x", got, want)
	}
	if uint64(len(data)) != h.ExpertsOffset+16*h.ExpertBlobSize {
		t.Fatalf("file size %d, want %d", len(data), h.ExpertsOffset+16*h.ExpertBlobSize)
	}
}
