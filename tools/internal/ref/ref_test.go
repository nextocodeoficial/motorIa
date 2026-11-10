package ref

import (
	"math"
	"testing"

	"motoria/tools/internal/mia"
)

func tinyCfg() mia.Config {
	return mia.Config{
		Dim: 64, NLayers: 2, NHeads: 4, NKVHeads: 2, HeadDim: 16, Vocab: 96,
		NExperts: 8, NActive: 2, ExpertHidden: 32, NormTopK: true,
		RopeTheta: 1e6, NormEps: 1e-6,
	}
}

func TestRMSNormUnitWeightGivesUnitRMS(t *testing.T) {
	x := []float32{1, -2, 3, 4}
	out := rmsnorm(x, []float32{1, 1, 1, 1}, 0)
	var ss float64
	for _, v := range out {
		ss += float64(v) * float64(v)
	}
	if rms := math.Sqrt(ss / 4); math.Abs(rms-1) > 1e-5 {
		t.Fatalf("rms = %v, want 1", rms)
	}
}

func TestTopKRouting(t *testing.T) {
	tests := []struct {
		name     string
		logits   []float32
		k        int
		norm     bool
		wantIdx  []int
		wantSum1 bool
	}{
		{"picks largest in order, renormalized", []float32{0, 3, 1, 2}, 2, true, []int{1, 3}, true},
		{"without renorm weights are raw softmax", []float32{0, 3, 1, 2}, 2, false, []int{1, 3}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, w := route(tt.logits, tt.k, tt.norm)
			for i := range tt.wantIdx {
				if idx[i] != tt.wantIdx[i] {
					t.Fatalf("idx = %v, want %v", idx, tt.wantIdx)
				}
			}
			sum := w[0] + w[1]
			if got := math.Abs(float64(sum)-1) < 1e-6; got != tt.wantSum1 {
				t.Fatalf("weights %v sum %v, renorm expectation %v", w, sum, tt.wantSum1)
			}
		})
	}
}

func TestRopePreservesNorm(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	before := norm(x)
	rope(x, 8, 17, 1e4)
	if math.Abs(before-norm(x)) > 1e-4 {
		t.Fatalf("rope changed the vector norm")
	}
}

func norm(x []float32) float64 {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s)
}

func TestIdenticalExpertsMatchAnyRouting(t *testing.T) {
	cfg := tinyCfg()
	m := Random(cfg, 1)
	for l := range m.Experts {
		for e := range m.Experts[l] {
			m.Experts[l][e] = m.Experts[l][0]
		}
	}
	a := NewSession(m).Step(5, 0)
	// Different router, same experts: output must not change (weights sum to 1).
	for l := range m.Layers {
		for i := range m.Layers[l].Router {
			m.Layers[l].Router[i] *= -3
		}
	}
	b := NewSession(m).Step(5, 0)
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > 1e-4 {
			t.Fatalf("logit %d differs: %v vs %v", i, a[i], b[i])
		}
	}
}

func TestPrefixLogitsAreCausal(t *testing.T) {
	m := Random(tinyCfg(), 2)
	s1 := NewSession(m)
	s1.Step(3, 0)
	x := s1.Step(7, 1)
	s2 := NewSession(m)
	s2.Step(3, 0)
	y := s2.Step(7, 1)
	s2.Step(9, 2)
	for i := range x {
		if x[i] != y[i] {
			t.Fatal("logits at position 1 are not deterministic")
		}
	}
}
