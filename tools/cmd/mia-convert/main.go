// mia-convert turns a Hugging Face Qwen3-MoE checkpoint directory into a
// single .mia file for the C engine.
//
//	go run ./cmd/mia-convert -in ~/models/Qwen3-30B-A3B -out qwen3-30b-a3b-q8.mia -type q8_0
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"motoria/tools/internal/hf"
	"motoria/tools/internal/mia"
	"motoria/tools/internal/tok"
)

// progress wraps a Source to report expert conversion progress.
type progress struct {
	mia.Source
	start time.Time
	total int
	done  int
}

func (p *progress) Expert(l, e int) (g, u, d []float32, err error) {
	p.done++
	if p.done%64 == 0 || p.done == p.total {
		el := time.Since(p.start)
		eta := time.Duration(float64(el) / float64(p.done) * float64(p.total-p.done))
		fmt.Fprintf(os.Stderr, "\rexperts %d/%d  elapsed %s  eta %s   ", p.done, p.total,
			el.Round(time.Second), eta.Round(time.Second))
	}
	return p.Source.Expert(l, e)
}

func main() {
	in := flag.String("in", "", "Hugging Face checkpoint directory")
	out := flag.String("out", "", "output .mia file")
	typ := flag.String("type", "q8_0", "weight type: q8_0 or f32")
	flag.Parse()
	if *in == "" || *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	wt := mia.Q80
	switch *typ {
	case "q8_0":
	case "f32":
		wt = mia.F32
	default:
		fmt.Fprintf(os.Stderr, "unknown -type %q\n", *typ)
		os.Exit(2)
	}

	ck, err := hf.Open(*in)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open checkpoint:", err)
		os.Exit(1)
	}
	defer ck.Close()

	var tokBlob []byte
	if tj := filepath.Join(*in, "tokenizer.json"); fileExists(tj) {
		if tokBlob, err = tok.Export(tj); err != nil {
			fmt.Fprintln(os.Stderr, "tokenizer:", err)
			os.Exit(1)
		}
	} else {
		fmt.Fprintln(os.Stderr, "warning: no tokenizer.json, engine will need token ids")
	}

	c := ck.Config()
	l := mia.NewLayout(c, wt)
	total := c.NLayers * c.NExperts
	fmt.Fprintf(os.Stderr, "model: %d layers, %d experts (top-%d), dim %d, vocab %d\n",
		c.NLayers, c.NExperts, c.NActive, c.Dim, c.Vocab)
	fmt.Fprintf(os.Stderr, "dense (resident): %.2f GB, experts (streamed): %.2f GB as %s\n",
		gb(l.DenseSize), gb(l.ExpertBlobSize*total), wt)

	src := &progress{Source: ck, start: time.Now(), total: total}
	if err := mia.WriteFile(*out, src, wt, tokBlob); err != nil {
		fmt.Fprintln(os.Stderr, "\nwrite:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "\nwrote %s\n", *out)
}

func gb(n int) float64 { return float64(n) / (1 << 30) }

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}
