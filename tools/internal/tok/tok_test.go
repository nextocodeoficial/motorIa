package tok

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

const tinyJSON = `{
  "added_tokens": [
    {"id": 7, "content": "<|im_end|>", "special": true},
    {"id": 8, "content": "<|im_start|>", "special": true},
    {"id": 9, "content": "<|endoftext|>", "special": true}
  ],
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "c": 2, "Ġ": 3, "ab": 4, "abc": 5, "Ġabc": 6},
    "merges": %s
  }
}`

type parsed struct {
	nVocab, nMerges, nSpecial uint32
	eos, imStart, imEnd       uint32
	pieces                    [][]byte
	merges                    [][3]uint32
	specials                  []uint32
}

func parse(t *testing.T, b []byte) parsed {
	t.Helper()
	if string(b[:4]) != "MTK1" {
		t.Fatalf("magic %q", b[:4])
	}
	u := func(off int) uint32 { return binary.LittleEndian.Uint32(b[off:]) }
	p := parsed{nVocab: u(4), nMerges: u(8), nSpecial: u(12), eos: u(16), imStart: u(20), imEnd: u(24)}
	off := 28
	for i := uint32(0); i < p.nVocab; i++ {
		n := int(binary.LittleEndian.Uint16(b[off:]))
		p.pieces = append(p.pieces, b[off+2:off+2+n])
		off += 2 + n
	}
	for i := uint32(0); i < p.nMerges; i++ {
		p.merges = append(p.merges, [3]uint32{u(off), u(off + 4), u(off + 8)})
		off += 12
	}
	for i := uint32(0); i < p.nSpecial; i++ {
		p.specials = append(p.specials, u(off))
		off += 4
	}
	if off != len(b) {
		t.Fatalf("trailing bytes: parsed %d of %d", off, len(b))
	}
	return p
}

func TestExport(t *testing.T) {
	tests := []struct {
		name   string
		merges string
	}{
		{"merges as strings", `["a b", "ab c", "Ġ abc"]`},
		{"merges as pairs", `[["a","b"], ["ab","c"], ["Ġ","abc"]]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tokenizer.json")
			os.WriteFile(path, []byte(sprintf(tinyJSON, tt.merges)), 0o644)
			blob, err := Export(path)
			if err != nil {
				t.Fatal(err)
			}
			p := parse(t, blob)
			if p.nVocab != 10 || p.nMerges != 3 || p.nSpecial != 3 {
				t.Fatalf("counts %d/%d/%d", p.nVocab, p.nMerges, p.nSpecial)
			}
			if string(p.pieces[3]) != " " || string(p.pieces[6]) != " abc" {
				t.Fatalf("byte-level decode failed: %q %q", p.pieces[3], p.pieces[6])
			}
			if string(p.pieces[7]) != "<|im_end|>" {
				t.Fatalf("special piece %q", p.pieces[7])
			}
			if p.merges[2] != [3]uint32{3, 5, 6} {
				t.Fatalf("merge 2 = %v, want [3 5 6]", p.merges[2])
			}
			if p.eos != 9 || p.imStart != 8 || p.imEnd != 7 {
				t.Fatalf("ids eos=%d start=%d end=%d", p.eos, p.imStart, p.imEnd)
			}
		})
	}
}

func TestByteDecoderCoversAllBytes(t *testing.T) {
	dec := byteDecoder()
	if len(dec) != 256 {
		t.Fatalf("decoder has %d entries, want 256", len(dec))
	}
	if dec['Ġ'] != ' ' || dec['Ċ'] != '\n' || dec['a'] != 'a' {
		t.Fatal("unexpected GPT-2 byte mapping")
	}
}
