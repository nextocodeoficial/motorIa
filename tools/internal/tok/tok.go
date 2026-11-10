// Package tok converts a Hugging Face byte-level BPE tokenizer.json into the
// compact binary blob read by src/tokenizer/bpe.c:
//
//	"MTK1" u32 nVocab u32 nMerges u32 nSpecial u32 eos u32 imStart u32 imEnd
//	nVocab  x (u16 len, raw bytes)       piece bytes per id (empty if unused)
//	nMerges x (u32 left, u32 right, u32 result), in priority order
//	nSpecial x u32 id                     added tokens matched verbatim
package tok

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const none = ^uint32(0)

func sprintf(f string, a ...any) string { return fmt.Sprintf(f, a...) }

// byteDecoder is the inverse of GPT-2's bytes_to_unicode table.
func byteDecoder() map[rune]byte {
	var bs []int
	for b := '!'; b <= '~'; b++ {
		bs = append(bs, int(b))
	}
	for b := '¡'; b <= '¬'; b++ {
		bs = append(bs, int(b))
	}
	for b := '®'; b <= 'ÿ'; b++ {
		bs = append(bs, int(b))
	}
	cs := append([]int(nil), bs...)
	in := map[int]bool{}
	for _, b := range bs {
		in[b] = true
	}
	n := 0
	for b := 0; b < 256; b++ {
		if !in[b] {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	dec := make(map[rune]byte, 256)
	for i := range bs {
		dec[rune(cs[i])] = byte(bs[i])
	}
	return dec
}

type hfTokenizer struct {
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
	Model struct {
		Type   string            `json:"type"`
		Vocab  map[string]int    `json:"vocab"`
		Merges []json.RawMessage `json:"merges"`
	} `json:"model"`
}

func Export(path string) ([]byte, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var t hfTokenizer
	if err := json.Unmarshal(raw, &t); err != nil {
		return nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	if t.Model.Type != "BPE" {
		return nil, fmt.Errorf("tokenizer model %q not supported (want BPE)", t.Model.Type)
	}
	dec := byteDecoder()

	nVocab := 0
	for _, id := range t.Model.Vocab {
		nVocab = max(nVocab, id+1)
	}
	for _, a := range t.AddedTokens {
		nVocab = max(nVocab, a.ID+1)
	}
	pieces := make([][]byte, nVocab)
	for s, id := range t.Model.Vocab {
		b := make([]byte, 0, len(s))
		for _, r := range s {
			v, ok := dec[r]
			if !ok {
				return nil, fmt.Errorf("vocab entry %q is not byte-level encoded", s)
			}
			b = append(b, v)
		}
		pieces[id] = b
	}
	special := map[string]uint32{}
	var specials []uint32
	for _, a := range t.AddedTokens {
		pieces[a.ID] = []byte(a.Content)
		special[a.Content] = uint32(a.ID)
		specials = append(specials, uint32(a.ID))
	}

	var merges [][3]uint32
	for i, m := range t.Model.Merges {
		var l, r string
		var pair []string
		var s string
		if json.Unmarshal(m, &s) == nil {
			parts := strings.SplitN(s, " ", 2)
			if len(parts) != 2 {
				return nil, fmt.Errorf("merge %d: %q", i, s)
			}
			l, r = parts[0], parts[1]
		} else if json.Unmarshal(m, &pair) == nil && len(pair) == 2 {
			l, r = pair[0], pair[1]
		} else {
			return nil, fmt.Errorf("merge %d: unsupported format", i)
		}
		li, ok1 := t.Model.Vocab[l]
		ri, ok2 := t.Model.Vocab[r]
		res, ok3 := t.Model.Vocab[l+r]
		if !ok1 || !ok2 || !ok3 {
			return nil, fmt.Errorf("merge %d (%q %q) references unknown tokens", i, l, r)
		}
		merges = append(merges, [3]uint32{uint32(li), uint32(ri), uint32(res)})
	}

	id := func(s string) uint32 {
		if v, ok := special[s]; ok {
			return v
		}
		return none
	}
	le := binary.LittleEndian
	out := []byte("MTK1")
	for _, v := range []uint32{uint32(nVocab), uint32(len(merges)), uint32(len(specials)),
		id("<|endoftext|>"), id("<|im_start|>"), id("<|im_end|>")} {
		out = le.AppendUint32(out, v)
	}
	for _, p := range pieces {
		if len(p) > 0xffff {
			return nil, fmt.Errorf("token piece too long")
		}
		out = le.AppendUint16(out, uint16(len(p)))
		out = append(out, p...)
	}
	for _, m := range merges {
		out = le.AppendUint32(le.AppendUint32(le.AppendUint32(out, m[0]), m[1]), m[2])
	}
	for _, s := range specials {
		out = le.AppendUint32(out, s)
	}
	return out, nil
}
