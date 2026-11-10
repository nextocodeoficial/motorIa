package engine

import (
	"bufio"
	"strings"
	"testing"
)

func TestReadEvent(t *testing.T) {
	tests := []struct {
		name  string
		input string
		check func(t *testing.T, e Event, err error)
	}{
		{
			name:  "ready line becomes Info",
			input: "READY layers=48 experts=128 active=8 ctx=2048 slots=396 total=6144 wtype=q8_0 dense_gib=1.65 cache_gib=1.96\n",
			check: func(t *testing.T, e Event, err error) {
				r, ok := e.(Ready)
				if err != nil || !ok || r.Info.Layers != 48 || r.Info.Slots != 396 || r.Info.WType != "q8_0" || r.Info.CacheGiB != 1.96 {
					t.Fatalf("got %#v, %v", e, err)
				}
			},
		},
		{
			name:  "token carries raw bytes including newlines",
			input: "TOK 4 10 2\na\nb\n",
			check: func(t *testing.T, e Event, err error) {
				tk, ok := e.(Token)
				if err != nil || !ok || string(tk.Raw) != "a\nb\n" || tk.Hits != 10 || tk.Misses != 2 {
					t.Fatalf("got %#v, %v", e, err)
				}
			},
		},
		{
			name:  "progress",
			input: "PROG 3 30\n",
			check: func(t *testing.T, e Event, err error) {
				if p, ok := e.(Progress); err != nil || !ok || p.Done != 3 || p.Total != 30 {
					t.Fatalf("got %#v, %v", e, err)
				}
			},
		},
		{
			name:  "done with stats",
			input: "DONE reason=length prompt=12 gen=4 prefill_s=1.5 gen_s=2 ctx_used=16 ctx_max=512 hits=7 misses=3\n",
			check: func(t *testing.T, e Event, err error) {
				d, ok := e.(Done)
				if err != nil || !ok || d.Reason != "length" || d.Prompt != 12 || d.CtxUsed != 16 || d.PrefillSec != 1.5 || d.Misses != 3 {
					t.Fatalf("got %#v, %v", e, err)
				}
			},
		},
		{
			name:  "error keeps code and message",
			input: "ERR context_full message needs 60 tokens, 10 left\n",
			check: func(t *testing.T, e Event, err error) {
				x, ok := e.(*Error)
				if err != nil || !ok || x.Code != "context_full" || x.Msg != "message needs 60 tokens, 10 left" {
					t.Fatalf("got %#v, %v", e, err)
				}
			},
		},
		{
			name:  "unknown line is a protocol error",
			input: "WAT\n",
			check: func(t *testing.T, e Event, err error) {
				if err == nil {
					t.Fatal("expected error")
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, err := readEvent(bufio.NewReader(strings.NewReader(tt.input)))
			tt.check(t, e, err)
		})
	}
}

func TestUTF8Joiner(t *testing.T) {
	tests := []struct {
		name   string
		chunks [][]byte
		want   []string
	}{
		{"ascii passes through", [][]byte{[]byte("ab"), []byte("c")}, []string{"ab", "c"}},
		{"split rune is held until complete", [][]byte{{0xC3}, {0xA1, 'x'}}, []string{"", "áx"}},
		{"three byte rune across three chunks", [][]byte{{0xE2}, {0x82}, {0xAC}}, []string{"", "", "€"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var j utf8Joiner
			for i, c := range tt.chunks {
				if got := j.push(c); got != tt.want[i] {
					t.Fatalf("chunk %d: got %q, want %q", i, got, tt.want[i])
				}
			}
		})
	}
}
