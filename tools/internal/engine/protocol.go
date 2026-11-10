// Package engine drives the C `mia-serve` process. It mirrors the line
// protocol implemented in src/app/serve.c:
//
//	-> CHAT <nbytes> <max_tokens> <temperature> <top_p> <think>\n<nbytes>
//	-> RESET\n
//	<- READY k=v ...            once, after the model is loaded
//	<- PROG <done> <total>      one per prompt token read
//	<- TOK <n> <hits> <misses>\n<n raw bytes>
//	<- DONE k=v ...             end of a CHAT
//	<- OK                       end of a RESET
//	<- ERR <code> <message>     the command failed; engine still usable
//
// SIGUSR1 cancels the running CHAT (it then ends with DONE reason=cancel).
package engine

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
	"unicode/utf8"
)

type Event interface{ isEvent() }

type Info struct {
	Layers       int     `json:"layers"`
	Experts      int     `json:"experts"`
	Active       int     `json:"active"`
	Ctx          int     `json:"ctx"`
	Slots        int     `json:"slots"`
	TotalExperts int     `json:"total_experts"`
	WType        string  `json:"wtype"`
	DenseGiB     float64 `json:"dense_gib"`
	CacheGiB     float64 `json:"cache_gib"`
}

type Ready struct{ Info Info }

type Progress struct{ Done, Total int }

type Token struct {
	Raw          []byte // as sent by the engine; may split a UTF-8 rune
	Text         string // complete runes only (set by Process.Chat)
	Hits, Misses uint64 // cumulative expert-cache counters
}

type Done struct {
	Reason             string // stop | length | ctx | cancel
	Prompt, Gen        int
	PrefillSec, GenSec float64
	CtxUsed, CtxMax    int
	Hits, Misses       uint64
}

type ok struct{}

// Error is reported by the engine for a failed command.
type Error struct{ Code, Msg string }

func (e *Error) Error() string { return e.Code + ": " + e.Msg }

func (Ready) isEvent()    {}
func (Progress) isEvent() {}
func (Token) isEvent()    {}
func (Done) isEvent()     {}
func (ok) isEvent()       {}
func (*Error) isEvent()   {}

func kv(fields []string) map[string]string {
	m := make(map[string]string, len(fields))
	for _, f := range fields {
		if k, v, found := strings.Cut(f, "="); found {
			m[k] = v
		}
	}
	return m
}

func atoi(s string) int {
	n, _ := strconv.Atoi(s)
	return n
}

func atou(s string) uint64 {
	n, _ := strconv.ParseUint(s, 10, 64)
	return n
}

func atof(s string) float64 {
	f, _ := strconv.ParseFloat(s, 64)
	return f
}

func readEvent(r *bufio.Reader) (Event, error) {
	line, err := r.ReadString('\n')
	if err != nil {
		return nil, err
	}
	line = strings.TrimSuffix(line, "\n")
	kind, rest, _ := strings.Cut(line, " ")
	f := strings.Fields(rest)
	switch kind {
	case "READY":
		m := kv(f)
		return Ready{Info{
			Layers: atoi(m["layers"]), Experts: atoi(m["experts"]), Active: atoi(m["active"]),
			Ctx: atoi(m["ctx"]), Slots: atoi(m["slots"]), TotalExperts: atoi(m["total"]),
			WType: m["wtype"], DenseGiB: atof(m["dense_gib"]), CacheGiB: atof(m["cache_gib"]),
		}}, nil
	case "PROG":
		if len(f) != 2 {
			break
		}
		return Progress{atoi(f[0]), atoi(f[1])}, nil
	case "TOK":
		if len(f) != 3 {
			break
		}
		raw := make([]byte, atoi(f[0]))
		if _, err := io.ReadFull(r, raw); err != nil {
			return nil, err
		}
		return Token{Raw: raw, Hits: atou(f[1]), Misses: atou(f[2])}, nil
	case "DONE":
		m := kv(f)
		return Done{
			Reason: m["reason"], Prompt: atoi(m["prompt"]), Gen: atoi(m["gen"]),
			PrefillSec: atof(m["prefill_s"]), GenSec: atof(m["gen_s"]),
			CtxUsed: atoi(m["ctx_used"]), CtxMax: atoi(m["ctx_max"]),
			Hits: atou(m["hits"]), Misses: atou(m["misses"]),
		}, nil
	case "OK":
		return ok{}, nil
	case "ERR":
		code, msg, _ := strings.Cut(rest, " ")
		return &Error{Code: code, Msg: msg}, nil
	}
	return nil, fmt.Errorf("engine protocol: unexpected line %q", line)
}

// utf8Joiner holds back a trailing incomplete rune until the next chunk.
type utf8Joiner struct{ pending []byte }

func (j *utf8Joiner) push(b []byte) string {
	buf := append(j.pending, b...)
	cut := len(buf)
	// Look back at most 3 bytes for the start of an unfinished rune.
	for i := len(buf) - 1; i >= 0 && i >= len(buf)-3; i-- {
		if utf8.RuneStart(buf[i]) {
			if !utf8.FullRune(buf[i:]) {
				cut = i
			}
			break
		}
	}
	out := string(buf[:cut])
	j.pending = append([]byte(nil), buf[cut:]...)
	return out
}

func (j *utf8Joiner) flush() string {
	out := string(j.pending)
	j.pending = nil
	return out
}
