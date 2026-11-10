// Package chat holds the single conversation the engine keeps in its KV
// cache. It is the application layer between the HTTP adapter (web) and
// the engine port.
package chat

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"

	"motoria/tools/internal/engine"
)

// Engine is the port implemented by *engine.Process.
type Engine interface {
	Info() engine.Info
	Chat(ctx context.Context, text string, p engine.Params, emit func(engine.Event)) (engine.Done, error)
	Reset() error
	Cancel() error
}

var ErrEmpty = errors.New("message is empty")

type Stats struct {
	PromptTokens int     `json:"prompt_tokens"`
	GenTokens    int     `json:"gen_tokens"`
	PrefillSec   float64 `json:"prefill_sec"`
	GenSec       float64 `json:"gen_sec"`
	TokensPerSec float64 `json:"tokens_per_sec"`
	HitRate      float64 `json:"hit_rate"` // expert cache, this turn, 0..1
	Reason       string  `json:"reason"`   // stop | length | ctx | cancel
}

type Message struct {
	Role    string `json:"role"` // user | assistant
	Content string `json:"content"`
	Stats   *Stats `json:"stats,omitempty"`
}

type Status struct {
	Name    string      `json:"name"`
	Model   engine.Info `json:"model"`
	CtxUsed int         `json:"ctx_used"`
	CtxMax  int         `json:"ctx_max"`
	Busy    bool        `json:"busy"`
}

type Service struct {
	eng  Engine
	name string
	busy atomic.Bool

	mu                   sync.Mutex
	msgs                 []Message
	ctxUsed              int
	lastHits, lastMisses uint64
}

func NewService(eng Engine) *Service { return &Service{eng: eng} }

// WithName sets the model name shown to clients.
func (s *Service) WithName(name string) *Service {
	s.name = name
	return s
}

// Send runs one turn. emit receives engine events as they stream.
func (s *Service) Send(ctx context.Context, text string, p engine.Params, emit func(engine.Event)) (Message, error) {
	if strings.TrimSpace(text) == "" {
		return Message{}, ErrEmpty
	}
	if !s.busy.CompareAndSwap(false, true) {
		return Message{}, engine.ErrBusy
	}
	defer s.busy.Store(false)

	var reply strings.Builder
	done, err := s.eng.Chat(ctx, text, p, func(e engine.Event) {
		if t, ok := e.(engine.Token); ok {
			reply.WriteString(t.Text)
		}
		emit(e)
	})

	s.mu.Lock()
	defer s.mu.Unlock()
	if err != nil {
		var ee *engine.Error
		if errors.As(err, &ee) && ee.Code == "engine" {
			s.msgs, s.ctxUsed = nil, 0 // the engine dropped its context
		}
		return Message{}, err
	}

	st := &Stats{
		PromptTokens: done.Prompt, GenTokens: done.Gen,
		PrefillSec: done.PrefillSec, GenSec: done.GenSec, Reason: done.Reason,
	}
	if done.GenSec > 0 {
		st.TokensPerSec = float64(done.Gen) / done.GenSec
	}
	dh, dm := done.Hits-s.lastHits, done.Misses-s.lastMisses
	if dh+dm > 0 {
		st.HitRate = float64(dh) / float64(dh+dm)
	}
	s.lastHits, s.lastMisses = done.Hits, done.Misses
	s.ctxUsed = done.CtxUsed

	answer := Message{Role: "assistant", Content: reply.String(), Stats: st}
	s.msgs = append(s.msgs, Message{Role: "user", Content: text}, answer)
	return answer, nil
}

// Stop cancels the running turn, if any.
func (s *Service) Stop() {
	if s.busy.Load() {
		s.eng.Cancel()
	}
}

// Reset starts a new conversation.
func (s *Service) Reset() error {
	if !s.busy.CompareAndSwap(false, true) {
		return engine.ErrBusy
	}
	defer s.busy.Store(false)
	if err := s.eng.Reset(); err != nil {
		return err
	}
	s.mu.Lock()
	s.msgs, s.ctxUsed = nil, 0
	s.mu.Unlock()
	return nil
}

func (s *Service) Messages() []Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Message{}, s.msgs...)
}

func (s *Service) Status() Status {
	s.mu.Lock()
	defer s.mu.Unlock()
	info := s.eng.Info()
	return Status{Name: s.name, Model: info, CtxUsed: s.ctxUsed, CtxMax: info.Ctx, Busy: s.busy.Load()}
}
