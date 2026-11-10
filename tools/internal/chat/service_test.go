package chat

import (
	"context"
	"errors"
	"testing"

	"motoria/tools/internal/engine"
)

// fakeEngine replies with fixed tokens and records calls.
type fakeEngine struct {
	reply     []string
	err       error
	resets    int
	cancelled int
	hits      uint64
	ctxUsed   int
	block     chan struct{} // if set, Chat waits for it
}

func (f *fakeEngine) Info() engine.Info { return engine.Info{Ctx: 100, Layers: 2} }

func (f *fakeEngine) Chat(ctx context.Context, text string, p engine.Params, emit func(engine.Event)) (engine.Done, error) {
	if f.block != nil {
		<-f.block
	}
	if f.err != nil {
		return engine.Done{}, f.err
	}
	emit(engine.Progress{Done: 1, Total: 1})
	for _, t := range f.reply {
		f.hits += 3
		emit(engine.Token{Text: t, Hits: f.hits, Misses: 1})
	}
	f.ctxUsed += 10
	return engine.Done{Reason: "stop", Prompt: 5, Gen: len(f.reply), GenSec: 2,
		CtxUsed: f.ctxUsed, CtxMax: 100, Hits: f.hits, Misses: 1}, nil
}

func (f *fakeEngine) Reset() error  { f.resets++; f.ctxUsed = 0; return nil }
func (f *fakeEngine) Cancel() error { f.cancelled++; return nil }

func TestSendRecordsBothTurnsWithStats(t *testing.T) {
	eng := &fakeEngine{reply: []string{"Hola", ", ", "mundo"}}
	s := NewService(eng)
	var streamed string
	msg, err := s.Send(context.Background(), "hola", engine.Params{MaxTokens: 10}, func(e engine.Event) {
		if tk, ok := e.(engine.Token); ok {
			streamed += tk.Text
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content != "Hola, mundo" || streamed != msg.Content {
		t.Fatalf("content %q, streamed %q", msg.Content, streamed)
	}
	if msg.Stats == nil || msg.Stats.GenTokens != 3 || msg.Stats.TokensPerSec != 1.5 {
		t.Fatalf("stats %+v", msg.Stats)
	}
	msgs := s.Messages()
	if len(msgs) != 2 || msgs[0].Role != "user" || msgs[1].Role != "assistant" {
		t.Fatalf("messages %+v", msgs)
	}
	if st := s.Status(); st.CtxUsed != 10 || st.CtxMax != 100 || st.Busy {
		t.Fatalf("status %+v", st)
	}
}

func TestSendRejectsBlankMessage(t *testing.T) {
	s := NewService(&fakeEngine{})
	if _, err := s.Send(context.Background(), "  \n", engine.Params{}, func(engine.Event) {}); !errors.Is(err, ErrEmpty) {
		t.Fatalf("err = %v, want ErrEmpty", err)
	}
}

func TestContextFullLeavesHistoryUntouched(t *testing.T) {
	eng := &fakeEngine{err: &engine.Error{Code: "context_full", Msg: "no room"}}
	s := NewService(eng)
	_, err := s.Send(context.Background(), "hola", engine.Params{}, func(engine.Event) {})
	var ee *engine.Error
	if !errors.As(err, &ee) || ee.Code != "context_full" {
		t.Fatalf("err = %v", err)
	}
	if len(s.Messages()) != 0 {
		t.Fatal("rejected message was stored")
	}
}

func TestEngineFailureClearsHistory(t *testing.T) {
	eng := &fakeEngine{reply: []string{"ok"}}
	s := NewService(eng)
	s.Send(context.Background(), "uno", engine.Params{}, func(engine.Event) {})
	eng.err = &engine.Error{Code: "engine", Msg: "conversation was reset"}
	s.Send(context.Background(), "dos", engine.Params{}, func(engine.Event) {})
	if n := len(s.Messages()); n != 0 {
		t.Fatalf("history has %d messages, engine already dropped them", n)
	}
}

func TestOneMessageAtATime(t *testing.T) {
	eng := &fakeEngine{reply: []string{"x"}, block: make(chan struct{})}
	s := NewService(eng)
	done := make(chan error)
	go func() {
		_, err := s.Send(context.Background(), "uno", engine.Params{}, func(engine.Event) {})
		done <- err
	}()
	for !s.Status().Busy {
	}
	if _, err := s.Send(context.Background(), "dos", engine.Params{}, func(engine.Event) {}); !errors.Is(err, engine.ErrBusy) {
		t.Fatalf("err = %v, want ErrBusy", err)
	}
	if err := s.Reset(); !errors.Is(err, engine.ErrBusy) {
		t.Fatalf("reset while busy: %v", err)
	}
	s.Stop()
	if eng.cancelled != 1 {
		t.Fatal("stop did not cancel the engine")
	}
	close(eng.block)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestResetClearsEverything(t *testing.T) {
	eng := &fakeEngine{reply: []string{"x"}}
	s := NewService(eng)
	s.Send(context.Background(), "uno", engine.Params{}, func(engine.Event) {})
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}
	if len(s.Messages()) != 0 || s.Status().CtxUsed != 0 || eng.resets != 1 {
		t.Fatal("reset left state behind")
	}
}
