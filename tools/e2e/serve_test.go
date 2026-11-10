package e2e

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"motoria/tools/internal/engine"
	"motoria/tools/internal/mia"
	"motoria/tools/internal/ref"
	"motoria/tools/internal/tok"
)

// chatTokenizerJSON: ids 0..91 are '!'..'|', 92 space, 93 newline,
// 94 <|im_start|>, 95 <|im_end|>; enough to encode the Qwen3 chat template.
func chatTokenizerJSON(t *testing.T) string {
	t.Helper()
	vocab := map[string]int{}
	for i := 0; i < 92; i++ {
		vocab[string(rune(byte(33+i)))] = i
	}
	vocab["Ġ"], vocab["Ċ"] = 92, 93
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	js := `{"added_tokens":[{"id":94,"content":"<|im_start|>","special":true},` +
		`{"id":95,"content":"<|im_end|>","special":true}],` +
		`"model":{"type":"BPE","vocab":` + mustJSON(vocab) + `,"merges":[]}}`
	if err := os.WriteFile(path, []byte(js), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func startServe(t *testing.T, ctxLen string) *engine.Process {
	t.Helper()
	bin := os.Getenv("MIA_SERVE")
	if testing.Short() || bin == "" {
		t.Skip("set MIA_SERVE to the mia-serve binary (make e2e)")
	}
	blob, err := tok.Export(chatTokenizerJSON(t))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "chat.mia")
	if err := mia.WriteFile(path, ref.Random(tinyCfg(), 5), mia.Q80, blob); err != nil {
		t.Fatal(err)
	}
	p, err := engine.Start(bin, "-m", path, "--ram", "64M", "-c", ctxLen, "--seed", "1")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { p.Close() })
	return p
}

func TestServeKeepsContextAcrossTurns(t *testing.T) {
	p := startServe(t, "512")
	if info := p.Info(); info.Ctx != 512 || info.Layers != 2 || info.Experts != 8 {
		t.Fatalf("info = %+v", info)
	}
	greedy := engine.Params{MaxTokens: 4, Temperature: 0, TopP: 1}

	var progress, tokens int
	first, err := p.Chat(context.Background(), "hi", greedy, func(e engine.Event) {
		switch e.(type) {
		case engine.Progress:
			progress++
		case engine.Token:
			tokens++
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if progress != first.Prompt || first.Prompt == 0 {
		t.Fatalf("progress events %d, prompt tokens %d", progress, first.Prompt)
	}
	if tokens != first.Gen || first.Gen > 4 {
		t.Fatalf("token events %d, gen %d", tokens, first.Gen)
	}
	if first.Reason != "length" && first.Reason != "stop" {
		t.Fatalf("reason %q", first.Reason)
	}

	second, err := p.Chat(context.Background(), "ok", greedy, func(engine.Event) {})
	if err != nil {
		t.Fatal(err)
	}
	if second.CtxUsed <= first.CtxUsed {
		t.Fatalf("context did not grow: %d then %d", first.CtxUsed, second.CtxUsed)
	}
	// Only the new turn is prefilled, not the whole history.
	if second.Prompt >= second.CtxUsed {
		t.Fatalf("second turn re-read history: prompt %d, ctx %d", second.Prompt, second.CtxUsed)
	}

	if err := p.Reset(); err != nil {
		t.Fatal(err)
	}
	again, err := p.Chat(context.Background(), "hi", greedy, func(engine.Event) {})
	if err != nil {
		t.Fatal(err)
	}
	if again.CtxUsed != first.CtxUsed {
		t.Fatalf("after reset ctx %d, want %d (fresh conversation)", again.CtxUsed, first.CtxUsed)
	}
}

func TestServeRejectsMessageThatOverflowsContext(t *testing.T) {
	p := startServe(t, "48")
	long := make([]byte, 200)
	for i := range long {
		long[i] = 'a'
	}
	_, err := p.Chat(context.Background(), string(long), engine.Params{MaxTokens: 4, TopP: 1}, func(engine.Event) {})
	var ee *engine.Error
	if !errors.As(err, &ee) || ee.Code != "context_full" {
		t.Fatalf("err = %v, want context_full", err)
	}
	// The engine stays usable after a rejected message.
	if _, err := p.Chat(context.Background(), "hi", engine.Params{MaxTokens: 2, TopP: 1}, func(engine.Event) {}); err != nil {
		t.Fatalf("engine unusable after rejection: %v", err)
	}
}
