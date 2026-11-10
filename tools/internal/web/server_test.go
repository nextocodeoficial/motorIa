package web

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/fstest"

	"motoria/tools/internal/chat"
	"motoria/tools/internal/engine"
)

type stubEngine struct{ err error }

func (stubEngine) Info() engine.Info {
	return engine.Info{Ctx: 64, Layers: 48, Experts: 128, Active: 8, Slots: 396, TotalExperts: 6144}
}
func (s stubEngine) Chat(ctx context.Context, text string, p engine.Params, emit func(engine.Event)) (engine.Done, error) {
	if s.err != nil {
		return engine.Done{}, s.err
	}
	emit(engine.Progress{Done: 1, Total: 2})
	emit(engine.Progress{Done: 2, Total: 2})
	emit(engine.Token{Text: "¡Hola!", Hits: 4, Misses: 4})
	return engine.Done{Reason: "stop", Prompt: 2, Gen: 1, GenSec: 1, CtxUsed: 3, CtxMax: 64, Hits: 4, Misses: 4}, nil
}
func (stubEngine) Reset() error  { return nil }
func (stubEngine) Cancel() error { return nil }

func newServer(eng chat.Engine) *httptest.Server {
	static := fstest.MapFS{"index.html": {Data: []byte("<!doctype html><title>mia</title>")}}
	return httptest.NewServer(Handler(chat.NewService(eng), static))
}

type sseEvent struct{ name, data string }

func readSSE(t *testing.T, resp *http.Response) []sseEvent {
	t.Helper()
	var out []sseEvent
	var cur sseEvent
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			cur.name = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			cur.data = strings.TrimPrefix(line, "data: ")
		case line == "":
			out = append(out, cur)
			cur = sseEvent{}
		}
	}
	return out
}

func post(t *testing.T, url, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestChatStreamsProgressTokensAndDone(t *testing.T) {
	srv := newServer(stubEngine{})
	defer srv.Close()
	resp := post(t, srv.URL+"/api/chat", `{"message":"hola","temperature":0}`)
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type %q", ct)
	}
	evs := readSSE(t, resp)
	var names []string
	for _, e := range evs {
		names = append(names, e.name)
	}
	if got := strings.Join(names, ","); got != "progress,progress,token,done" {
		t.Fatalf("events %s", got)
	}
	var tok struct{ Text string }
	json.Unmarshal([]byte(evs[2].data), &tok)
	if tok.Text != "¡Hola!" {
		t.Fatalf("token %q", evs[2].data)
	}
	var done struct {
		Message chat.Message `json:"message"`
		CtxUsed int          `json:"ctx_used"`
	}
	json.Unmarshal([]byte(evs[3].data), &done)
	if done.Message.Content != "¡Hola!" || done.CtxUsed != 3 {
		t.Fatalf("done %s", evs[3].data)
	}

	// History endpoint now has both turns.
	r, _ := http.Get(srv.URL + "/api/messages")
	var msgs []chat.Message
	json.NewDecoder(r.Body).Decode(&msgs)
	if len(msgs) != 2 {
		t.Fatalf("messages %+v", msgs)
	}
}

func TestChatValidation(t *testing.T) {
	srv := newServer(stubEngine{})
	defer srv.Close()
	tests := []struct {
		name string
		body string
	}{
		{"empty message", `{"message":"   "}`},
		{"bad json", `{`},
		{"temperature out of range", `{"message":"x","temperature":9}`},
		{"top_p out of range", `{"message":"x","top_p":1.5}`},
		{"max_tokens out of range", `{"message":"x","max_tokens":999999}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := post(t, srv.URL+"/api/chat", tt.body)
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status %d", resp.StatusCode)
			}
		})
	}
}

func TestEngineErrorBecomesErrorEvent(t *testing.T) {
	srv := newServer(stubEngine{err: &engine.Error{Code: "context_full", Msg: "no room left"}})
	defer srv.Close()
	resp := post(t, srv.URL+"/api/chat", `{"message":"hola"}`)
	defer resp.Body.Close()
	evs := readSSE(t, resp)
	if len(evs) != 1 || evs[0].name != "error" || !strings.Contains(evs[0].data, "context_full") {
		t.Fatalf("events %+v", evs)
	}
}

func TestStatusAndStaticFallback(t *testing.T) {
	srv := newServer(stubEngine{})
	defer srv.Close()
	r, _ := http.Get(srv.URL + "/api/status")
	var st struct {
		Model  struct{ Layers, Slots int }
		CtxMax int `json:"ctx_max"`
	}
	json.NewDecoder(r.Body).Decode(&st)
	if st.Model.Layers != 48 || st.CtxMax != 64 {
		t.Fatalf("status %+v", st)
	}
	for _, path := range []string{"/", "/some/client/route"} {
		r, _ := http.Get(srv.URL + path)
		if r.StatusCode != 200 || r.Header.Get("Content-Type") != "text/html; charset=utf-8" {
			t.Fatalf("%s: %d %s", path, r.StatusCode, r.Header.Get("Content-Type"))
		}
	}
	r, _ = http.Get(srv.URL + "/api/nope")
	if r.StatusCode != http.StatusNotFound {
		t.Fatalf("unknown api route: %d", r.StatusCode)
	}
}
