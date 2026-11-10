// Package web is the HTTP adapter: a JSON + Server-Sent Events API over
// chat.Service, plus the static Vue app.
package web

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"strings"

	"motoria/tools/internal/chat"
	"motoria/tools/internal/engine"
)

type chatRequest struct {
	Message     string   `json:"message"`
	Temperature *float32 `json:"temperature"`
	TopP        *float32 `json:"top_p"`
	MaxTokens   *int     `json:"max_tokens"`
	Think       bool     `json:"think"`
}

func (r chatRequest) params() (engine.Params, error) {
	p := engine.Params{MaxTokens: 1024, Temperature: 0.7, TopP: 0.8, Think: r.Think}
	if r.Temperature != nil {
		p.Temperature = *r.Temperature
	}
	if r.TopP != nil {
		p.TopP = *r.TopP
	}
	if r.MaxTokens != nil {
		p.MaxTokens = *r.MaxTokens
	}
	switch {
	case p.Temperature < 0 || p.Temperature > 2:
		return p, errors.New("temperature must be between 0 and 2")
	case p.TopP <= 0 || p.TopP > 1:
		return p, errors.New("top_p must be greater than 0 and at most 1")
	case p.MaxTokens < 1 || p.MaxTokens > 8192:
		return p, errors.New("max_tokens must be between 1 and 8192")
	}
	return p, nil
}

// Handler serves the API under /api/ and the single-page app otherwise.
func Handler(svc *chat.Service, static fs.FS) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/status", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, svc.Status())
	})
	mux.HandleFunc("GET /api/messages", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, svc.Messages())
	})
	mux.HandleFunc("POST /api/reset", func(w http.ResponseWriter, r *http.Request) {
		if err := svc.Reset(); err != nil {
			writeError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, svc.Status())
	})
	mux.HandleFunc("POST /api/stop", func(w http.ResponseWriter, r *http.Request) {
		svc.Stop()
		w.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("POST /api/chat", func(w http.ResponseWriter, r *http.Request) {
		handleChat(svc, w, r)
	})
	mux.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "no such endpoint"})
	})
	mux.Handle("/", spa(static))
	return mux
}

func handleChat(svc *chat.Service, w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "request body must be JSON"})
		return
	}
	if strings.TrimSpace(req.Message) == "" {
		writeError(w, chat.ErrEmpty)
		return
	}
	p, err := req.params()
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if svc.Status().Busy {
		writeError(w, engine.ErrBusy)
		return
	}

	flusher, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	send := func(event string, v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
		if flusher != nil {
			flusher.Flush()
		}
	}

	// r.Context() ends when the browser disconnects, which cancels the turn.
	msg, err := svc.Send(r.Context(), req.Message, p, func(e engine.Event) {
		switch e := e.(type) {
		case engine.Progress:
			send("progress", map[string]int{"done": e.Done, "total": e.Total})
		case engine.Token:
			send("token", map[string]any{"text": e.Text, "hits": e.Hits, "misses": e.Misses})
		}
	})
	if err != nil {
		code := "internal"
		var ee *engine.Error
		switch {
		case errors.As(err, &ee):
			code = ee.Code
		case errors.Is(err, engine.ErrBusy):
			code = "busy"
		}
		send("error", map[string]string{"code": code, "message": err.Error()})
		return
	}
	st := svc.Status()
	send("done", map[string]any{"message": msg, "ctx_used": st.CtxUsed, "ctx_max": st.CtxMax})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	switch {
	case errors.Is(err, chat.ErrEmpty):
		status = http.StatusBadRequest
	case errors.Is(err, engine.ErrBusy):
		status = http.StatusConflict
	}
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

// spa serves files from static and falls back to index.html for routes.
func spa(static fs.FS) http.Handler {
	files := http.FileServer(http.FS(static))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(r.URL.Path, "/")
		if name != "" {
			if _, err := fs.Stat(static, name); err == nil {
				files.ServeHTTP(w, r)
				return
			}
		}
		index, err := fs.ReadFile(static, "index.html")
		if err != nil {
			http.Error(w, "web app not built: run `make web`", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(index)
	})
}
