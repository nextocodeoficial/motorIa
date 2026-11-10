// mia-web serves a browser chat UI for a .mia model.
//
//	mia-web -model qwen3-30b-a3b-q8.mia -ram 6G
//
// It starts mia-serve (the C engine) as a child process, keeps it loaded,
// and serves the embedded Vue app plus a JSON/SSE API on -addr.
package main

import (
	"context"
	"embed"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"motoria/tools/internal/chat"
	"motoria/tools/internal/engine"
	"motoria/tools/internal/web"
)

//go:embed all:dist
var dist embed.FS

func findEngine(flagValue string) (string, error) {
	if flagValue != "" {
		return flagValue, nil
	}
	if exe, err := os.Executable(); err == nil {
		sibling := filepath.Join(filepath.Dir(exe), "mia-serve")
		if _, err := os.Stat(sibling); err == nil {
			return sibling, nil
		}
	}
	if p, err := exec.LookPath("mia-serve"); err == nil {
		return p, nil
	}
	return "", errors.New("mia-serve not found next to mia-web or in PATH; pass -engine")
}

func main() {
	model := flag.String("model", "", "path to the .mia model file")
	ram := flag.String("ram", "6G", "RAM budget for the engine, e.g. 4G")
	ctxLen := flag.Int("ctx", 4096, "conversation length in tokens")
	addr := flag.String("addr", "127.0.0.1:8080", "listen address")
	enginePath := flag.String("engine", "", "path to mia-serve (default: next to this binary)")
	flag.Parse()
	if *model == "" {
		flag.Usage()
		os.Exit(2)
	}
	bin, err := findEngine(*enginePath)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("loading %s (RAM budget %s)…", filepath.Base(*model), *ram)
	proc, err := engine.Start(bin, "-m", *model, "--ram", *ram, "-c", strconv.Itoa(*ctxLen))
	if err != nil {
		log.Fatal(err)
	}
	info := proc.Info()
	log.Printf("model ready: %d layers, %d experts, %d/%d cached", info.Layers, info.Experts, info.Slots, info.TotalExperts)

	static, err := fs.Sub(dist, "dist")
	if err != nil {
		log.Fatal(err)
	}
	name := strings.TrimSuffix(filepath.Base(*model), filepath.Ext(*model))
	svc := chat.NewService(proc).WithName(name)
	srv := &http.Server{Addr: *addr, Handler: web.Handler(svc, static)}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		proc.Cancel()
		srv.Shutdown(shutdown)
	}()

	fmt.Printf("chat UI: http://%s\n", *addr)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
	proc.Close()
}
