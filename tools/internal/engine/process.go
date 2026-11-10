package engine

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
)

// ErrBusy is returned when a command arrives while another one runs.
var ErrBusy = errors.New("engine is busy with another message")

type Params struct {
	MaxTokens         int
	Temperature, TopP float32
	Think             bool
}

// Process is a running mia-serve. One command at a time.
type Process struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	out    *bufio.Reader
	info   Info
	mu     sync.Mutex
	stderr *tailBuffer
}

// Start launches the engine and waits until the model is loaded.
func Start(bin string, args ...string) (*Process, error) {
	cmd := exec.Command(bin, args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	tail := &tailBuffer{max: 4096}
	cmd.Stderr = io.MultiWriter(os.Stderr, tail)
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	p := &Process{cmd: cmd, stdin: stdin, out: bufio.NewReaderSize(stdout, 1<<16), stderr: tail}
	ev, err := readEvent(p.out)
	if err != nil {
		cmd.Process.Kill()
		cmd.Wait()
		return nil, fmt.Errorf("engine failed to start: %s", strings.TrimSpace(tail.String()))
	}
	r, ok := ev.(Ready)
	if !ok {
		p.Close()
		return nil, fmt.Errorf("engine protocol: expected READY, got %T", ev)
	}
	p.info = r.Info
	return p, nil
}

func (p *Process) Info() Info { return p.info }

// Chat sends one user message and streams events until the reply ends.
// Cancelling ctx stops generation; the partial reply stays in context.
func (p *Process) Chat(ctx context.Context, text string, prm Params, emit func(Event)) (Done, error) {
	if !p.mu.TryLock() {
		return Done{}, ErrBusy
	}
	defer p.mu.Unlock()

	think := 0
	if prm.Think {
		think = 1
	}
	if _, err := fmt.Fprintf(p.stdin, "CHAT %d %d %g %g %d\n%s",
		len(text), prm.MaxTokens, prm.Temperature, prm.TopP, think, text); err != nil {
		return Done{}, p.died(err)
	}

	stop := context.AfterFunc(ctx, func() { p.Cancel() })
	defer stop()

	var j utf8Joiner
	for {
		ev, err := readEvent(p.out)
		if err != nil {
			return Done{}, p.died(err)
		}
		switch e := ev.(type) {
		case Progress:
			emit(e)
		case Token:
			if e.Text = j.push(e.Raw); e.Text != "" {
				emit(e)
			}
		case Done:
			if rest := j.flush(); rest != "" {
				emit(Token{Text: rest, Hits: e.Hits, Misses: e.Misses})
			}
			return e, nil
		case *Error:
			return Done{}, e
		default:
			return Done{}, fmt.Errorf("engine protocol: unexpected %T during chat", ev)
		}
	}
}

// Reset clears the conversation context.
func (p *Process) Reset() error {
	if !p.mu.TryLock() {
		return ErrBusy
	}
	defer p.mu.Unlock()
	if _, err := io.WriteString(p.stdin, "RESET\n"); err != nil {
		return p.died(err)
	}
	ev, err := readEvent(p.out)
	if err != nil {
		return p.died(err)
	}
	if e, isErr := ev.(*Error); isErr {
		return e
	}
	return nil
}

// Cancel asks the running Chat to stop. Safe to call any time.
func (p *Process) Cancel() error { return p.cmd.Process.Signal(syscall.SIGUSR1) }

func (p *Process) Close() error {
	p.stdin.Close()
	err := p.cmd.Wait()
	return err
}

func (p *Process) died(err error) error {
	return fmt.Errorf("engine stopped (%v): %s", err, strings.TrimSpace(p.stderr.String()))
}

// tailBuffer keeps the last `max` bytes written (engine stderr).
type tailBuffer struct {
	mu  sync.Mutex
	buf []byte
	max int
}

func (t *tailBuffer) Write(b []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, b...)
	if len(t.buf) > t.max {
		t.buf = t.buf[len(t.buf)-t.max:]
	}
	return len(b), nil
}

func (t *tailBuffer) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return string(t.buf)
}
