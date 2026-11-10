# motorIa

A small C inference engine for **Qwen3-30B-A3B** (Mixture-of-Experts) on Apple Silicon. Dense weights stay in RAM; experts are streamed from disk through a cache, so the 30B model runs inside a small RAM budget (default 6 GB).

The project has three parts:

| Part | Language | What it does |
| --- | --- | --- |
| `src/`, `include/` | C | Engine: `mia` (CLI), `mia-serve` (stdin/stdout server), `mia-dump` |
| `tools/` | Go | `mia-convert` (Hugging Face → `.mia`) and `mia-web` (HTTP server for the chat) |
| `web/` | Vue + Vite | Chat UI, embedded into `mia-web` |

## Requirements

- macOS on Apple Silicon. The Makefile builds with `-mcpu=apple-m4`; on M1–M3, override it (see [Build](#2-build-the-engine)).
- Xcode Command Line Tools (`clang`, `make`): `xcode-select --install`
- Go 1.26+
- Node.js 20+ and npm (only for the chat UI)
- Python 3 + `huggingface_hub` (only to download the model)
- Disk space: ~61 GB for the original checkpoint + ~34 GB for the converted `.mia` file. You can delete the checkpoint after converting.

## 1. Download the model

The engine uses the original BF16 checkpoint of [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) (`config.json`, `tokenizer.json` and the `*.safetensors` shards).

Install the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

Download the checkpoint (the examples use `/Volumes/nuevo/models`; change it to any folder with enough space):

```bash
hf download Qwen/Qwen3-30B-A3B --local-dir /Volumes/nuevo/models/Qwen3-30B-A3B
```

If the download is interrupted, run the same command again: it resumes where it stopped.

## 2. Build the engine

```bash
make
```

This produces `build/mia`, `build/mia-serve` and `build/mia-dump`.

On a chip other than M4, override the CPU target:

```bash
make CFLAGS="-std=c99 -O3 -Wall -Wextra -mcpu=native -fblocks"
```

## 3. Convert the model to `.mia`

`mia-convert` reads the checkpoint and writes a single `.mia` file with Q8_0 weights (~34 GB). It shows progress and an ETA per expert.

```bash
cd tools
go run ./cmd/mia-convert \
  -in  /Volumes/nuevo/models/Qwen3-30B-A3B \
  -out /Volumes/nuevo/models/qwen3-30b-a3b-q8.mia \
  -type q8_0
cd ..
```

`-type f32` is also available, but the file is about four times larger.

## 4. Run

### Command line

```bash
build/mia -m /Volumes/nuevo/models/qwen3-30b-a3b-q8.mia -p "Explain what a Mixture-of-Experts model is" --ram 6G
```

Main options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-n N` | 256 | Tokens to generate |
| `--ram SIZE` | 4G | Total RAM budget (e.g. `4G`, `512M`) |
| `-c N` | 2048 | Context length |
| `-t T` | 0.7 | Temperature |
| `--top-p P` | 0.8 | Nucleus sampling |
| `--seed N` | time | RNG seed |
| `--raw` | off | No chat template, plain completion |
| `--think` | off | Let Qwen3 emit its `<think>` block |

### Chat in the browser

Install the UI dependencies once:

```bash
cd web && npm install && cd ..
```

Build the UI, the server and start it:

```bash
make chat MODEL=/Volumes/nuevo/models/qwen3-30b-a3b-q8.mia RAM=6G
```

Open <http://127.0.0.1:8080>.

`mia-web` also accepts `-ctx` (conversation length, default 4096), `-addr` (listen address) and `-engine` (path to `mia-serve`):

```bash
build/mia-web -model /Volumes/nuevo/models/qwen3-30b-a3b-q8.mia -ram 6G -ctx 8192 -addr 127.0.0.1:9000
```

For UI development with hot reload, keep `mia-web` running on port 8080 and run `npm run dev` in `web/`; Vite proxies `/api` to it.

## Tests

```bash
make test
```

```bash
make e2e
```

```bash
cd web && npm test
```

`make test` runs the C unit tests, `make e2e` runs the Go end-to-end tests against the built binaries, and `npm test` runs the UI tests.

## Clean

```bash
make clean
```
