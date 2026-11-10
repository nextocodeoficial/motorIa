/*
 * mia: run a .mia MoE model within a fixed RAM budget.
 *
 *   mia -m model.mia -p "prompt" [-n 256] [--ram 4G] [-c 2048]
 *       [-t 0.7] [--top-p 0.8] [--seed N] [--raw] [--think]
 *
 * RAM = dense weights (always resident) + KV cache/activations + an LRU
 * cache of experts filling whatever budget is left. Experts that miss the
 * cache are read from disk on demand.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "mia/runtime.h"
#include "mia/sampler.h"

typedef struct {
    const char *model, *prompt;
    int n_gen, ctx, raw, think;
    double ram;
    float temp, top_p;
    unsigned long long seed;
} options;

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s -m model.mia -p prompt [options]\n"
            "  -n N        tokens to generate (default 256)\n"
            "  --ram SIZE  total RAM budget, e.g. 4G, 512M (default 4G)\n"
            "  -c N        context length (default 2048)\n"
            "  -t T        temperature, 0 = greedy (default 0.7)\n"
            "  --top-p P   nucleus sampling (default 0.8)\n"
            "  --seed N    RNG seed (default: time)\n"
            "  --raw       no chat template, plain completion\n"
            "  --think     let Qwen3 emit its <think> block\n",
            argv0);
}

static double parse_size(const char *s) {
    char *end;
    double v = strtod(s, &end);
    switch (*end) {
    case 'G': case 'g': return v * (1 << 30);
    case 'M': case 'm': return v * (1 << 20);
    case 'K': case 'k': return v * (1 << 10);
    default: return v;
    }
}

static int parse_args(int argc, char **argv, options *o) {
    *o = (options){NULL, NULL, 256, 2048, 0, 0, 4.0 * (1 << 30), 0.7f, 0.8f,
                   (unsigned long long)time(NULL)};
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--raw")) o->raw = 1;
        else if (!strcmp(a, "--think")) o->think = 1;
        else if (!v) return -1;
        else if (!strcmp(a, "-m")) o->model = v, i++;
        else if (!strcmp(a, "-p")) o->prompt = v, i++;
        else if (!strcmp(a, "-n")) o->n_gen = atoi(v), i++;
        else if (!strcmp(a, "-c")) o->ctx = atoi(v), i++;
        else if (!strcmp(a, "-t")) o->temp = (float)atof(v), i++;
        else if (!strcmp(a, "--top-p")) o->top_p = (float)atof(v), i++;
        else if (!strcmp(a, "--ram")) o->ram = parse_size(v), i++;
        else if (!strcmp(a, "--seed")) o->seed = strtoull(v, NULL, 10), i++;
        else return -1;
    }
    return o->model && o->prompt && o->ctx > 0 ? 0 : -1;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double gib(double b) { return b / (1 << 30); }

int main(int argc, char **argv) {
    options o;
    if (parse_args(argc, argv, &o) != 0) {
        usage(argv[0]);
        return 2;
    }
    char err[512];
    mia_runtime rt;
    if (mia_runtime_open(&rt, o.model, o.ram, o.ctx, err, sizeof(err)) != 0) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    if (!o.raw && rt.tok.im_start < 0) {
        fprintf(stderr, "error: tokenizer has no <|im_start|>; use --raw\n");
        return 1;
    }
    mia_runtime_describe(&rt, stderr);

    mia_sampler smp;
    if (mia_sampler_init(&smp, o.temp, o.top_p, (int)rt.f.h.vocab, o.seed)) {
        fprintf(stderr, "error: out of memory\n");
        return 1;
    }

    char *text = o.raw ? strdup(o.prompt) : mia_chat_turn(o.prompt, o.think, 0);
    int *ids = malloc(sizeof(int) * (size_t)o.ctx);
    int n_prompt = text && ids ? mia_tok_encode(&rt.tok, text, ids, o.ctx) : -1;
    free(text);
    if (n_prompt <= 0) {
        fprintf(stderr, "error: prompt is empty or longer than -c %d\n",
                o.ctx);
        return 1;
    }

    double t0 = now();
    const float *logits = NULL;
    for (int p = 0; p < n_prompt; p++) {
        if (!(logits = mia_model_step(&rt.m, ids[p], p))) {
            fprintf(stderr, "error: forward failed at prompt token %d\n", p);
            return 1;
        }
    }
    double t1 = now();

    float *work = malloc(sizeof(float) * rt.f.h.vocab);
    int pos = n_prompt, generated = 0;
    while (generated < o.n_gen && pos < o.ctx) {
        memcpy(work, logits, sizeof(float) * rt.f.h.vocab);
        int next = mia_sample(&smp, work);
        if (mia_tok_is_stop(&rt.tok, next)) break;
        int len;
        const unsigned char *piece = mia_tok_piece(&rt.tok, next, &len);
        fwrite(piece, 1, (size_t)len, stdout);
        fflush(stdout);
        generated++;
        if (!(logits = mia_model_step(&rt.m, next, pos++))) {
            fprintf(stderr, "\nerror: forward failed at position %d\n", pos);
            return 1;
        }
    }
    double t2 = now();
    printf("\n");

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    const mia_expert_cache *c = &rt.cache;
    double lookups = (double)(c->hits + c->misses);
    fprintf(stderr,
            "\nprompt: %d tok in %.2fs (%.2f tok/s) | gen: %d tok in %.2fs "
            "(%.2f tok/s)\n"
            "experts: %.1f%% hit rate, %.2f GiB read from disk | peak RSS "
            "%.2f GiB\n",
            n_prompt, t1 - t0, n_prompt / (t1 - t0), generated, t2 - t1,
            generated > 0 ? generated / (t2 - t1) : 0.0,
            lookups > 0 ? 100.0 * c->hits / lookups : 0.0,
            gib((double)c->bytes_read), gib((double)ru.ru_maxrss));

    free(work);
    free(ids);
    mia_sampler_free(&smp);
    mia_runtime_close(&rt);
    return 0;
}
