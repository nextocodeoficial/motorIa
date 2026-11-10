/*
 * mia-serve: keeps a model loaded and holds one conversation's KV cache,
 * so each new message only prefills its own tokens. Driven over
 * stdin/stdout by tools/internal/engine (protocol documented there).
 *
 *   mia-serve -m model.mia [--ram 4G] [-c 4096] [--seed N]
 *
 * SIGUSR1 cancels the running CHAT. Diagnostics go to stderr.
 */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mia/runtime.h"
#include "mia/sampler.h"

#define MAX_MESSAGE (1 << 20)

static volatile sig_atomic_t g_cancel = 0;
static void on_cancel(int sig) { (void)sig, g_cancel = 1; }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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

typedef struct {
    mia_runtime rt;
    mia_sampler smp;
    int pos;   /* tokens in the KV cache */
    int *ids;  /* scratch for one message, ctx entries */
    float *work;
} server;

static void emit_err(const char *code, const char *msg) {
    printf("ERR %s %s\n", code, msg);
    fflush(stdout);
}

static void emit_done(server *s, const char *reason, int prompt, int gen,
                      double prefill_s, double gen_s) {
    printf("DONE reason=%s prompt=%d gen=%d prefill_s=%.3f gen_s=%.3f "
           "ctx_used=%d ctx_max=%d hits=%llu misses=%llu\n",
           reason, prompt, gen, prefill_s, gen_s, s->pos, s->rt.ctx,
           (unsigned long long)s->rt.cache.hits,
           (unsigned long long)s->rt.cache.misses);
    fflush(stdout);
}

static void chat(server *s, const char *user, int max_tokens, float temp,
                 float top_p, int think) {
    mia_runtime *rt = &s->rt;
    g_cancel = 0;
    s->smp.temperature = temp;
    s->smp.top_p = top_p > 0.0f && top_p <= 1.0f ? top_p : 1.0f;

    char *text = mia_chat_turn(user, think, s->pos > 0);
    int n = text ? mia_tok_encode(&rt->tok, text, s->ids, rt->ctx) : -1;
    free(text);
    int left = rt->ctx - s->pos - 1; /* keep room for at least one reply */
    if (n < 0 || n > left) {
        char msg[160];
        if (n < 0)
            snprintf(msg, sizeof(msg),
                     "this message is longer than the whole context (%d "
                     "tokens); start a new chat or shorten it", rt->ctx);
        else
            snprintf(msg, sizeof(msg),
                     "this message needs %d tokens but only %d are left; "
                     "start a new chat", n, left);
        emit_err("context_full", msg);
        return;
    }

    const int start = s->pos;
    const float *logits = NULL;
    double t0 = now();
    for (int i = 0; i < n; i++) {
        if (g_cancel) { /* roll back: later steps overwrite these slots */
            s->pos = start;
            emit_done(s, "cancel", i, 0, now() - t0, 0);
            return;
        }
        if (!(logits = mia_model_step(&rt->m, s->ids[i], s->pos))) {
            s->pos = 0;
            emit_err("engine", "forward pass failed reading the model file; "
                               "conversation was reset");
            return;
        }
        s->pos++;
        printf("PROG %d %d\n", i + 1, n);
        fflush(stdout);
    }
    double t1 = now();

    const char *reason = "length";
    int gen = 0;
    while (gen < max_tokens) {
        if (s->pos >= rt->ctx) {
            reason = "ctx";
            break;
        }
        memcpy(s->work, logits, sizeof(float) * rt->f.h.vocab);
        int next = mia_sample(&s->smp, s->work);
        if (mia_tok_is_stop(&rt->tok, next)) {
            reason = "stop";
            break;
        }
        int len;
        const unsigned char *piece = mia_tok_piece(&rt->tok, next, &len);
        printf("TOK %d %llu %llu\n", len, (unsigned long long)rt->cache.hits,
               (unsigned long long)rt->cache.misses);
        fwrite(piece, 1, (size_t)len, stdout);
        fflush(stdout);
        gen++;
        if (!(logits = mia_model_step(&rt->m, next, s->pos))) {
            s->pos = 0;
            emit_err("engine", "forward pass failed reading the model file; "
                               "conversation was reset");
            return;
        }
        s->pos++;
        if (g_cancel) {
            reason = "cancel";
            break;
        }
    }
    emit_done(s, reason, n, gen, t1 - t0, now() - t1);
}

int main(int argc, char **argv) {
    const char *model = NULL;
    double ram = 4.0 * (1 << 30);
    int ctx = 4096;
    unsigned long long seed = (unsigned long long)time(NULL);
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-m")) model = argv[i + 1];
        else if (!strcmp(argv[i], "--ram")) ram = parse_size(argv[i + 1]);
        else if (!strcmp(argv[i], "-c")) ctx = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--seed")) seed = strtoull(argv[i + 1], 0, 10);
    }
    if (!model) {
        fprintf(stderr, "usage: %s -m model.mia [--ram 4G] [-c 4096] "
                        "[--seed N]\n", argv[0]);
        return 2;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_cancel;
    sa.sa_flags = SA_RESTART;
    sigaction(SIGUSR1, &sa, NULL);

    static server s;
    char err[512];
    if (mia_runtime_open(&s.rt, model, ram, ctx, err, sizeof(err)) != 0) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    if (s.rt.tok.im_start < 0) {
        fprintf(stderr, "error: tokenizer has no <|im_start|>; this model "
                        "cannot chat\n");
        return 1;
    }
    s.ids = malloc(sizeof(int) * (size_t)ctx);
    s.work = malloc(sizeof(float) * s.rt.f.h.vocab);
    if (!s.ids || !s.work ||
        mia_sampler_init(&s.smp, 0.7f, 0.8f, (int)s.rt.f.h.vocab, seed)) {
        fprintf(stderr, "error: out of memory\n");
        return 1;
    }
    mia_runtime_describe(&s.rt, stderr);
    const mia_header *h = &s.rt.f.h;
    printf("READY layers=%u experts=%u active=%u ctx=%d slots=%d total=%d "
           "wtype=%s dense_gib=%.2f cache_gib=%.2f\n",
           h->n_layers, h->n_experts, h->n_active, ctx, s.rt.slots,
           s.rt.total_experts, s.rt.f.wt == MIA_Q8_0 ? "q8_0" : "f32",
           s.rt.dense_bytes / (1 << 30), s.rt.cache_bytes / (1 << 30));
    fflush(stdout);

    char line[256];
    char *msg = malloc(MAX_MESSAGE + 1);
    while (fgets(line, sizeof(line), stdin)) {
        int nbytes, max_tokens, think;
        float temp, top_p;
        if (!strcmp(line, "RESET\n")) {
            s.pos = 0;
            printf("OK\n");
            fflush(stdout);
        } else if (sscanf(line, "CHAT %d %d %f %f %d", &nbytes, &max_tokens,
                          &temp, &top_p, &think) == 5) {
            if (nbytes < 0 || nbytes > MAX_MESSAGE ||
                fread(msg, 1, (size_t)nbytes, stdin) != (size_t)nbytes) {
                emit_err("bad_request", "message too large or truncated");
                continue;
            }
            msg[nbytes] = '\0';
            chat(&s, msg, max_tokens, temp, top_p, think);
        } else if (!strcmp(line, "QUIT\n")) {
            break;
        } else {
            line[strcspn(line, "\n")] = '\0';
            emit_err("bad_request", "unknown command");
        }
    }
    free(msg);
    free(s.ids);
    free(s.work);
    mia_sampler_free(&s.smp);
    mia_runtime_close(&s.rt);
    return 0;
}
