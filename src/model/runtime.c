#include "mia/runtime.h"

#include <stdlib.h>
#include <string.h>

static double gib(double b) { return b / (1 << 30); }

/* Tokenizer, sampler buffers and allocator slack. */
#define RUNTIME_OVERHEAD (16.0 * (1 << 20))

int mia_runtime_open(mia_runtime *rt, const char *path, double ram_bytes,
                     int ctx, char *err, size_t errlen) {
    memset(rt, 0, sizeof(*rt));
    if (ctx <= 0) {
        snprintf(err, errlen, "context length must be positive");
        return -1;
    }
    if (mia_file_open(&rt->f, path, err, errlen) != 0) return -1;
    mia_file *f = &rt->f;

    size_t tlen;
    unsigned char *tblob = mia_file_read_tokenizer(f, &tlen);
    int tok_ok = tblob && mia_tok_load(&rt->tok, tblob, tlen) == 0;
    free(tblob);
    if (!tok_ok) {
        snprintf(err, errlen, "model has no usable tokenizer");
        mia_file_close(f);
        return -1;
    }

    rt->ctx = ctx;
    rt->total_experts = (int)(f->h.n_layers * f->h.n_experts);
    rt->dense_bytes = (double)f->h.dense_size;
    rt->state_bytes = (double)mia_model_state_bytes(f, ctx);
    double blob = (double)f->h.expert_blob_size;
    double fixed = rt->dense_bytes + rt->state_bytes + RUNTIME_OVERHEAD;
    double avail = ram_bytes - fixed;
    int slots = avail > 0 ? (int)(avail / blob) : 0;
    if (slots < (int)f->h.n_active) {
        snprintf(err, errlen,
                 "RAM budget %.2f GiB too small: dense %.2f + state %.2f GiB "
                 "+ %u experts x %.1f MiB needed (try --ram %.1fG or a "
                 "smaller -c)",
                 gib(ram_bytes), gib(rt->dense_bytes), gib(rt->state_bytes),
                 f->h.n_active, blob / (1 << 20),
                 gib(fixed + blob * f->h.n_active) + 0.1);
        mia_runtime_close(rt);
        return -1;
    }
    if (slots > rt->total_experts) slots = rt->total_experts;
    rt->slots = slots;
    rt->cache_bytes = blob * slots;

    if (mia_cache_init(&rt->cache, (int)f->h.n_layers, (int)f->h.n_experts,
                       f->h.expert_blob_size, slots, mia_file_read_expert,
                       f) != 0 ||
        mia_model_init(&rt->m, f, &rt->cache, ctx) != 0) {
        snprintf(err, errlen, "out of memory");
        mia_runtime_close(rt);
        return -1;
    }
    return 0;
}

void mia_runtime_close(mia_runtime *rt) {
    mia_model_free(&rt->m);
    mia_cache_free(&rt->cache);
    mia_tok_free(&rt->tok);
    if (rt->f.dense || rt->f.fd > 0) mia_file_close(&rt->f);
    memset(rt, 0, sizeof(*rt));
}

void mia_runtime_describe(const mia_runtime *rt, FILE *out) {
    const mia_header *h = &rt->f.h;
    fprintf(out,
            "mia: %u layers, %u experts (top-%u), %s | RAM plan: dense %.2f "
            "GiB + state %.2f GiB + cache %d/%d experts (%.2f GiB)\n",
            h->n_layers, h->n_experts, h->n_active,
            rt->f.wt == MIA_Q8_0 ? "q8_0" : "f32", gib(rt->dense_bytes),
            gib(rt->state_bytes), rt->slots, rt->total_experts,
            gib(rt->cache_bytes));
}

char *mia_chat_turn(const char *user, int think, int continuing) {
    const char *close = continuing ? "<|im_end|>\n" : "";
    const char *head = "<|im_start|>user\n";
    const char *tail = think ? "<|im_end|>\n<|im_start|>assistant\n"
                             : "<|im_end|>\n<|im_start|>assistant\n"
                               "<think>\n\n</think>\n\n";
    size_t n = strlen(close) + strlen(head) + strlen(user) + strlen(tail) + 1;
    char *s = malloc(n);
    if (s) snprintf(s, n, "%s%s%s%s", close, head, user, tail);
    return s;
}
