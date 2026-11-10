#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mia/matmul.h"
#include "mia/model.h"

typedef struct {
    int d, hd, nh, nkv, qd, kvd, V, E, K, eh, L;
} dims;

static dims dims_of(const mia_header *h) {
    dims s;
    s.d = (int)h->dim, s.hd = (int)h->head_dim, s.nh = (int)h->n_heads;
    s.nkv = (int)h->n_kv_heads, s.V = (int)h->vocab, s.E = (int)h->n_experts;
    s.K = (int)h->n_active, s.eh = (int)h->expert_hidden;
    s.L = (int)h->n_layers;
    s.qd = s.nh * s.hd, s.kvd = s.nkv * s.hd;
    return s;
}

static int max3(int a, int b, int c) {
    int m = a > b ? a : b;
    return m > c ? m : c;
}

size_t mia_model_state_bytes(const mia_file *f, int max_seq) {
    dims s = dims_of(&f->h);
    size_t floats = (size_t)s.d * 3 + (size_t)s.qd * 2 + (size_t)s.kvd * 2 +
                    (size_t)s.eh * 2 + (size_t)s.E + (size_t)s.V +
                    2 * (size_t)s.L * max_seq * s.kvd;
    return floats * sizeof(float) +
           sizeof(mia_block_q8) * (size_t)(max3(s.d, s.qd, s.eh) / MIA_QK);
}

int mia_model_init(mia_model *m, const mia_file *f, mia_expert_cache *cache,
                   int max_seq) {
    memset(m, 0, sizeof(*m));
    dims s = dims_of(&f->h);
    m->f = f, m->cache = cache, m->max_seq = max_seq;
    m->x = calloc(s.d, sizeof(float));
    m->xn = calloc(s.d, sizeof(float));
    m->tmp = calloc(s.d, sizeof(float));
    m->q = calloc(s.qd, sizeof(float));
    m->att = calloc(s.qd, sizeof(float));
    m->k = calloc(s.kvd, sizeof(float));
    m->v = calloc(s.kvd, sizeof(float));
    m->gate = calloc(s.eh, sizeof(float));
    m->up = calloc(s.eh, sizeof(float));
    m->router = calloc(s.E, sizeof(float));
    m->logits = calloc(s.V, sizeof(float));
    m->kc = calloc((size_t)s.L * max_seq * s.kvd, sizeof(float));
    m->vc = calloc((size_t)s.L * max_seq * s.kvd, sizeof(float));
    m->xq = calloc(max3(s.d, s.qd, s.eh) / MIA_QK, sizeof(mia_block_q8));
    if (!m->x || !m->xn || !m->tmp || !m->q || !m->att || !m->k || !m->v ||
        !m->gate || !m->up || !m->router || !m->logits || !m->kc || !m->vc ||
        !m->xq) {
        mia_model_free(m);
        return -1;
    }
    return 0;
}

void mia_model_free(mia_model *m) {
    float *bufs[] = {m->x,  m->xn,     m->tmp,    m->q,  m->att,
                     m->k,  m->v,      m->gate,   m->up, m->router,
                     m->logits, m->kc, m->vc};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) free(bufs[i]);
    free(m->xq);
    memset(m, 0, sizeof(*m));
}

/* ---- primitives -------------------------------------------------------- */

static void rmsnorm(float *out, const float *x, const float *w, int n,
                    float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = (float)(1.0 / sqrt(ss / n + eps));
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

/* NeoX rotary embedding: element i pairs with i + hd/2. */
static void rope(float *x, int hd, int pos, float theta) {
    int half = hd / 2;
    for (int i = 0; i < half; i++) {
        double freq = pow((double)theta, -2.0 * i / hd);
        double a = pos * freq, c = cos(a), s = sin(a);
        float x0 = x[i], x1 = x[i + half];
        x[i] = (float)(x0 * c - x1 * s);
        x[i + half] = (float)(x0 * s + x1 * c);
    }
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) mx = x[i] > mx ? x[i] : mx;
    float sum = 0;
    for (int i = 0; i < n; i++) sum += x[i] = expf(x[i] - mx);
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* Top-k by repeated selection: k is small (8) and E modest (128). */
static void top_k(const float *p, int n, int k, int *idx) {
    for (int j = 0; j < k; j++) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            int taken = 0;
            for (int t = 0; t < j; t++) taken |= idx[t] == i;
            if (!taken && (best < 0 || p[i] > p[best])) best = i;
        }
        idx[j] = best;
    }
}

/* ---- blocks ------------------------------------------------------------ */

static void attention(mia_model *m, dims s, int l, int pos) {
    const mia_file *f = m->f;
    const mia_layer *L = &f->layers[l];
    const float eps = f->h.norm_eps;

    rmsnorm(m->xn, m->x, L->attn_norm, s.d, eps);
    mia_matvec(m->q, L->wq, f->wt, m->xn, m->xq, s.qd, s.d);
    mia_matvec(m->k, L->wk, f->wt, m->xn, m->xq, s.kvd, s.d);
    mia_matvec(m->v, L->wv, f->wt, m->xn, m->xq, s.kvd, s.d);
    for (int h = 0; h < s.nh; h++) {
        float *qh = m->q + h * s.hd;
        rmsnorm(qh, qh, L->q_norm, s.hd, eps);
        rope(qh, s.hd, pos, f->h.rope_theta);
    }
    for (int h = 0; h < s.nkv; h++) {
        float *kh = m->k + h * s.hd;
        rmsnorm(kh, kh, L->k_norm, s.hd, eps);
        rope(kh, s.hd, pos, f->h.rope_theta);
    }
    size_t layer_off = (size_t)l * m->max_seq * s.kvd;
    float *kc = m->kc + layer_off, *vc = m->vc + layer_off;
    memcpy(kc + (size_t)pos * s.kvd, m->k, sizeof(float) * s.kvd);
    memcpy(vc + (size_t)pos * s.kvd, m->v, sizeof(float) * s.kvd);

    const int group = s.nh / s.nkv;
    const float scale = 1.0f / sqrtf((float)s.hd);
    float *scores = malloc(sizeof(float) * (pos + 1));
    for (int h = 0; h < s.nh; h++) {
        const float *qh = m->q + h * s.hd;
        int kvh = h / group;
        for (int t = 0; t <= pos; t++) {
            const float *kt = kc + (size_t)t * s.kvd + kvh * s.hd;
            float d = 0;
            for (int i = 0; i < s.hd; i++) d += qh[i] * kt[i];
            scores[t] = d * scale;
        }
        softmax(scores, pos + 1);
        float *out = m->att + h * s.hd;
        memset(out, 0, sizeof(float) * s.hd);
        for (int t = 0; t <= pos; t++) {
            const float *vt = vc + (size_t)t * s.kvd + kvh * s.hd;
            for (int i = 0; i < s.hd; i++) out[i] += scores[t] * vt[i];
        }
    }
    free(scores);
    mia_matvec(m->tmp, L->wo, f->wt, m->att, m->xq, s.d, s.qd);
    for (int i = 0; i < s.d; i++) m->x[i] += m->tmp[i];
}

static int moe(mia_model *m, dims s, int l) {
    const mia_file *f = m->f;
    const mia_layer *L = &f->layers[l];
    int idx[64];
    float w[64];
    if (s.K > 64) return -1;

    rmsnorm(m->xn, m->x, L->ffn_norm, s.d, f->h.norm_eps);
    mia_matvec(m->router, L->router, MIA_F32, m->xn, NULL, s.E, s.d);
    softmax(m->router, s.E);
    top_k(m->router, s.E, s.K, idx);
    float sum = 0;
    for (int j = 0; j < s.K; j++) sum += w[j] = m->router[idx[j]];
    if (f->h.norm_topk)
        for (int j = 0; j < s.K; j++) w[j] /= sum;

    for (int j = 0; j < s.K; j++) {
        const unsigned char *blob = mia_cache_get(m->cache, l, idx[j]);
        if (!blob) return -1;
        mia_matvec(m->gate, blob, f->wt, m->xn, m->xq, s.eh, s.d);
        mia_matvec(m->up, blob + f->off_up, f->wt, m->xn, m->xq, s.eh, s.d);
        for (int i = 0; i < s.eh; i++) {
            float g = m->gate[i];
            m->gate[i] = g / (1.0f + expf(-g)) * m->up[i];
        }
        mia_matvec(m->tmp, blob + f->off_down, f->wt, m->gate, m->xq, s.d,
                   s.eh);
        for (int i = 0; i < s.d; i++) m->x[i] += w[j] * m->tmp[i];
    }
    return 0;
}

const float *mia_model_step(mia_model *m, int token, int pos) {
    const mia_file *f = m->f;
    dims s = dims_of(&f->h);
    if (pos < 0 || pos >= m->max_seq || token < 0 || token >= s.V)
        return NULL;

    const unsigned char *row =
        (const unsigned char *)f->token_embd +
        (size_t)token * mia_row_bytes(f->wt, s.d);
    mia_dequantize_row(f->wt, row, m->x, s.d);

    for (int l = 0; l < s.L; l++) {
        attention(m, s, l, pos);
        if (moe(m, s, l) != 0) return NULL;
    }
    rmsnorm(m->xn, m->x, f->output_norm, s.d, f->h.norm_eps);
    mia_matvec(m->logits, f->output, f->wt, m->xn, m->xq, s.V, s.d);
    return m->logits;
}
