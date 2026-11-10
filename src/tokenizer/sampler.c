#include <math.h>
#include <stdlib.h>

#include "mia/sampler.h"

int mia_sampler_init(mia_sampler *s, float temperature, float top_p, int n,
                     uint64_t seed) {
    s->temperature = temperature;
    s->top_p = top_p > 0.0f && top_p <= 1.0f ? top_p : 1.0f;
    s->n = n;
    s->rng = seed ? seed : 0x2545F4914F6CDD1DULL;
    s->buf = malloc(sizeof(mia_prob) * (size_t)n);
    return s->buf ? 0 : -1;
}

void mia_sampler_free(mia_sampler *s) {
    free(s->buf);
    s->buf = NULL;
}

static float rand01(mia_sampler *s) { /* xorshift64* */
    s->rng ^= s->rng >> 12;
    s->rng ^= s->rng << 25;
    s->rng ^= s->rng >> 27;
    return (float)((s->rng * 0x2545F4914F6CDD1DULL) >> 40) / 16777216.0f;
}

static int by_prob_desc(const void *a, const void *b) {
    float x = ((const mia_prob *)a)->p, y = ((const mia_prob *)b)->p;
    return (x < y) - (x > y);
}

int mia_sample(mia_sampler *s, float *logits) {
    int best = 0;
    for (int i = 1; i < s->n; i++)
        if (logits[i] > logits[best]) best = i;
    if (s->temperature <= 0.0f) return best;

    float mx = logits[best], sum = 0.0f;
    for (int i = 0; i < s->n; i++)
        sum += logits[i] = expf((logits[i] - mx) / s->temperature);
    /* Drop tokens that can never be inside the nucleus before sorting. */
    float cutoff = (1.0f - s->top_p) / (float)(s->n > 1 ? s->n - 1 : 1);
    int m = 0;
    for (int i = 0; i < s->n; i++) {
        float p = logits[i] / sum;
        if (p >= cutoff) s->buf[m].p = p, s->buf[m++].i = i;
    }
    qsort(s->buf, (size_t)m, sizeof(mia_prob), by_prob_desc);
    /* Nucleus: smallest prefix whose mass reaches top_p. */
    float cum = 0.0f;
    int last = m - 1;
    for (int i = 0; i < m; i++) {
        cum += s->buf[i].p;
        if (cum >= s->top_p) {
            last = i;
            break;
        }
    }
    float r = rand01(s) * cum, acc = 0.0f;
    for (int i = 0; i <= last; i++) {
        acc += s->buf[i].p;
        if (r < acc) return s->buf[i].i;
    }
    return s->buf[last].i;
}
