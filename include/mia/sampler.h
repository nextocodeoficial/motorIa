#ifndef MIA_SAMPLER_H
#define MIA_SAMPLER_H

#include <stdint.h>

typedef struct {
    float p;
    int i;
} mia_prob;

/* temperature <= 0 means greedy. top_p in (0, 1]. */
typedef struct {
    float temperature, top_p;
    int n;
    uint64_t rng;
    mia_prob *buf;
} mia_sampler;

int mia_sampler_init(mia_sampler *s, float temperature, float top_p, int n,
                     uint64_t seed);
void mia_sampler_free(mia_sampler *s);
/* Picks a token id. `logits` (n values) is overwritten. */
int mia_sample(mia_sampler *s, float *logits);

#endif
