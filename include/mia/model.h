#ifndef MIA_MODEL_H
#define MIA_MODEL_H

#include "mia/expert_cache.h"
#include "mia/mia_file.h"

/* Inference state for one sequence: activations + KV cache. */
typedef struct {
    const mia_file *f;
    mia_expert_cache *cache;
    int max_seq;
    float *x, *xn, *q, *k, *v, *att, *tmp, *gate, *up, *router, *logits;
    float *kc, *vc; /* [layer][pos][kv_dim] */
    mia_block_q8 *xq;
} mia_model;

int mia_model_init(mia_model *m, const mia_file *f, mia_expert_cache *cache,
                   int max_seq);
void mia_model_free(mia_model *m);

/* Feeds `token` at position `pos` (sequential from 0). Returns vocab logits,
 * or NULL on an I/O error or when pos >= max_seq. */
const float *mia_model_step(mia_model *m, int token, int pos);

/* Bytes of RAM the model state needs besides dense weights and the cache. */
size_t mia_model_state_bytes(const mia_file *f, int max_seq);

#endif
