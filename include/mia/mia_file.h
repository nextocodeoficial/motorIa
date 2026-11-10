#ifndef MIA_FILE_H
#define MIA_FILE_H

#include <stddef.h>
#include <stdint.h>

#include "mia/quant.h"

/* On-disk header. Mirrors `Header` in tools/internal/mia/format.go. */
typedef struct {
    char magic[4];
    uint32_t version, wtype, dim, n_layers, n_heads, n_kv_heads, head_dim;
    uint32_t vocab, n_experts, n_active, expert_hidden, norm_topk;
    float rope_theta, norm_eps;
    uint32_t pad;
    uint64_t dense_offset, dense_size;
    uint64_t experts_offset, expert_blob_size;
    uint64_t tok_offset, tok_size;
} mia_header;

typedef char mia_header_size_check[sizeof(mia_header) == 112 ? 1 : -1];

typedef struct {
    const float *attn_norm, *q_norm, *k_norm, *ffn_norm, *router;
    const void *wq, *wk, *wv, *wo;
} mia_layer;

typedef struct {
    mia_header h;
    mia_wtype wt;
    int fd;
    unsigned char *dense; /* resident: embeddings, attention, norms, routers */
    const void *token_embd, *output;
    const float *output_norm;
    mia_layer *layers;
    size_t off_up, off_down; /* sub-matrix offsets inside an expert blob */
} mia_file;

/* Opens the file, validates it and loads the dense section into RAM. */
int mia_file_open(mia_file *f, const char *path, char *err, size_t errlen);
void mia_file_close(mia_file *f);

/* mia_expert_loader implementation: one aligned read per expert. */
int mia_file_read_expert(void *f, int layer, int expert, void *dst);

/* Reads the tokenizer blob (malloc'd, caller frees). NULL if absent. */
unsigned char *mia_file_read_tokenizer(const mia_file *f, size_t *len);

#endif
