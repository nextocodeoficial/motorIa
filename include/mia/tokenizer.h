#ifndef MIA_TOKENIZER_H
#define MIA_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

/*
 * Byte-level BPE (GPT-2 / Qwen2 style). Loads the MTK1 blob produced by
 * tools/internal/tok. Pre-tokenization approximates the Qwen2 regex: ASCII
 * is exact; multi-byte UTF-8 counts as letters except common punctuation
 * blocks. Any mismatch yields a different but still valid token sequence.
 */
typedef struct {
    int n_vocab;
    uint32_t *off;
    uint16_t *len;
    unsigned char *bytes;
    int byte_id[256];
    uint32_t mcap; /* merge hash: open addressing, power of two */
    uint64_t *mkey;
    uint32_t *mrank, *mres;
    int n_special;
    int *special;
    int eos, im_start, im_end; /* -1 when absent */
} mia_tokenizer;

int mia_tok_load(mia_tokenizer *t, const unsigned char *blob, size_t n);
void mia_tok_free(mia_tokenizer *t);

/* Encodes UTF-8 text; returns token count (<= max) or -1 on overflow. */
int mia_tok_encode(const mia_tokenizer *t, const char *text, int *out,
                   int max);
const unsigned char *mia_tok_piece(const mia_tokenizer *t, int id, int *len);
int mia_tok_is_stop(const mia_tokenizer *t, int id);

#endif
