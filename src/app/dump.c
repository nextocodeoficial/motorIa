/*
 * mia-dump: test harness. Runs a token sequence and writes raw float32
 * logits [n_tokens][vocab] to a file.
 *   mia-dump <model.mia> <out.bin> <cache_slots|0=all> <tok>...
 */
#include <stdio.h>
#include <stdlib.h>

#include "mia/model.h"

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.mia out.bin slots tok...\n", argv[0]);
        return 2;
    }
    char err[256];
    mia_file f;
    if (mia_file_open(&f, argv[1], err, sizeof(err)) != 0) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    int slots = atoi(argv[3]);
    if (slots <= 0) slots = (int)(f.h.n_layers * f.h.n_experts);
    mia_expert_cache cache;
    mia_model m;
    int n = argc - 4;
    if (mia_cache_init(&cache, (int)f.h.n_layers, (int)f.h.n_experts,
                       f.h.expert_blob_size, slots, mia_file_read_expert,
                       &f) != 0 ||
        mia_model_init(&m, &f, &cache, n) != 0) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        perror(argv[2]);
        return 1;
    }
    for (int p = 0; p < n; p++) {
        const float *logits = mia_model_step(&m, atoi(argv[4 + p]), p);
        if (!logits) {
            fprintf(stderr, "step %d failed\n", p);
            return 1;
        }
        fwrite(logits, sizeof(float), f.h.vocab, out);
    }
    fclose(out);
    fprintf(stderr, "cache: %llu hits, %llu misses\n",
            (unsigned long long)cache.hits, (unsigned long long)cache.misses);
    mia_model_free(&m);
    mia_cache_free(&cache);
    mia_file_close(&f);
    return 0;
}
