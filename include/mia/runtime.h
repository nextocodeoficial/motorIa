#ifndef MIA_RUNTIME_H
#define MIA_RUNTIME_H

#include <stdio.h>

#include "mia/model.h"
#include "mia/tokenizer.h"

/*
 * Everything needed to run a chat model within a RAM budget: the file,
 * its tokenizer, the expert cache sized from the budget and the model
 * state. Must not be moved after mia_runtime_open (internal pointers).
 */
typedef struct {
    mia_file f;
    mia_tokenizer tok;
    mia_expert_cache cache;
    mia_model m;
    int ctx, slots, total_experts;
    double dense_bytes, state_bytes, cache_bytes;
} mia_runtime;

int mia_runtime_open(mia_runtime *rt, const char *path, double ram_bytes,
                     int ctx, char *err, size_t errlen);
void mia_runtime_close(mia_runtime *rt);
void mia_runtime_describe(const mia_runtime *rt, FILE *out);

/*
 * Qwen3 chat-template text for one user turn. When `continuing`, the
 * previous assistant reply is closed first (its stop token is never fed
 * back into the model). Returns malloc'd text.
 */
char *mia_chat_turn(const char *user, int think, int continuing);

#endif
