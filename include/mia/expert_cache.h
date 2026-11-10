#ifndef MIA_EXPERT_CACHE_H
#define MIA_EXPERT_CACHE_H

#include <stddef.h>
#include <stdint.h>

/* Port: fills `dst` (blob_size bytes) with expert (layer, expert). 0 = ok. */
typedef int (*mia_expert_loader)(void *ctx, int layer, int expert, void *dst);

/*
 * Fixed-capacity LRU cache of expert weight blobs. Memory use is exactly
 * n_slots * blob_size regardless of model size; misses go to the loader.
 * A pointer returned by mia_cache_get stays valid until the next get call.
 */
typedef struct {
    int n_layers, n_experts, n_slots;
    size_t blob_size;
    unsigned char *slab;
    int *slot_of; /* key -> slot, -1 if absent */
    int *key_of;  /* slot -> key, -1 if empty */
    int *prev, *next;
    int head, tail; /* head = most recently used */
    mia_expert_loader load;
    void *ctx;
    uint64_t hits, misses, bytes_read;
} mia_expert_cache;

int mia_cache_init(mia_expert_cache *c, int n_layers, int n_experts,
                   size_t blob_size, int n_slots, mia_expert_loader load,
                   void *ctx);
const unsigned char *mia_cache_get(mia_expert_cache *c, int layer, int expert);
void mia_cache_free(mia_expert_cache *c);

#endif
