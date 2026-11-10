#include "mia/expert_cache.h"

#include <stdlib.h>
#include <string.h>

enum { SLAB_ALIGN = 16384 };

static void unlink_slot(mia_expert_cache *c, int s) {
    if (c->prev[s] >= 0) c->next[c->prev[s]] = c->next[s];
    else c->head = c->next[s];
    if (c->next[s] >= 0) c->prev[c->next[s]] = c->prev[s];
    else c->tail = c->prev[s];
    c->prev[s] = c->next[s] = -1;
}

static void push_front(mia_expert_cache *c, int s) {
    c->prev[s] = -1;
    c->next[s] = c->head;
    if (c->head >= 0) c->prev[c->head] = s;
    c->head = s;
    if (c->tail < 0) c->tail = s;
}

int mia_cache_init(mia_expert_cache *c, int n_layers, int n_experts,
                   size_t blob_size, int n_slots, mia_expert_loader load,
                   void *ctx) {
    memset(c, 0, sizeof(*c));
    if (n_slots < 1 || n_layers < 1 || n_experts < 1) return -1;
    int n_keys = n_layers * n_experts;
    if (n_slots > n_keys) n_slots = n_keys;
    c->n_layers = n_layers;
    c->n_experts = n_experts;
    c->n_slots = n_slots;
    c->blob_size = blob_size;
    c->load = load;
    c->ctx = ctx;
    void *slab = NULL;
    if (posix_memalign(&slab, SLAB_ALIGN, blob_size * (size_t)n_slots) != 0)
        return -1;
    c->slab = slab;
    c->slot_of = malloc(sizeof(int) * (size_t)n_keys);
    c->key_of = malloc(sizeof(int) * (size_t)n_slots);
    c->prev = malloc(sizeof(int) * (size_t)n_slots);
    c->next = malloc(sizeof(int) * (size_t)n_slots);
    if (!c->slot_of || !c->key_of || !c->prev || !c->next) {
        mia_cache_free(c);
        return -1;
    }
    for (int k = 0; k < n_keys; k++) c->slot_of[k] = -1;
    c->head = c->tail = -1;
    for (int s = 0; s < n_slots; s++) {
        c->key_of[s] = -1;
        c->prev[s] = c->next[s] = -1;
        push_front(c, s); /* empty slots, all equally evictable */
    }
    return 0;
}

const unsigned char *mia_cache_get(mia_expert_cache *c, int layer,
                                   int expert) {
    if (layer < 0 || layer >= c->n_layers || expert < 0 ||
        expert >= c->n_experts)
        return NULL;
    int key = layer * c->n_experts + expert;
    int s = c->slot_of[key];
    if (s >= 0) {
        c->hits++;
        unlink_slot(c, s);
        push_front(c, s);
        return c->slab + (size_t)s * c->blob_size;
    }
    /* Miss: reuse the least recently used slot. */
    s = c->tail;
    if (c->key_of[s] >= 0) c->slot_of[c->key_of[s]] = -1;
    c->key_of[s] = -1;
    unsigned char *dst = c->slab + (size_t)s * c->blob_size;
    c->misses++;
    if (c->load(c->ctx, layer, expert, dst) != 0) return NULL;
    c->bytes_read += c->blob_size;
    c->key_of[s] = key;
    c->slot_of[key] = s;
    unlink_slot(c, s);
    push_front(c, s);
    return dst;
}

void mia_cache_free(mia_expert_cache *c) {
    free(c->slab);
    free(c->slot_of);
    free(c->key_of);
    free(c->prev);
    free(c->next);
    memset(c, 0, sizeof(*c));
}
