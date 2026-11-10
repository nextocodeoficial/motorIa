#include <string.h>

#include "check.h"
#include "mia/expert_cache.h"

typedef struct {
    int calls;
    int fail_key; /* -1: never fail */
} fake_disk;

static int fake_load(void *ctx, int layer, int expert, void *dst) {
    fake_disk *d = ctx;
    int key = layer * 10 + expert;
    d->calls++;
    if (key == d->fail_key) return -1;
    memset(dst, key, 64);
    return 0;
}

static void test_miss_then_hit(void) {
    fake_disk d = {0, -1};
    mia_expert_cache c;
    CHECK(mia_cache_init(&c, 2, 10, 64, 3, fake_load, &d) == 0);
    const unsigned char *a = mia_cache_get(&c, 1, 2);
    CHECK(a && a[0] == 12 && a[63] == 12);
    const unsigned char *b = mia_cache_get(&c, 1, 2);
    CHECK(a == b);
    CHECK(d.calls == 1);
    CHECK(c.hits == 1 && c.misses == 1 && c.bytes_read == 64);
    mia_cache_free(&c);
}

static void test_evicts_least_recently_used(void) {
    fake_disk d = {0, -1};
    mia_expert_cache c;
    mia_cache_init(&c, 1, 10, 64, 2, fake_load, &d);
    mia_cache_get(&c, 0, 1);
    mia_cache_get(&c, 0, 2);
    mia_cache_get(&c, 0, 1); /* 1 is now most recent; 2 is LRU */
    mia_cache_get(&c, 0, 3); /* evicts 2 */
    CHECK(d.calls == 3);
    mia_cache_get(&c, 0, 1); /* still cached */
    CHECK(d.calls == 3);
    const unsigned char *p = mia_cache_get(&c, 0, 2); /* reload */
    CHECK(d.calls == 4 && p[0] == 2);
    mia_cache_free(&c);
}

static void test_single_slot_still_works(void) {
    fake_disk d = {0, -1};
    mia_expert_cache c;
    mia_cache_init(&c, 1, 10, 64, 1, fake_load, &d);
    for (int i = 0; i < 5; i++) {
        const unsigned char *p = mia_cache_get(&c, 0, i);
        CHECK(p && p[0] == i);
    }
    CHECK(c.misses == 5);
    mia_cache_free(&c);
}

static void test_load_failure_does_not_poison_slot(void) {
    fake_disk d = {0, 3};
    mia_expert_cache c;
    mia_cache_init(&c, 1, 10, 64, 2, fake_load, &d);
    CHECK(mia_cache_get(&c, 0, 3) == NULL);
    d.fail_key = -1;
    const unsigned char *p = mia_cache_get(&c, 0, 3);
    CHECK(p && p[0] == 3);
    CHECK(d.calls == 2);
    mia_cache_free(&c);
}

static void test_rejects_out_of_range(void) {
    fake_disk d = {0, -1};
    mia_expert_cache c;
    mia_cache_init(&c, 1, 4, 64, 2, fake_load, &d);
    CHECK(mia_cache_get(&c, 0, 4) == NULL);
    CHECK(mia_cache_get(&c, 1, 0) == NULL);
    CHECK(d.calls == 0);
    mia_cache_free(&c);
}

int main(void) {
    RUN(test_miss_then_hit);
    RUN(test_evicts_least_recently_used);
    RUN(test_single_slot_still_works);
    RUN(test_load_failure_does_not_poison_slot);
    RUN(test_rejects_out_of_range);
    return CHECK_SUMMARY();
}
