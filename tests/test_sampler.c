#include "check.h"
#include "mia/sampler.h"

static void test_greedy_picks_argmax(void) {
    float logits[5] = {0.1f, 2.0f, -1.0f, 1.9f, 0.0f};
    mia_sampler s;
    mia_sampler_init(&s, 0.0f, 1.0f, 5, 1);
    CHECK(mia_sample(&s, logits) == 1);
    mia_sampler_free(&s);
}

static void test_tiny_top_p_keeps_only_best(void) {
    mia_sampler s;
    mia_sampler_init(&s, 1.0f, 1e-6f, 5, 7);
    for (int i = 0; i < 50; i++) {
        float logits[5] = {0.1f, 2.0f, -1.0f, 1.9f, 0.0f}; /* overwritten */
        CHECK(mia_sample(&s, logits) == 1);
    }
    mia_sampler_free(&s);
}

static void test_equal_logits_sample_both(void) {
    int counts[2] = {0, 0};
    mia_sampler s;
    mia_sampler_init(&s, 1.0f, 1.0f, 2, 42);
    for (int i = 0; i < 2000; i++) {
        float logits[2] = {3.0f, 3.0f};
        counts[mia_sample(&s, logits)]++;
    }
    CHECK(counts[0] > 850 && counts[1] > 850);
    mia_sampler_free(&s);
}

int main(void) {
    RUN(test_greedy_picks_argmax);
    RUN(test_tiny_top_p_keeps_only_best);
    RUN(test_equal_logits_sample_both);
    return CHECK_SUMMARY();
}
