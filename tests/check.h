#ifndef MIA_TESTS_CHECK_H
#define MIA_TESTS_CHECK_H

#include <math.h>
#include <stdio.h>

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__,   \
                    #cond);                                                    \
            g_failures++;                                                      \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                  \
    do {                                                                       \
        double a_ = (a), b_ = (b);                                             \
        if (!(fabs(a_ - b_) <= (tol))) {                                       \
            fprintf(stderr, "%s:%d: %s=%g vs %s=%g (tol %g)\n", __FILE__,      \
                    __LINE__, #a, a_, #b, b_, (double)(tol));                  \
            g_failures++;                                                      \
        }                                                                      \
    } while (0)

#define RUN(test_fn)                                                           \
    do {                                                                       \
        int before_ = g_failures;                                              \
        test_fn();                                                             \
        printf("  %s %s\n", g_failures == before_ ? "ok  " : "FAIL", #test_fn); \
    } while (0)

#define CHECK_SUMMARY() (g_failures ? 1 : 0)

/* Deterministic xorshift RNG so tests are reproducible. */
static unsigned int g_rng = 0x9E3779B9u;
static inline float rand_uniform(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (float)(g_rng & 0xFFFFFF) / (float)0x1000000 * 2.0f - 1.0f;
}

#endif
