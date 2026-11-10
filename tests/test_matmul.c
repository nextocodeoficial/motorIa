#include <stdlib.h>

#include "check.h"
#include "mia/matmul.h"

/* Plain double-precision reference: out = W x. */
static void naive(double *out, const float *w, const float *x, int rows,
                  int cols) {
    for (int r = 0; r < rows; r++) {
        double s = 0;
        for (int c = 0; c < cols; c++) s += (double)w[r * cols + c] * x[c];
        out[r] = s;
    }
}

static void check_f32(int rows, int cols) {
    float *w = malloc(sizeof(float) * rows * cols);
    float *x = malloc(sizeof(float) * cols);
    float *out = malloc(sizeof(float) * rows);
    double *ref = malloc(sizeof(double) * rows);
    for (int i = 0; i < rows * cols; i++) w[i] = rand_uniform();
    for (int i = 0; i < cols; i++) x[i] = rand_uniform();
    naive(ref, w, x, rows, cols);
    mia_matvec(out, w, MIA_F32, x, NULL, rows, cols);
    for (int r = 0; r < rows; r++) CHECK_NEAR(out[r], ref[r], 1e-3);
    free(w), free(x), free(out), free(ref);
}

static void test_f32_small(void) { check_f32(7, 32); }
static void test_f32_large_parallel(void) { check_f32(1024, 512); }

static void check_q8(int rows, int cols) {
    float *w = malloc(sizeof(float) * rows * cols);
    float *x = malloc(sizeof(float) * cols);
    size_t rb = mia_row_bytes(MIA_Q8_0, cols);
    unsigned char *wq = malloc(rb * rows);
    mia_block_q8 *xq = malloc(sizeof(mia_block_q8) * (cols / MIA_QK));
    float *fast = malloc(sizeof(float) * rows);
    float *scalar = malloc(sizeof(float) * rows);
    double *ref = malloc(sizeof(double) * rows);
    for (int i = 0; i < rows * cols; i++) w[i] = rand_uniform();
    for (int i = 0; i < cols; i++) x[i] = rand_uniform();
    for (int r = 0; r < rows; r++)
        mia_quantize_row_q8(w + (size_t)r * cols,
                            (mia_block_q8 *)(wq + rb * r), cols);
    naive(ref, w, x, rows, cols);

    mia_matvec(fast, wq, MIA_Q8_0, x, xq, rows, cols);
    mia_matvec_scalar(scalar, wq, MIA_Q8_0, x, xq, rows, cols);
    for (int r = 0; r < rows; r++) {
        /* SIMD and scalar kernels compute the same integer dot products. */
        CHECK_NEAR(fast[r], scalar[r], 1e-4 * (1 + fabs(scalar[r])));
        /* Quantizing both sides keeps us close to the exact product. */
        CHECK_NEAR(fast[r], ref[r], 0.02 * sqrt((double)cols));
    }
    free(w), free(x), free(wq), free(xq), free(fast), free(scalar), free(ref);
}

static void test_q8_small(void) { check_q8(5, 64); }
static void test_q8_large_parallel(void) { check_q8(2048, 768); }

int main(void) {
    RUN(test_f32_small);
    RUN(test_f32_large_parallel);
    RUN(test_q8_small);
    RUN(test_q8_large_parallel);
    return CHECK_SUMMARY();
}
