#include "mia/matmul.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__APPLE__) && defined(__BLOCKS__)
#include <dispatch/dispatch.h>
#define MIA_HAVE_GCD 1
#endif

/* ---- scalar kernels ---------------------------------------------------- */

static float dot_f32_scalar(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float dot_q8_scalar(const mia_block_q8 *a, const mia_block_q8 *b,
                           int nb) {
    float s = 0.0f;
    for (int i = 0; i < nb; i++) {
        int32_t acc = 0;
        for (int j = 0; j < MIA_QK; j++) acc += a[i].q[j] * b[i].q[j];
        s += (float)acc * a[i].d * b[i].d;
    }
    return s;
}

/* ---- SIMD kernels ------------------------------------------------------ */

#if defined(__ARM_NEON)
static float dot_f32_fast(const float *a, const float *b, int n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    float32x4_t s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static float dot_q8_fast(const mia_block_q8 *a, const mia_block_q8 *b,
                         int nb) {
    float32x4_t acc = vdupq_n_f32(0);
    for (int i = 0; i < nb; i++) {
        int32x4_t s = vdupq_n_s32(0);
        s = vdotq_s32(s, vld1q_s8(a[i].q), vld1q_s8(b[i].q));
        s = vdotq_s32(s, vld1q_s8(a[i].q + 16), vld1q_s8(b[i].q + 16));
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(s), a[i].d * b[i].d);
    }
    return vaddvq_f32(acc);
}
#else
#define dot_f32_fast dot_f32_scalar
#define dot_q8_fast dot_q8_scalar
#endif

/* ---- row loops --------------------------------------------------------- */

static void rows_kernel(float *out, const void *w, mia_wtype t, const float *x,
                        const mia_block_q8 *xq, int r0, int r1, int cols,
                        int fast) {
    if (t == MIA_F32) {
        const float *wf = w;
        for (int r = r0; r < r1; r++)
            out[r] = fast ? dot_f32_fast(wf + (size_t)r * cols, x, cols)
                          : dot_f32_scalar(wf + (size_t)r * cols, x, cols);
        return;
    }
    const int nb = cols / MIA_QK;
    const mia_block_q8 *wq = w;
    for (int r = r0; r < r1; r++)
        out[r] = fast ? dot_q8_fast(wq + (size_t)r * nb, xq, nb)
                      : dot_q8_scalar(wq + (size_t)r * nb, xq, nb);
}

void mia_matvec_scalar(float *out, const void *w, mia_wtype t, const float *x,
                       mia_block_q8 *xq, int rows, int cols) {
    if (t == MIA_Q8_0) mia_quantize_row_q8(x, xq, cols);
    rows_kernel(out, w, t, x, xq, 0, rows, cols, 0);
}

enum { CHUNK_ROWS = 64, PARALLEL_MIN_WORK = 1 << 16 };

void mia_matvec(float *out, const void *w, mia_wtype t, const float *x,
                mia_block_q8 *xq, int rows, int cols) {
    if (t == MIA_Q8_0) mia_quantize_row_q8(x, xq, cols);
#ifdef MIA_HAVE_GCD
    if ((long)rows * cols >= PARALLEL_MIN_WORK && rows > CHUNK_ROWS) {
        size_t chunks = (size_t)(rows + CHUNK_ROWS - 1) / CHUNK_ROWS;
        dispatch_apply(chunks, DISPATCH_APPLY_AUTO, ^(size_t c) {
          int r0 = (int)c * CHUNK_ROWS;
          int r1 = r0 + CHUNK_ROWS < rows ? r0 + CHUNK_ROWS : rows;
          rows_kernel(out, w, t, x, xq, r0, r1, cols, 1);
        });
        return;
    }
#endif
    rows_kernel(out, w, t, x, xq, 0, rows, cols, 1);
}
