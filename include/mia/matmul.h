#ifndef MIA_MATMUL_H
#define MIA_MATMUL_H

#include "mia/quant.h"

/*
 * out[r] = sum_c W[r][c] * x[c], W stored row-major as `t`.
 * For Q8_0, `xq` is scratch for cols/MIA_QK blocks (x gets quantized once);
 * for F32 it may be NULL. Rows are split across cores when large.
 */
void mia_matvec(float *out, const void *w, mia_wtype t, const float *x,
                mia_block_q8 *xq, int rows, int cols);

/* Single-threaded portable kernel; reference for testing the SIMD path. */
void mia_matvec_scalar(float *out, const void *w, mia_wtype t, const float *x,
                       mia_block_q8 *xq, int rows, int cols);

#endif
