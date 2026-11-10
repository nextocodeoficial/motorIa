#include "mia/quant.h"

#include <math.h>
#include <string.h>

size_t mia_row_bytes(mia_wtype t, int cols) {
    if (t == MIA_Q8_0) return (size_t)(cols / MIA_QK) * sizeof(mia_block_q8);
    return (size_t)cols * sizeof(float);
}

void mia_quantize_row_q8(const float *x, mia_block_q8 *y, int n) {
    for (int b = 0; b < n / MIA_QK; b++) {
        const float *xb = x + b * MIA_QK;
        float amax = 0.0f;
        for (int i = 0; i < MIA_QK; i++) {
            float a = fabsf(xb[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[b].d = d;
        for (int i = 0; i < MIA_QK; i++) {
            float v = xb[i] * id;
            y[b].q[i] = (int8_t)roundf(v);
        }
    }
}

void mia_dequantize_row(mia_wtype t, const void *src, float *dst, int n) {
    if (t == MIA_F32) {
        memcpy(dst, src, sizeof(float) * (size_t)n);
        return;
    }
    const mia_block_q8 *q = src;
    for (int b = 0; b < n / MIA_QK; b++)
        for (int i = 0; i < MIA_QK; i++)
            dst[b * MIA_QK + i] = q[b].d * (float)q[b].q[i];
}
