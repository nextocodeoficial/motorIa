#ifndef MIA_QUANT_H
#define MIA_QUANT_H

#include <stddef.h>
#include <stdint.h>

#define MIA_QK 32 /* values per quantization block */

typedef enum { MIA_F32 = 0, MIA_Q8_0 = 1 } mia_wtype;

/* Symmetric 8-bit block: value = d * q. 36 bytes, shared with tools/. */
typedef struct {
    float d;
    int8_t q[MIA_QK];
} mia_block_q8;

/* Bytes used by one row of `cols` values (cols % MIA_QK == 0 for Q8_0). */
size_t mia_row_bytes(mia_wtype t, int cols);

void mia_quantize_row_q8(const float *x, mia_block_q8 *y, int n);
void mia_dequantize_row(mia_wtype t, const void *src, float *dst, int n);

#endif
