#include "check.h"
#include "mia/quant.h"

static void test_row_bytes_by_type(void) {
    CHECK(mia_row_bytes(MIA_F32, 64) == 256);
    CHECK(mia_row_bytes(MIA_Q8_0, 64) == 2 * sizeof(mia_block_q8));
    CHECK(sizeof(mia_block_q8) == 36);
}

static void test_q8_roundtrip_error_is_bounded(void) {
    enum { N = 256 };
    float x[N], y[N];
    mia_block_q8 q[N / MIA_QK];
    for (int i = 0; i < N; i++) x[i] = rand_uniform() * 3.0f;
    mia_quantize_row_q8(x, q, N);
    mia_dequantize_row(MIA_Q8_0, q, y, N);
    for (int b = 0; b < N / MIA_QK; b++) {
        float half_step = q[b].d * 0.5f + 1e-7f;
        for (int i = 0; i < MIA_QK; i++)
            CHECK(fabsf(x[b * MIA_QK + i] - y[b * MIA_QK + i]) <= half_step);
    }
}

static void test_q8_zero_block_is_exact(void) {
    float x[MIA_QK] = {0}, y[MIA_QK];
    mia_block_q8 q;
    mia_quantize_row_q8(x, &q, MIA_QK);
    mia_dequantize_row(MIA_Q8_0, &q, y, MIA_QK);
    CHECK(q.d == 0.0f);
    for (int i = 0; i < MIA_QK; i++) CHECK(y[i] == 0.0f);
}

static void test_q8_max_maps_to_127(void) {
    float x[MIA_QK] = {0};
    x[3] = -2.0f;
    x[7] = 1.0f;
    mia_block_q8 q;
    mia_quantize_row_q8(x, &q, MIA_QK);
    CHECK(q.q[3] == -127);
    CHECK(q.q[7] == 64); /* round(1.0 * 127/2) = round(63.5) = 64 */
}

static void test_f32_dequantize_copies(void) {
    float x[4] = {1, -2, 3.5f, 0}, y[4];
    mia_dequantize_row(MIA_F32, x, y, 4);
    for (int i = 0; i < 4; i++) CHECK(y[i] == x[i]);
}

int main(void) {
    RUN(test_row_bytes_by_type);
    RUN(test_q8_roundtrip_error_is_bounded);
    RUN(test_q8_zero_block_is_exact);
    RUN(test_q8_max_maps_to_127);
    RUN(test_f32_dequantize_copies);
    return CHECK_SUMMARY();
}
