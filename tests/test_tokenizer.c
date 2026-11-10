#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "mia/tokenizer.h"

/* Builds an MTK1 blob in memory (same format as tools/internal/tok). */
typedef struct {
    unsigned char *p;
    size_t n;
} buf;

static void put(buf *b, const void *src, size_t n) {
    b->p = realloc(b->p, b->n + n);
    memcpy(b->p + b->n, src, n);
    b->n += n;
}
static void u32(buf *b, uint32_t v) { put(b, &v, 4); }
static void piece(buf *b, const char *s) {
    uint16_t n = (uint16_t)strlen(s);
    put(b, &n, 2);
    put(b, s, n);
}

/* ids: a=0 b=1 c=2 ' '=3 ab=4 abc=5 ' abc'=6 <|im_end|>=7 '!'=8 'x'=9 */
static buf tiny_blob(void) {
    buf b = {0};
    put(&b, "MTK1", 4);
    u32(&b, 10), u32(&b, 3), u32(&b, 1);
    u32(&b, 0xFFFFFFFFu), u32(&b, 0xFFFFFFFFu), u32(&b, 7);
    const char *v[] = {"a", "b", "c", " ", "ab", "abc", " abc", "<|im_end|>",
                       "!", "x"};
    for (int i = 0; i < 10; i++) piece(&b, v[i]);
    u32(&b, 0), u32(&b, 1), u32(&b, 4); /* a + b -> ab   (rank 0) */
    u32(&b, 4), u32(&b, 2), u32(&b, 5); /* ab + c -> abc (rank 1) */
    u32(&b, 3), u32(&b, 5), u32(&b, 6); /* ' ' + abc -> ' abc' */
    u32(&b, 7);
    return b;
}

static int encode(const char *text, int *ids) {
    buf b = tiny_blob();
    mia_tokenizer t;
    CHECK(mia_tok_load(&t, b.p, b.n) == 0);
    int n = mia_tok_encode(&t, text, ids, 64);
    mia_tok_free(&t);
    free(b.p);
    return n;
}

static void test_merges_follow_rank_order(void) {
    int ids[64];
    int n = encode("abc abc", ids);
    CHECK(n == 2 && ids[0] == 5 && ids[1] == 6);
}

static void test_unmergeable_bytes_stay_single(void) {
    int ids[64];
    int n = encode("ba", ids);
    CHECK(n == 2 && ids[0] == 1 && ids[1] == 0);
}

static void test_special_tokens_match_verbatim(void) {
    int ids[64];
    int n = encode("abc<|im_end|>abc", ids);
    CHECK(n == 3 && ids[0] == 5 && ids[1] == 7 && ids[2] == 5);
}

static void test_punctuation_splits_from_words(void) {
    int ids[64];
    /* "abc!abc": '!' is its own pre-token, so no merge crosses it. */
    int n = encode("abc!abc", ids);
    CHECK(n == 3 && ids[0] == 5 && ids[1] == 8 && ids[2] == 5);
}

static void test_decode_roundtrip(void) {
    buf b = tiny_blob();
    mia_tokenizer t;
    mia_tok_load(&t, b.p, b.n);
    int ids[64];
    const char *text = "abc abc!<|im_end|>";
    int n = mia_tok_encode(&t, text, ids, 64);
    char out[64] = {0};
    size_t pos = 0;
    for (int i = 0; i < n; i++) {
        int len;
        const unsigned char *p = mia_tok_piece(&t, ids[i], &len);
        memcpy(out + pos, p, (size_t)len);
        pos += (size_t)len;
    }
    CHECK(strcmp(out, text) == 0);
    CHECK(mia_tok_is_stop(&t, 7));
    CHECK(!mia_tok_is_stop(&t, 5));
    mia_tok_free(&t);
    free(b.p);
}

static void test_rejects_bad_blob(void) {
    mia_tokenizer t;
    unsigned char junk[] = "NOPE....";
    CHECK(mia_tok_load(&t, junk, sizeof(junk)) != 0);
}

int main(void) {
    RUN(test_merges_follow_rank_order);
    RUN(test_unmergeable_bytes_stay_single);
    RUN(test_special_tokens_match_verbatim);
    RUN(test_punctuation_splits_from_words);
    RUN(test_decode_roundtrip);
    RUN(test_rejects_bad_blob);
    return CHECK_SUMMARY();
}
