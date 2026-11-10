#include <stdlib.h>
#include <string.h>

#include "mia/tokenizer.h"

#define EMPTY_RANK 0xFFFFFFFFu

/* ---- loading ----------------------------------------------------------- */

typedef struct {
    const unsigned char *p;
    size_t n, off;
    int bad;
} reader;

static uint32_t rd32(reader *r) {
    if (r->off + 4 > r->n) return r->bad = 1, 0;
    uint32_t v;
    memcpy(&v, r->p + r->off, 4);
    r->off += 4;
    return v;
}

static uint16_t rd16(reader *r) {
    if (r->off + 2 > r->n) return r->bad = 1, 0;
    uint16_t v;
    memcpy(&v, r->p + r->off, 2);
    r->off += 2;
    return v;
}

static uint32_t hash64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    return (uint32_t)k;
}

static int opt_id(uint32_t v, int n) { return v < (uint32_t)n ? (int)v : -1; }

int mia_tok_load(mia_tokenizer *t, const unsigned char *blob, size_t n) {
    memset(t, 0, sizeof(*t));
    reader r = {blob, n, 4, 0};
    if (n < 28 || memcmp(blob, "MTK1", 4) != 0) return -1;
    uint32_t nv = rd32(&r), nm = rd32(&r), ns = rd32(&r);
    uint32_t eos = rd32(&r), ims = rd32(&r), ime = rd32(&r);
    if (nv == 0 || nv > (1u << 24) || nm > (1u << 24) || ns > nv) return -1;
    t->n_vocab = (int)nv;
    t->eos = opt_id(eos, (int)nv);
    t->im_start = opt_id(ims, (int)nv);
    t->im_end = opt_id(ime, (int)nv);

    t->off = malloc(sizeof(uint32_t) * nv);
    t->len = malloc(sizeof(uint16_t) * nv);
    t->bytes = malloc(n); /* pieces total < blob size */
    if (!t->off || !t->len || !t->bytes) goto fail;
    for (int b = 0; b < 256; b++) t->byte_id[b] = -1;
    uint32_t pos = 0;
    for (uint32_t i = 0; i < nv; i++) {
        uint16_t len = rd16(&r);
        if (r.bad || r.off + len > n) goto fail;
        memcpy(t->bytes + pos, blob + r.off, len);
        t->off[i] = pos, t->len[i] = len;
        if (len == 1 && t->byte_id[t->bytes[pos]] < 0)
            t->byte_id[t->bytes[pos]] = (int)i;
        pos += len, r.off += len;
    }

    t->mcap = 16;
    while (t->mcap < nm * 2) t->mcap <<= 1;
    t->mkey = malloc(sizeof(uint64_t) * t->mcap);
    t->mrank = malloc(sizeof(uint32_t) * t->mcap);
    t->mres = malloc(sizeof(uint32_t) * t->mcap);
    if (!t->mkey || !t->mrank || !t->mres) goto fail;
    for (uint32_t i = 0; i < t->mcap; i++) t->mrank[i] = EMPTY_RANK;
    for (uint32_t rank = 0; rank < nm; rank++) {
        uint32_t a = rd32(&r), b = rd32(&r), res = rd32(&r);
        if (r.bad || a >= nv || b >= nv || res >= nv) goto fail;
        uint64_t key = (uint64_t)a << 32 | b;
        uint32_t h = hash64(key) & (t->mcap - 1);
        while (t->mrank[h] != EMPTY_RANK && t->mkey[h] != key)
            h = (h + 1) & (t->mcap - 1);
        if (t->mrank[h] != EMPTY_RANK) continue; /* keep best (first) rank */
        t->mkey[h] = key, t->mrank[h] = rank, t->mres[h] = res;
    }

    t->n_special = (int)ns;
    t->special = malloc(sizeof(int) * (ns ? ns : 1));
    if (!t->special) goto fail;
    for (uint32_t i = 0; i < ns; i++) {
        uint32_t id = rd32(&r);
        if (r.bad || id >= nv) goto fail;
        t->special[i] = (int)id;
    }
    if (r.bad) goto fail;
    return 0;
fail:
    mia_tok_free(t);
    return -1;
}

void mia_tok_free(mia_tokenizer *t) {
    free(t->off), free(t->len), free(t->bytes);
    free(t->mkey), free(t->mrank), free(t->mres), free(t->special);
    memset(t, 0, sizeof(*t));
}

const unsigned char *mia_tok_piece(const mia_tokenizer *t, int id, int *len) {
    if (id < 0 || id >= t->n_vocab) return *len = 0, (const unsigned char *)"";
    *len = t->len[id];
    return t->bytes + t->off[id];
}

int mia_tok_is_stop(const mia_tokenizer *t, int id) {
    return id >= 0 && (id == t->eos || id == t->im_end);
}

/* ---- pre-tokenization -------------------------------------------------- */

enum { C_L, C_N, C_SP, C_NL, C_P };

static int char_class(const unsigned char *s, size_t n, size_t *clen) {
    unsigned char c = s[0];
    if (c < 0x80) {
        *clen = 1;
        if ((c | 32) >= 'a' && (c | 32) <= 'z') return C_L;
        if (c >= '0' && c <= '9') return C_N;
        if (c == '\n' || c == '\r') return C_NL;
        if (c == ' ' || c == '\t' || c == '\v' || c == '\f') return C_SP;
        return C_P;
    }
    size_t l = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    *clen = l < n ? l : n;
    if (c == 0xC2) return C_P; /* U+0080..U+00BF: ¡ ¿ « » ° ... */
    if ((c == 0xE2 || c == 0xE3) && n > 1 && s[1] == 0x80) return C_P;
    return C_L;
}

static size_t run_of(const unsigned char *s, size_t n, size_t i, int a,
                     int b) {
    while (i < n) {
        size_t l;
        int k = char_class(s + i, n - i, &l);
        if (k != a && k != b) break;
        i += l;
    }
    return i;
}

static int lower(unsigned char c) { return c | 32; }

/* Length of the next pre-token at s (n > 0). Mirrors the Qwen2 regex. */
static size_t pretoken_len(const unsigned char *s, size_t n) {
    size_t l0;
    int c0 = char_class(s, n, &l0);

    if (s[0] == '\'' && n >= 2) { /* 's 't 're 've 'm 'll 'd */
        int a = lower(s[1]), b = n >= 3 ? lower(s[2]) : 0;
        if (n >= 3 && ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
                       (a == 'l' && b == 'l')))
            return 3;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    }
    if (c0 == C_L) return run_of(s, n, 0, C_L, C_L);
    if (c0 != C_NL && c0 != C_N && l0 < n) {
        size_t l1;
        if (char_class(s + l0, n - l0, &l1) == C_L)
            return run_of(s, n, l0, C_L, C_L);
    }
    if (c0 == C_N) return l0;
    {
        size_t i = 0, l1;
        if (s[0] == ' ' && n > 1 && char_class(s + 1, n - 1, &l1) == C_P) i = 1;
        if (char_class(s + i, n - i, &l1) == C_P) {
            i = run_of(s, n, i, C_P, C_P);
            return run_of(s, n, i, C_NL, C_NL);
        }
    }
    /* whitespace */
    size_t end = run_of(s, n, 0, C_SP, C_NL), last_nl = 0;
    for (size_t i = 0; i < end; i++)
        if (s[i] == '\n' || s[i] == '\r') last_nl = i + 1;
    if (last_nl) return last_nl;
    if (end == n || end == 1) return end;
    return end - 1; /* leave one space to prefix the next word */
}

/* ---- BPE --------------------------------------------------------------- */

static int lookup_merge(const mia_tokenizer *t, int a, int b, uint32_t *rank,
                        int *res) {
    uint64_t key = (uint64_t)(uint32_t)a << 32 | (uint32_t)b;
    uint32_t h = hash64(key) & (t->mcap - 1);
    while (t->mrank[h] != EMPTY_RANK) {
        if (t->mkey[h] == key) {
            *rank = t->mrank[h], *res = (int)t->mres[h];
            return 1;
        }
        h = (h + 1) & (t->mcap - 1);
    }
    return 0;
}

static int bpe(const mia_tokenizer *t, const unsigned char *s, size_t n,
               int *out, int room) {
    int *ids = malloc(sizeof(int) * (n ? n : 1));
    if (!ids) return -1;
    int m = 0;
    for (size_t i = 0; i < n; i++)
        if (t->byte_id[s[i]] >= 0) ids[m++] = t->byte_id[s[i]];
    for (;;) {
        uint32_t best = EMPTY_RANK, rank;
        int at = -1, res, best_res = 0;
        for (int i = 0; i + 1 < m; i++)
            if (lookup_merge(t, ids[i], ids[i + 1], &rank, &res) &&
                rank < best)
                best = rank, at = i, best_res = res;
        if (at < 0) break;
        ids[at] = best_res;
        memmove(ids + at + 1, ids + at + 2, sizeof(int) * (size_t)(m - at - 2));
        m--;
    }
    if (m > room) {
        free(ids);
        return -1;
    }
    memcpy(out, ids, sizeof(int) * (size_t)m);
    free(ids);
    return m;
}

static int special_at(const mia_tokenizer *t, const unsigned char *s,
                      size_t n, size_t *len) {
    int best = -1;
    *len = 0;
    for (int i = 0; i < t->n_special; i++) {
        int id = t->special[i];
        size_t l = t->len[id];
        if (l > *len && l <= n && memcmp(s, t->bytes + t->off[id], l) == 0)
            best = id, *len = l;
    }
    return best;
}

int mia_tok_encode(const mia_tokenizer *t, const char *text, int *out,
                   int max) {
    const unsigned char *s = (const unsigned char *)text;
    size_t n = strlen(text), pos = 0;
    int count = 0;
    while (pos < n) {
        size_t sl;
        int sp = special_at(t, s + pos, n - pos, &sl);
        if (sp >= 0) {
            if (count >= max) return -1;
            out[count++] = sp;
            pos += sl;
            continue;
        }
        size_t seg = pos + 1; /* segment runs until the next special token */
        while (seg < n && special_at(t, s + seg, n - seg, &sl) < 0) seg++;
        while (pos < seg) {
            size_t pl = pretoken_len(s + pos, seg - pos);
            int got = bpe(t, s + pos, pl, out + count, max - count);
            if (got < 0) return -1;
            count += got;
            pos += pl;
        }
    }
    return count;
}
