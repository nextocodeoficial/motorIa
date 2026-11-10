#include "mia/mia_file.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { TENSOR_ALIGN = 64, READ_CHUNK = 64 << 20 };

static size_t align_up(size_t n, size_t a) { return (n + a - 1) / a * a; }

static int pread_full(int fd, void *dst, size_t n, uint64_t off) {
    unsigned char *p = dst;
    while (n > 0) {
        size_t want = n < READ_CHUNK ? n : READ_CHUNK;
        ssize_t got = pread(fd, p, want, (off_t)off);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return -1;
        p += got, n -= (size_t)got, off += (uint64_t)got;
    }
    return 0;
}

/* Walks the dense section in the exact order of NewLayout() in format.go. */
typedef struct {
    unsigned char *base;
    size_t off;
    mia_wtype wt;
} cursor;

static const void *take(cursor *c, int rows, int cols, int quantized) {
    const void *p = c->base + c->off;
    size_t bytes = quantized ? (size_t)rows * mia_row_bytes(c->wt, cols)
                             : (size_t)rows * cols * sizeof(float);
    c->off = align_up(c->off + bytes, TENSOR_ALIGN);
    return p;
}

int mia_file_open(mia_file *f, const char *path, char *err, size_t errlen) {
    memset(f, 0, sizeof(*f));
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) {
        snprintf(err, errlen, "open %s: %s", path, strerror(errno));
        return -1;
    }
    mia_header *h = &f->h;
    if (pread_full(f->fd, h, sizeof(*h), 0) != 0 ||
        memcmp(h->magic, "MIA1", 4) != 0 || h->version != 1) {
        snprintf(err, errlen, "%s: not a MIA v1 file", path);
        goto fail;
    }
    if (h->wtype > MIA_Q8_0 || h->n_kv_heads == 0 ||
        h->n_heads % h->n_kv_heads != 0 || h->n_active > h->n_experts ||
        h->head_dim % 2 != 0 || h->dim % MIA_QK != 0 ||
        h->expert_hidden % MIA_QK != 0 ||
        (h->n_heads * h->head_dim) % MIA_QK != 0) {
        snprintf(err, errlen, "%s: unsupported model shape", path);
        goto fail;
    }
    f->wt = (mia_wtype)h->wtype;

    if (posix_memalign((void **)&f->dense, 16384, h->dense_size) != 0 ||
        pread_full(f->fd, f->dense, h->dense_size, h->dense_offset) != 0) {
        snprintf(err, errlen, "%s: cannot load %llu dense bytes", path,
                 (unsigned long long)h->dense_size);
        goto fail;
    }
    /* Expert reads bypass the unified buffer cache: RAM stays bounded. */
    fcntl(f->fd, F_NOCACHE, 1);

    int d = (int)h->dim, hd = (int)h->head_dim, V = (int)h->vocab;
    int qd = (int)h->n_heads * hd, kvd = (int)h->n_kv_heads * hd;
    cursor c = {f->dense, 0, f->wt};
    f->token_embd = take(&c, V, d, 1);
    f->layers = calloc(h->n_layers, sizeof(mia_layer));
    if (!f->layers) goto fail;
    for (uint32_t l = 0; l < h->n_layers; l++) {
        mia_layer *L = &f->layers[l];
        L->attn_norm = take(&c, 1, d, 0);
        L->wq = take(&c, qd, d, 1);
        L->wk = take(&c, kvd, d, 1);
        L->wv = take(&c, kvd, d, 1);
        L->wo = take(&c, d, qd, 1);
        L->q_norm = take(&c, 1, hd, 0);
        L->k_norm = take(&c, 1, hd, 0);
        L->ffn_norm = take(&c, 1, d, 0);
        L->router = take(&c, (int)h->n_experts, d, 0);
    }
    f->output_norm = take(&c, 1, d, 0);
    f->output = take(&c, V, d, 1);
    if (c.off != h->dense_size) {
        snprintf(err, errlen, "%s: dense layout mismatch (%zu vs %llu)", path,
                 c.off, (unsigned long long)h->dense_size);
        goto fail;
    }
    int eh = (int)h->expert_hidden;
    size_t mat = (size_t)eh * mia_row_bytes(f->wt, d);
    f->off_up = align_up(mat, TENSOR_ALIGN);
    f->off_down = f->off_up + align_up(mat, TENSOR_ALIGN);
    if (f->off_down + (size_t)d * mia_row_bytes(f->wt, eh) >
        h->expert_blob_size) {
        snprintf(err, errlen, "%s: expert blob too small", path);
        goto fail;
    }
    return 0;
fail:
    mia_file_close(f);
    return -1;
}

void mia_file_close(mia_file *f) {
    if (f->fd > 0) close(f->fd);
    free(f->dense);
    free(f->layers);
    memset(f, 0, sizeof(*f));
    f->fd = -1;
}

int mia_file_read_expert(void *ctx, int layer, int expert, void *dst) {
    const mia_file *f = ctx;
    uint64_t key = (uint64_t)layer * f->h.n_experts + (uint64_t)expert;
    return pread_full(f->fd, dst, f->h.expert_blob_size,
                      f->h.experts_offset + key * f->h.expert_blob_size);
}

unsigned char *mia_file_read_tokenizer(const mia_file *f, size_t *len) {
    *len = 0;
    if (f->h.tok_size == 0) return NULL;
    unsigned char *buf = malloc(f->h.tok_size);
    if (!buf || pread_full(f->fd, buf, f->h.tok_size, f->h.tok_offset) != 0) {
        free(buf);
        return NULL;
    }
    *len = f->h.tok_size;
    return buf;
}
