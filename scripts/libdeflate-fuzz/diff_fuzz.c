/*
 * Differential libFuzzer harness: patched libdeflate vs. unpatched upstream.
 *
 * The unpatched library is linked in with every defined global symbol renamed
 * to "up_<name>" (see build.sh); the patched one keeps its normal names.
 *
 * Input: byte 0 selects the output buffer size, the rest is the payload. The
 * payload is decompressed as raw DEFLATE and as gzip by both implementations.
 * A result-code mismatch aborts. On success the consumed-input count, the
 * output size, and the output bytes must match as well; the successful call
 * is then repeated on the same decompressors with an exactly-sized output
 * buffer (no actual_out_nbytes) and a buffer one byte too small.
 * Output buffers are flanked by unmapped guard pages.
 */
#include <libdeflate.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define UP(ret, name, args) \
	ret up_##name args __asm__("up_libdeflate_" #name)

UP(struct libdeflate_decompressor *, alloc_decompressor, (void));
UP(void, free_decompressor, (struct libdeflate_decompressor *));
UP(enum libdeflate_result, deflate_decompress_ex,
   (struct libdeflate_decompressor *, const void *, size_t, void *, size_t,
    size_t *, size_t *));
UP(enum libdeflate_result, gzip_decompress_ex,
   (struct libdeflate_decompressor *, const void *, size_t, void *, size_t,
    size_t *, size_t *));

typedef enum libdeflate_result (*decomp_fn)(struct libdeflate_decompressor *,
					    const void *, size_t, void *,
					    size_t, size_t *, size_t *);

static const struct {
	const char *name;
	struct libdeflate_decompressor *(*alloc)(void);
	void (*free)(struct libdeflate_decompressor *);
	decomp_fn fn[2]; /* [0] raw DEFLATE, [1] gzip */
} impl[2] = {
	{ "patched", libdeflate_alloc_decompressor,
	  libdeflate_free_decompressor,
	  { libdeflate_deflate_decompress_ex, libdeflate_gzip_decompress_ex } },
	{ "upstream", up_alloc_decompressor, up_free_decompressor,
	  { up_deflate_decompress_ex, up_gzip_decompress_ex } },
};

static const size_t ratio[] = { 1, 2, 3, 4, 5, 8, 16, 64, 258 };
#define MAX_OUT ((size_t)8 << 20)

struct gbuf { uint8_t *base, *start, *end; };

static void
gb_alloc(struct gbuf *b, size_t size)
{
	const size_t pg = sysconf(_SC_PAGESIZE);
	const size_t n = (size + pg - 1) / pg;

	b->base = mmap(NULL, (n + 2) * pg, PROT_READ | PROT_WRITE,
		       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (b->base == MAP_FAILED)
		abort();
	b->start = b->base + pg;
	b->end = b->start + n * pg;
	munmap(b->base, pg);
	munmap(b->end, pg);
}

static void
gb_free(struct gbuf *b)
{
	munmap(b->start, b->end - b->start);
}

struct res {
	enum libdeflate_result rc;
	size_t in_n, out_n;
	struct gbuf buf;
	uint8_t *out;
};

static void
mismatch(const char *what, int gz, size_t avail, int exact)
{
	fprintf(stderr, "DIFFERENTIAL MISMATCH: %s (gzip=%d avail=%zu exact=%d)\n",
		what, gz, avail, exact);
	abort();
}

/* Run both implementations on 'in' and compare. Returns the patched result. */
static enum libdeflate_result
compare(struct libdeflate_decompressor *d[2], int gz, const uint8_t *in,
	size_t insize, size_t avail, int exact, size_t *out_n)
{
	struct res r[2];

	for (int i = 0; i < 2; i++) {
		size_t *outp = exact == 1 ? NULL : &r[i].out_n;

		r[i].in_n = r[i].out_n = 0;
		gb_alloc(&r[i].buf, avail);
		r[i].out = r[i].buf.end - avail;
		r[i].rc = impl[i].fn[gz](d[i], in, insize, r[i].out, avail,
					 &r[i].in_n, outp);
	}
	if (r[0].rc != r[1].rc)
		mismatch("result code", gz, avail, exact);
	if (r[0].rc == LIBDEFLATE_SUCCESS) {
		if (exact != 1 && r[0].out_n != r[1].out_n)
			mismatch("actual_out_nbytes", gz, avail, exact);
		if (r[0].in_n != r[1].in_n)
			mismatch("actual_in_nbytes", gz, avail, exact);
		size_t n = exact == 1 ? avail : r[0].out_n;
		if (memcmp(r[0].out, r[1].out, n) != 0)
			mismatch("output bytes", gz, avail, exact);
		*out_n = n;
	}
	enum libdeflate_result rc = r[0].rc;
	gb_free(&r[0].buf);
	gb_free(&r[1].buf);
	return rc;
}

int
LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
	if (size < 1)
		return 0;
	const size_t insize = size - 1;
	size_t avail = insize * ratio[data[0] % (sizeof(ratio) / sizeof(ratio[0]))];
	struct gbuf cb;

	if (avail > MAX_OUT)
		avail = MAX_OUT;
	gb_alloc(&cb, insize);
	uint8_t *in = cb.end - insize;
	memcpy(in, data + 1, insize);

	for (int gz = 0; gz < 2; gz++) {
		struct libdeflate_decompressor *d[2] = {
			impl[0].alloc(), impl[1].alloc()
		};
		size_t n = 0;

		if (compare(d, gz, in, insize, avail, 0, &n) ==
		    LIBDEFLATE_SUCCESS) {
			compare(d, gz, in, insize, n, 1, &n);
			if (n > 0)
				compare(d, gz, in, insize, n - 1, 0, &n);
		}
		impl[0].free(d[0]);
		impl[1].free(d[1]);
	}
	gb_free(&cb);
	return 0;
}
