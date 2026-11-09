/*
 * Simulation 2: measuring the memory hierarchy of a real machine.
 *
 *   latency  : pointer chasing through a random cyclic linked list.
 *              Each load depends on the previous one, so memory-level
 *              parallelism and hardware prefetching cannot hide latency.
 *   bandwidth: STREAM-style "triad"  a[i] = b[i] + s*c[i]  and a read-only
 *              sum, for working sets from L1-sized to DRAM-sized.
 *   peak     : a register-resident FMA loop giving the compute ceiling.
 *
 * Build:  gcc -O3 -march=native -o membench membench.c
 * Run:    ./membench latency  > latency.csv
 *         ./membench bandwidth > bandwidth.csv
 *         ./membench peak
 * Each point is the median of REPS repetitions.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LINE 64
#define REPS 5

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n) {
    qsort(v, n, sizeof(double), cmp);
    return v[n / 2];
}

static uint64_t rng = 88172645463325252ULL;
static uint64_t xorshift(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}

/* ---------------------------------------------------------------- latency */
static void latency(void) {
    printf("size_kib,ns_per_load\n");
    for (size_t bytes = 4 << 10; bytes <= ((size_t)1 << 30); bytes *= 2) {
        for (int half = 0; half < 2; half++) {
            size_t sz = half ? bytes + bytes / 2 : bytes;  /* 1x and 1.5x points */
            if (sz > ((size_t)1 << 30)) break;
            size_t n = sz / LINE;
            char *buf = aligned_alloc(4096, n * LINE);
            size_t *perm = malloc(n * sizeof(size_t));
            for (size_t i = 0; i < n; i++) perm[i] = i;
            for (size_t i = n - 1; i > 0; i--) {          /* Fisher-Yates */
                size_t j = xorshift() % (i + 1), t = perm[i];
                perm[i] = perm[j]; perm[j] = t;
            }
            /* one pointer per cache line, visiting lines in random order */
            for (size_t i = 0; i < n; i++)
                *(void **)(buf + perm[i] * LINE) = buf + perm[(i + 1) % n] * LINE;
            free(perm);

            size_t steps = 20000000;
            double v[REPS];
            void **p = (void **)buf;
            for (size_t i = 0; i < n; i++) p = (void **)*p;   /* warm up */
            for (int r = 0; r < REPS; r++) {
                double t0 = now();
                for (size_t i = 0; i < steps; i++) p = (void **)*p;
                v[r] = (now() - t0) / steps * 1e9;
            }
            if (p == NULL) puts("");                 /* keep p live */
            printf("%zu,%.3f\n", sz >> 10, median(v, REPS));
            fflush(stdout);
            free(buf);
        }
    }
}

/* -------------------------------------------------------------- bandwidth */
static void bandwidth(void) {
    printf("size_kib,triad_gbs,read_gbs\n");
    for (size_t bytes = 16 << 10; bytes <= ((size_t)1536 << 20); bytes *= 2) {
        size_t n = bytes / (3 * sizeof(double));     /* 3 arrays share 'bytes' */
        double *a = aligned_alloc(64, n * 8 + 64), *b = aligned_alloc(64, n * 8 + 64),
               *c = aligned_alloc(64, n * 8 + 64);
        for (size_t i = 0; i < n; i++) { a[i] = 0; b[i] = 1; c[i] = 2; }
        size_t iters = (size_t)(4e9 / bytes) + 1;    /* ~4 GB of traffic per rep */
        double vt[REPS], vr[REPS], s = 3.0, sum = 0;
        for (int r = 0; r < REPS; r++) {
            double t0 = now();
            for (size_t it = 0; it < iters; it++) {
                for (size_t i = 0; i < n; i++) a[i] = b[i] + s * c[i];
                __asm__ volatile("" ::: "memory");
            }
            vt[r] = 3.0 * 8 * n * iters / (now() - t0) / 1e9;
            t0 = now();
            for (size_t it = 0; it < iters; it++) {
                double acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                for (size_t i = 0; i + 3 < n; i += 4) {
                    acc0 += b[i]; acc1 += b[i + 1]; acc2 += b[i + 2]; acc3 += b[i + 3];
                }
                for (size_t i = 0; i + 3 < n; i += 4) {
                    acc0 += c[i]; acc1 += c[i + 1]; acc2 += c[i + 2]; acc3 += c[i + 3];
                }
                for (size_t i = 0; i + 3 < n; i += 4) {
                    acc0 += a[i]; acc1 += a[i + 1]; acc2 += a[i + 2]; acc3 += a[i + 3];
                }
                sum += acc0 + acc1 + acc2 + acc3;
                __asm__ volatile("" ::: "memory");
            }
            vr[r] = 3.0 * 8 * n * iters / (now() - t0) / 1e9;
        }
        if (sum == 42) puts("");
        printf("%zu,%.2f,%.2f\n", bytes >> 10, median(vt, REPS), median(vr, REPS));
        fflush(stdout);
        free(a); free(b); free(c);
    }
}

/* ------------------------------------------------------------------- peak */
static void peak(void) {
    /* 16 independent accumulators x 8-wide vectors keep the FMA units busy */
    typedef double v8 __attribute__((vector_size(64)));
    v8 acc[16], m = {1.0000001, 1.0000001, 1.0000001, 1.0000001,
                     1.0000001, 1.0000001, 1.0000001, 1.0000001},
                ad = {1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9};
    for (int k = 0; k < 16; k++) acc[k] = ad * (k + 1);
    size_t iters = 50000000;
    double v[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        for (size_t i = 0; i < iters; i++)
            for (int k = 0; k < 16; k++) acc[k] = acc[k] * m + ad;
        v[r] = 2.0 * 8 * 16 * iters / (now() - t0) / 1e9;
    }
    double s = 0;
    for (int k = 0; k < 16; k++) for (int j = 0; j < 8; j++) s += acc[k][j];
    printf("peak_gflops,%.2f\n(checksum %g)\n", median(v, REPS), s);
}

/* ----------------------------------------------------- matrix multiply */
static void matmul(void) {
    /* naive i-j-k vs. tiled, N = 1024, double precision */
    const int N = 1024, T = 64;
    double *A = aligned_alloc(64, sizeof(double) * N * N),
           *B = aligned_alloc(64, sizeof(double) * N * N),
           *C = aligned_alloc(64, sizeof(double) * N * N);
    for (int i = 0; i < N * N; i++) { A[i] = 1.0 / (i % 7 + 1); B[i] = 1.0 / (i % 5 + 1); }
    double flops = 2.0 * N * N * N;

    memset(C, 0, sizeof(double) * N * N);
    double t0 = now();
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0;
            for (int k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    double tn = now() - t0;

    memset(C, 0, sizeof(double) * N * N);
    t0 = now();
    for (int ii = 0; ii < N; ii += T)
        for (int kk = 0; kk < N; kk += T)
            for (int jj = 0; jj < N; jj += T)
                for (int i = ii; i < ii + T; i++)
                    for (int k = kk; k < kk + T; k++) {
                        double a = A[i * N + k];
                        for (int j = jj; j < jj + T; j++) C[i * N + j] += a * B[k * N + j];
                    }
    double tt = now() - t0;
    printf("variant,seconds,gflops\nnaive,%.3f,%.3f\ntiled64,%.3f,%.3f\n",
           tn, flops / tn / 1e9, tt, flops / tt / 1e9);
    printf("(checksum %g)\n", C[12345]);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s latency|bandwidth|peak|matmul\n", argv[0]); return 1; }
    if (!strcmp(argv[1], "latency")) latency();
    else if (!strcmp(argv[1], "bandwidth")) bandwidth();
    else if (!strcmp(argv[1], "peak")) peak();
    else if (!strcmp(argv[1], "matmul")) matmul();
    return 0;
}
