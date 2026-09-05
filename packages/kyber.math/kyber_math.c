/*
 * kyber_math.c  -  Kyber standard math library
 * Compiled by NForce into kyber_math.obj and linked by kyberlink.
 * No external dependencies - pure C99.
 */

#include "kyber_math.h"
#include <math.h>
#include <float.h>
#include <string.h>

/* ── Activations ──────────────────────────────────────────────────────────── */

float kyber_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float kyber_leaky_relu(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

float kyber_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float kyber_tanh_act(float x) {
    return tanhf(x);
}

float kyber_swish(float x) {
    return x * kyber_sigmoid(x);
}

/* ── Scalar ops ───────────────────────────────────────────────────────────── */

float kyber_clamp(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

float kyber_lerp(float a, float b, float t) {
    return a + t * (b - a);
}

float kyber_norm1d(float x, float mean, float variance, float eps) {
    return (x - mean) / sqrtf(variance + eps);
}

/* ── Vector ops ───────────────────────────────────────────────────────────── */

void kyber_vec_relu(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = kyber_relu(in[i]);
}

void kyber_vec_sigmoid(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = kyber_sigmoid(in[i]);
}

void kyber_vec_softmax(const float* in, float* out, int n) {
    float max_val = in[0];
    for (int i = 1; i < n; i++)
        if (in[i] > max_val) max_val = in[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(in[i] - max_val);
        sum += out[i];
    }
    for (int i = 0; i < n; i++)
        out[i] /= sum;
}

void kyber_vec_add(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

void kyber_vec_mul(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = a[i] * b[i];
}

void kyber_vec_scale(const float* in, float scale, float* out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = in[i] * scale;
}

float kyber_vec_dot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

float kyber_vec_sum(const float* v, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += v[i];
    return s;
}

float kyber_vec_max(const float* v, int n) {
    float m = v[0];
    for (int i = 1; i < n; i++) if (v[i] > m) m = v[i];
    return m;
}

float kyber_vec_min(const float* v, int n) {
    float m = v[0];
    for (int i = 1; i < n; i++) if (v[i] < m) m = v[i];
    return m;
}

void kyber_vec_fill(float* v, float val, int n) {
    for (int i = 0; i < n; i++) v[i] = val;
}

void kyber_vec_copy(const float* src, float* dst, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}

/* ── Matrix ops ───────────────────────────────────────────────────────────── */

void kyber_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    /* C = A * B,  A is M×K,  B is K×N,  C is M×N */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

void kyber_mat_add(const float* A, const float* B, float* C, int rows, int cols) {
    int n = rows * cols;
    for (int i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

void kyber_mat_transpose(const float* A, float* At, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            At[j * rows + i] = A[i * cols + j];
}

/* ── Batch norm ───────────────────────────────────────────────────────────── */

void kyber_batch_norm(const float* in, float* out, int n,
                      float mean, float variance,
                      float gamma, float beta, float eps) {
    for (int i = 0; i < n; i++) {
        float norm = kyber_norm1d(in[i], mean, variance, eps);
        out[i] = gamma * norm + beta;
    }
}
