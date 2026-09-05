#pragma once
#ifdef __cplusplus
extern "C" {
#endif

/* Activations */
float  kyber_relu(float x);
float  kyber_leaky_relu(float x, float alpha);
float  kyber_sigmoid(float x);
float  kyber_tanh_act(float x);
float  kyber_swish(float x);

/* Scalar ops */
float  kyber_clamp(float x, float lo, float hi);
float  kyber_lerp(float a, float b, float t);
float  kyber_norm1d(float x, float mean, float variance, float eps);

/* Vector ops (stride-1 float arrays) */
void   kyber_vec_relu(const float* in, float* out, int n);
void   kyber_vec_sigmoid(const float* in, float* out, int n);
void   kyber_vec_softmax(const float* in, float* out, int n);
void   kyber_vec_add(const float* a, const float* b, float* out, int n);
void   kyber_vec_mul(const float* a, const float* b, float* out, int n);
void   kyber_vec_scale(const float* in, float scale, float* out, int n);
float  kyber_vec_dot(const float* a, const float* b, int n);
float  kyber_vec_sum(const float* v, int n);
float  kyber_vec_max(const float* v, int n);
float  kyber_vec_min(const float* v, int n);
void   kyber_vec_fill(float* v, float val, int n);
void   kyber_vec_copy(const float* src, float* dst, int n);

/* Matrix ops (row-major, M x K and K x N -> M x N) */
void   kyber_matmul(const float* A, const float* B, float* C, int M, int K, int N);
void   kyber_mat_add(const float* A, const float* B, float* C, int rows, int cols);
void   kyber_mat_transpose(const float* A, float* At, int rows, int cols);

/* Batch norm */
void   kyber_batch_norm(const float* in, float* out, int n,
                        float mean, float variance, float gamma, float beta, float eps);

#ifdef __cplusplus
}
#endif
