#ifndef M_LOG_SQRT_2PI_F
#define M_LOG_SQRT_2PI_F 0.9189385332046727f
#endif

inline float gaussian_log(float mu, float sigma, float x) {
    return (x - mu) * (x - mu) / (-2.0f * sigma * sigma)
        - native_log(sigma) - M_LOG_SQRT_2PI_F;
}

// ============= With params ========================================

inline float gaussian_logpdf(__constant const float* params, float* x) {
    return gaussian_log(params[0], params[1], x[0]);
}
