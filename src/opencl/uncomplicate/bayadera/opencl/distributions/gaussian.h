#ifndef M_LOG_SQRT_2PI_F
#define M_LOG_SQRT_2PI_F 0.9189385332046727f
#endif

inline REAL gaussian_log(REAL mu, REAL sigma, REAL x) {
    return (x - mu) * (x - mu) / (-2.0f * sigma * sigma)
        - native_log(sigma) - M_LOG_SQRT_2PI_F;
}

// ============= With params ========================================

inline REAL gaussian_logpdf(__constant const REAL* params, REAL* x) {
    return gaussian_log(params[0], params[1], x[0]);
}
