#ifndef M_LOG_SQRT_2PI_F
#define M_LOG_SQRT_2PI_F 0.9189385332046727f
#endif

inline REAL gaussian_log(const REAL mu, const REAL sigma, const REAL x) {
    return (x - mu) * (x - mu) / (-2.0f * sigma * sigma)
        - native_log(sigma) - M_LOG_SQRT_2PI_F;
}

// ============= With params ========================================

inline REAL gaussian_logpdf(__constant const REAL* params, const REAL* x) {
    return gaussian_log(params[0], params[1], x[0]);
}

/* inline REAL gaussian_logpdf(__constant const REAL* params, const REAL* x) {
    return gaussian_log(params[0], params[1], x[0]) - params[3];
    }*/

inline REAL gaussian_loglik(__constant const REAL* data, const REAL* mu_sigma) {
    const uint n = (uint) data[0];
    REAL res = data[1];
    for (uint i = 1; i < n; i++){
        res += gaussian_log(mu_sigma[0], mu_sigma[1], data[i+1]);
    }
    return res;
}
