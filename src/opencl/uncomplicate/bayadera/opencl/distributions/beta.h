inline float lbeta (const float a, const float b) {
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}

inline float beta (const float a, const float b) {
    return native_exp(lbeta(a, b));
}

inline float beta_log(const float a, const float b, const float x) {
    return ((a - 1.0f) * native_log(x)) + ((b - 1.0f) * native_log(1 - x));
}

// ============= With params ========================================

inline float beta_mcmc_logpdf(__constant const float* params, float x) {
    return beta_log(params[0], params[1], x);
}

inline float beta_logpdf(__constant const float* params, float x) {
    return beta_log(params[0], params[1], x) - params[2];
}
