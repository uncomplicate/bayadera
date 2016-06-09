inline REAL lbeta (const REAL a, const REAL b) {
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}

inline REAL beta (const REAL a, const REAL b) {
    return native_exp(lbeta(a, b));
}

inline REAL beta_log(const REAL a, const REAL b, const REAL x) {
    return ((a - 1.0f) * native_log(x)) + ((b - 1.0f) * native_log(1 - x));
}

// ============= With params ========================================

inline REAL beta_mcmc_logpdf(__constant const REAL* params, REAL* x) {
    return beta_log(params[0], params[1], x[0]);
}

inline REAL beta_logpdf(__constant const REAL* params, REAL* x) {
    return beta_log(params[0], params[1], x[0]) - params[2];
}
