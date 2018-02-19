inline REAL lbeta (const REAL a, const REAL b) {
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}

inline REAL beta (const REAL a, const REAL b) {
    return native_exp(lbeta(a, b));
}

inline REAL beta_log_unscaled(const REAL a, const REAL b, const REAL x) {
    return (a - 1.0f) * native_log(x) + (b - 1.0f) * native_log(1 - x);
}

inline REAL beta_log(const REAL a, const REAL b, const REAL x) {
    return beta_log_unscaled(a, b, x) - lbeta(a, b);
}

// ============= With params ========================================

REAL beta_mcmc_logpdf(const REAL* params, const REAL* x) {
    return beta_log_unscaled(params[0], params[1], x[0]);
}

REAL beta_logpdf(const REAL* params, const REAL* x) {
    return beta_log_unscaled(params[0], params[1], x[0]) + params[2];
}
