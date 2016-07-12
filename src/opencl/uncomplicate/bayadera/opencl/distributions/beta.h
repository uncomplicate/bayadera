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

REAL beta_mcmc_logpdf(__constant const REAL* params, REAL* x) {
    const REAL x0 = x[0];
    const bool valid = (0.0f <= x0) && (x0 <= 1.0f);
    return valid ? beta_log_unscaled(params[0], params[1], x0) : NAN;
}

REAL beta_logpdf(__constant const REAL* params, REAL* x) {
    const REAL x0 = x[0];
    const bool valid = (0.0f <= x0) && (x0 <= 1.0f);
    return valid ? beta_log_unscaled(params[0], params[1], x0) + params[2] : NAN;
}
