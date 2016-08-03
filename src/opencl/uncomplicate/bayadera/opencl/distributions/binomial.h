inline REAL lbinco(const REAL n, const REAL k) {
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

inline REAL binomial_log_unscaled(const REAL n, const REAL p, const REAL k) {
    return (k * native_log(p)) + ((n - k) * native_log(1 - p));
}

inline REAL binomial_log(const REAL n, const REAL p, const REAL k) {
    return (k * native_log(p)) + ((n - k) * native_log(1 - p)) + lbinco(n, k);
}

// ============= With params ========================================

REAL binomial_mcmc_logpdf(__constant const REAL* params, const REAL* x) {
    return binomial_log_unscaled(params[0], params[1], x[0]);
}

REAL binomial_logpdf(__constant const REAL* params, const REAL* x) {
    return binomial_log_unscaled(params[0], params[1], x[0]) + params[2];
}

REAL binomial_loglik(__constant const REAL* params, const REAL* p) {
    const REAL pp = p[0];
    const bool valid = (0.0f < pp) && (pp < 1.0f);
    return valid ? binomial_log_unscaled(params[0], pp, params[1]) : NAN;
}
