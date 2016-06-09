inline REAL lbinco(const REAL n, const REAL k) {
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

inline REAL binomial_log(const REAL n, const REAL p, const REAL k) {
    return (k * native_log(p)) + ((n - k) * native_log(1 - p));
}

// ============= With params ========================================

inline REAL binomial_mcmc_logpdf(__constant const REAL* params, const REAL* x) {
    return binomial_log(params[0], params[1], x[0]);
}

inline REAL binomial_logpdf(__constant const REAL* params, const REAL* x) {
    return binomial_log(params[0], params[1], x[0]) + params[2];
}

inline REAL binomial_loglik(__constant const REAL* params, const REAL* p) {
    return binomial_log(params[0], p[0], params[1]);
}
