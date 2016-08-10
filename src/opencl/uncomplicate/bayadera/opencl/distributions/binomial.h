inline REAL lbinco(const REAL n, const REAL k) {
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

inline REAL binomial_log_unscaled(const REAL n, const REAL p, const REAL k) {
    return (k * native_log(p)) + ((n - k) * native_log(1 - p));
}

inline REAL binomial_log(const REAL n, const REAL p, const REAL k) {
    return (k * native_log(p)) + ((n - k) * native_log(1 - p)) + lbinco(n, k);
}

inline bool binomial_check_p(const REAL p) {
    return (0.0f < p) && (p < 1.0f);
}

inline bool binomial_check_nk(const REAL n, const REAL k) {
    return (0.0f <= k) && (k <= n);
}

inline bool binomial_check(const REAL n, const REAL p, const REAL k) {
    return binomial_check_nk(n, k) && binomial_check_p(p);
}

// ============= With params ========================================

REAL binomial_mcmc_logpdf(__constant const REAL* params, const REAL* k) {
    return binomial_log_unscaled(params[0], params[1], k[0]);
}

REAL binomial_logpdf(__constant const REAL* params, const REAL* k) {
    bool valid = binomial_check_nk(params[0], k[0]);
    return valid ?
        binomial_log(params[0], params[1], k[0]) : NAN;
}

REAL binomial_loglik(__constant const REAL* params, const REAL* p) {
    const bool valid = binomial_check_p(p[0]);
    return valid ? binomial_log_unscaled(params[0], p[0], params[1]) : NAN;
}
