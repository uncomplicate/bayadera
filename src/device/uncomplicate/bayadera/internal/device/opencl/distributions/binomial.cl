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

inline REAL binomial_mcmc_logpdf(const uint data_len, const uint params_len, const REAL* params,
                                 const uint dim, const REAL* k) {
    return binomial_log_unscaled(params[0], params[1], k[0]);
}

inline REAL binomial_logpdf(const uint data_len, const uint params_len, const REAL* params,
                            const uint dim, const REAL* k) {
    return binomial_log(params[0], params[1], k[0]);
}

inline REAL binomial_loglik(const uint data_len, const REAL* data, const uint dim, const REAL* p) {
    return binomial_log_unscaled(data[0], p[0], data[1]);
}
