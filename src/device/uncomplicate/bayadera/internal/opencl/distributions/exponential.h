inline REAL exponential_log_unscaled(const REAL lambda, const REAL x) {
    return - lambda * x;
}

inline REAL exponential_log(const REAL lambda, const REAL x) {
    return native_log(lambda) - lambda * x;
}

// ============= With params ========================================

REAL exponential_mcmc_logpdf(const uint params_len, const REAL* params, const uint dim, const REAL* x) {
    return (0.0f < x[0]) ? exponential_log_unscaled(params[0], x[0]) : NAN;
}


REAL exponential_logpdf(const uint params_len, const REAL* params, const uint dim, const REAL* x) {
    return (0.0f < x[0]) ? exponential_log_unscaled(params[0], x[0]) + params[1] :NAN;
}
