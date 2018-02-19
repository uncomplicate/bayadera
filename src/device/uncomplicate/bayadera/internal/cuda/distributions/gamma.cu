extern "C" {

    inline REAL gamma_log_unscaled(const REAL theta, const REAL k, const REAL x) {
        return (k - 1.0f) * log(x) - (x / theta);
    }

    inline REAL gamma_log_scale(const REAL theta, const REAL k) {
        return - lgamma(k) - k * log(theta);
    }


    inline REAL gamma_log(const REAL theta, const REAL k, const REAL x) {
        return gamma_log_unscaled(theta, k, x) + gamma_log_scale(theta, k);
    }

// ============= With params ========================================

    REAL gamma_mcmc_logpdf(const REAL* params, const REAL* x) {
        return gamma_log_unscaled(params[0], params[1], x[0]);
    }

    REAL gamma_logpdf(const REAL* params, const REAL* x) {
        return gamma_log_unscaled(params[0], params[1], x[0]) + params[2];
    }
}
