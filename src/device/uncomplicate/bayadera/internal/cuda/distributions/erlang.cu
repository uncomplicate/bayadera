extern "C" {

    inline REAL erlang_log_unscaled(const REAL lambda, const REAL k, const REAL x) {
        return (k - 1) * log(x) - lambda * x;
    }

    inline REAL erlang_log_scale(const REAL lambda, const REAL k) {
        return k * log(lambda) - lgamma(k);
    }

    inline REAL erlang_log(const REAL lambda, const REAL k, const REAL x) {
        return erlang_log_unscaled(lambda, k, x) + erlang_log_scale(lambda, k);
    }

// ============= With params ========================================

    REAL erlang_mcmc_logpdf(const uint32_t params_len, const REAL* params, const uint32_t dim, const REAL* x) {
        return erlang_log_unscaled(params[0], params[1], x[0]);
    }


    REAL erlang_logpdf(const uint32_t params_len, const REAL* params, const uint32_t dim, const REAL* x) {
        return erlang_log_unscaled(params[0], params[1], x[0]) + params[2];
    }
}
