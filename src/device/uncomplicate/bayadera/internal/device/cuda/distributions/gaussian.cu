extern "C" {

#include <stdint.h>
    
#ifndef M_LOG_SQRT_2PI_F
#define M_LOG_SQRT_2PI_F 0.9189385332046727f
#endif

    inline REAL gaussian_log_unscaled(const REAL mu, const REAL sigma, const REAL x) {
        return (x - mu) * (x - mu) / (-2.0f * sigma * sigma);
    }

    inline REAL gaussian_log_scale(const REAL sigma) {
        return - log(sigma) - M_LOG_SQRT_2PI_F;
    }

    inline REAL gaussian_log(const REAL mu, const REAL sigma, const REAL x) {
        return gaussian_log_unscaled(mu, sigma, x) + gaussian_log_scale(sigma);
    }

// ============= With params ========================================

    inline REAL gaussian_mcmc_logpdf(const uint32_t data_len, const uint32_t params_len, const REAL* params,
                                     const uint32_t dim, const REAL* x) {
        return gaussian_log_unscaled(params[0], params[1], x[0]);
    }

    inline REAL gaussian_logpdf(const uint32_t data_len, const uint32_t params_len, const REAL* params,
                                const uint32_t dim, const REAL* x) {
        return gaussian_log(params[0], params[1], x[0]);
    }

    inline REAL gaussian_loglik(const uint32_t data_len, const REAL* data,
                                const uint32_t dim, const REAL* mu_sigma) {
        const REAL mu = mu_sigma[0];
        const REAL sigma = mu_sigma[1];
        if (0.0f < sigma) {
            REAL res = gaussian_log_unscaled(mu, sigma, data[0])
                + data_len * gaussian_log_scale(sigma);
            for (uint32_t i = 1; i < data_len; i++){
                res += gaussian_log_unscaled(mu, sigma, data[i]);
            }
            return res;
        }
        return nanf("NaN");
    }
}
