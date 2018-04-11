extern "C" {

#ifndef M_LOG_SQRT_PI_F
#define M_LOG_SQRT_PI_F 0.5723649429247f
#endif

#include <stdint.h>
    
    inline REAL student_t_log_unscaled(const REAL nu, const REAL mu, const REAL sigma, const REAL x) {
        const REAL x_mu_sigma = (x - mu) / sigma;
        return - (0.5f * (nu + 1.0f) * log(1.0f + x_mu_sigma * x_mu_sigma / nu));
    }

    inline REAL student_t_log_scale(const REAL nu, const REAL sigma) {
        return lgamma(0.5f * (nu + 1.0f)) - lgamma(0.5f * nu)
            - M_LOG_SQRT_PI_F - 0.5f * log(nu) - log(sigma);
    }

    inline REAL student_t_log(const REAL nu, const REAL mu, const REAL sigma, const REAL x) {
        return student_t_log_unscaled(nu, mu, sigma, x) + student_t_log_scale(nu, sigma);
    }

// ============= With params ========================================

    inline REAL student_t_mcmc_logpdf(const uint32_t data_len, const uint32_t params_len, const REAL* params,
                                      const uint32_t dim, const REAL* x) {
        return student_t_log_unscaled(params[0], params[1], params[2], x[0]);
    }

    inline REAL student_t_logpdf(const uint32_t data_len, const uint32_t params_len, const REAL* params,
                                 const uint32_t dim, const REAL* x) {
        return student_t_log_unscaled(params[0], params[1], params[2], x[0]) + params[3];
    }

    inline REAL student_t_loglik(const uint32_t data_len, const REAL* data,
                                 const uint32_t dim, const REAL* nu_mu_sigma) {
        const REAL nu = nu_mu_sigma[0];
        const REAL mu = nu_mu_sigma[1];
        const REAL sigma = nu_mu_sigma[2];
        const bool valid = (0.0f < nu) && (0.0f < sigma);
        if (valid) {
            const REAL scale = student_t_log_scale(nu, sigma);
            REAL res = 0.0;
            for (uint32_t i = 0; i < data_len; i++){
                res += (student_t_log_unscaled(nu, mu, sigma, data[i]) + scale);
            }
            return res;
        }
        return nanf("NaN");
    }
}
