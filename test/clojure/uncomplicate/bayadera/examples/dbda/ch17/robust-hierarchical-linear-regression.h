REAL rhlr_loglik(const REAL* params, REAL* x) {

    const REAL nu = x[0];
    const REAL sigma = x[1];
    const bool valid = (0.0f < nu) && (0.0f < sigma);

    if (valid) {
        const REAL scale = student_t_log_scale(nu, sigma);
        REAL res = 0.0;
        uint idx = 1;
        for (uint i = 2; i < DIM; i+=2) {
            const REAL b0 = x[i];
            const REAL b1 = x[i+1];
            const uint next = idx + (uint)params[idx];
            while (idx < next) {
                res += student_t_log_unscaled(nu, b0 + b1 * params[idx+1], sigma, params[idx+2])
                    + scale;
                idx += 2;
            }
            idx++;
        }
        return res;
    }

    return NAN;
}

REAL rhlr_mcmc_logpdf(const REAL* params, REAL* x) {
    const bool valid = (1.0f < x[0]);

    if (valid) {
        REAL logp = exponential_log_unscaled(params[0], x[0] - 1)
            + uniform_log(params[1], params[2], x[1]);
        for (uint i = 0; i < DIM-2; i++) {
            logp += gaussian_log_unscaled(params[2*i+3], params[2*i+4], x[i+2]);
        }
        return logp;
    }
    return NAN;

}

REAL rhlr_logpdf(const REAL* params, REAL* x) {
    const bool valid = (1.0f < x[0]);

    if (valid) {
        REAL logp = exponential_log(params[0], x[0] - 1)
            + uniform_log(params[1], params[2], x[1]);
        for (uint i = 0; i < DIM-2; i++) {
            logp += gaussian_log(params[2*i+3], params[2*i+4], x[i+2]);
        }
        return logp;
    }
    return NAN;

}
