REAL mlr_loglik(__constant const REAL* params, REAL* x) {

    const REAL nu = x[0];
    const REAL sigma = x[1];
    const REAL b0 = x[2];
    const REAL b1 = x[3];
    const REAL b2 = x[4];

    const uint n = (uint)params[0];

    const bool valid = (0.0f < nu) && (0.0f < sigma);

    if (valid) {
        const REAL scale = t_log_scale(nu, sigma);
        REAL res = 0.0;
        for (uint i = 0; i < n; i = i+3) {
            res += t_log_unscaled(nu, b0 + b1 * params[i+1] + b2 * params[i+2], sigma, params[i+3])
                + scale;
        }
        return res;
    }
    return NAN;

}

REAL mlr_mcmc_logpdf(__constant const REAL* params, REAL* x) {
    const bool valid = (1.0f < x[0]);
    if (valid) {
        REAL logp = exponential_log_unscaled(params[0], x[0] - 1)
            + uniform_log(params[1], params[2], x[3]);
        for (uint i = 0; i < DIM-2; i++) {
            logp += gaussian_log_unscaled(params[2*i+3], params[2*i+4], x[i+2]);
        }
        return logp;
    }
    return NAN;
}

REAL mlr_logpdf(__constant const REAL* params, REAL* x) {
    bool valid = (1.0f < x[0]);
    if (valid) {
        REAL logp = exponential_log(params[0], x[0] - 1)
            + uniform_log(params[1], params[2], x[3]);
        for (uint i = 0; i < DIM-2; i++) {
            logp += gaussian_log(params[2*i+3], params[2*i+4], x[i+2]);
        }
        return logp;
    }
    return NAN;
}
