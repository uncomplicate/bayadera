REAL touch_loglik(const REAL* params, REAL* p) {
    REAL loglik = 0.0f;
    for (uint i = 0; i < (DIM - 2); i++) {
        const REAL theta = p[i];
        loglik += binomial_log_unscaled(params[2*i], theta, params[2*i+1]);
    }
    return loglik;

}
