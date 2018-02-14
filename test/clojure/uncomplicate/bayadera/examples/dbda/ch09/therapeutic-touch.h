REAL touch_logpdf(const REAL* params, REAL* x) {
    const REAL a = params[0];
    const REAL b = params[1];
    const REAL omega = x[DIM - 2];
    const REAL kappam2 = x[DIM - 1];
    const REAL ak = omega * kappam2 + 1.0f;
    const REAL bk = (1 - omega) * kappam2 + 1.0f;

    bool valid = (0.0f <= omega) & (omega <= 1.0f) && (0.0f <= kappam2);
    REAL logp = beta_log(a, b, omega) + gamma_log(params[2], params[3], kappam2)
        - (DIM - 2) * lbeta(ak, bk);

    for (uint i = 0; i < (DIM - 2); i++) {
        const REAL theta = x[i];
        valid = valid && (0.0f <= theta) && (theta <= 1.0f);
        logp += beta_log_unscaled(ak, bk, theta);
    }

    return valid ? logp : NAN;

}
