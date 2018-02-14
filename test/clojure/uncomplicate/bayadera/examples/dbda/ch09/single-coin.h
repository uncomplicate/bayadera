inline REAL single_coin_logpdf(const REAL* params, REAL* x) {
    const REAL a = params[0];
    const REAL b = params[1];
    const REAL k = params[2];
    const REAL theta = x[0];
    const REAL omega = x[1];
    const REAL ak = omega * (k - 2) + 1;
    const REAL bk = (1 - omega) * (k - 2) + 1;
    const bool valid = (0.0f < omega) && (omega < 1.0f) && (0.0f < theta) && (theta < 1.0f);
    return valid ?
        beta_log(ak, bk, theta) - lbeta(ak, bk)
        + beta_log(a, b, omega) - lbeta(a, b)
        : NAN;

}
