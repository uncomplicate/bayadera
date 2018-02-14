inline REAL multiple_coins_logpdf(const REAL* params, REAL* x) {
    const REAL a = params[0];
    const REAL b = params[1];
    const REAL k = params[2];
    const REAL theta0 = x[0];
    const REAL theta1 = x[1];
    const REAL omega = x[2];
    const REAL ak = omega * (k - 2) + 1;
    const REAL bk = (1 - omega) * (k - 2) + 1;
    const bool valid = (0.0f < omega) && (omega < 1.0f)
        && (0.0f < theta0) && (theta0 < 1.0f)
        && (0.0f < theta1) && (theta1 < 1.0f);
    return valid ?
        beta_log(a, b, omega) - lbeta(a, b)
        + beta_log(ak, bk, theta0) - lbeta(ak, bk)
        + beta_log(ak, bk, theta1) - lbeta(ak, bk)
        : NAN;

}
