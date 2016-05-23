inline float ch09_1mint_1coin_logpdf(__constant const float* params, float* x) {
    const float a = params[0];
    const float b = params[1];
    const float k = params[2];
    const float omega = x[0];
    const float theta = x[1];
    const float ak = omega * (k - 2) + 1;
    const float bk = (1 - omega) * (k - 2) + 1;
    return beta_log(ak, bk, theta) - lbeta(ak, bk)
        + beta_log(a, b, omega) - lbeta(a, b);
}
