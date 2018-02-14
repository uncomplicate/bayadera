inline REAL multiple_coins_loglik(const REAL* params, REAL* p) {
    return binomial_log(params[0], p[0], params[1])
        + binomial_log(params[2], p[1], params[3]);
}
