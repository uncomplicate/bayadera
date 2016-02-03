inline float lbinco(const float n, const float k) {
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

inline float binomial_log(const float n, const float p, const float k) {
    return (k * native_log(p))
        + ((n - k) * native_log(1 - p));
}

inline float binomial_loglik(__constant const float* params, const float p) {
    return binomial_log(params[0], p, params[1]);
}

inline float binomial_logpdf(__constant const float* params, const float x) {
    return binomial_log(params[0], params[1], x);
}
