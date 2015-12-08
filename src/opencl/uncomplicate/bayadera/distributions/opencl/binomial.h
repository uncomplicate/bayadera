inline float lbinco(float n, float k) {
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

inline float binomial_logpdf(float n, float p, float k) {
    return (k * native_log(p))
        + ((n - k) * native_log(1 - p))
        + lbinco(n, k);
}

inline float binomial_loglik(float n, float k, float p) {
    return (k * native_log(p))
        + ((n - k) * native_log(1 - p));
}
