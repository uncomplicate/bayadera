inline float lbeta (float a, float b) {
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}

inline float beta (float a, float b) {
    return native_exp(lbeta(a, b));
}

inline float beta_log(float a, float b, float x) {
    return ((a - 1.0f) * native_log(x)) + ((b - 1.0f) * native_log(1 - x));
}
