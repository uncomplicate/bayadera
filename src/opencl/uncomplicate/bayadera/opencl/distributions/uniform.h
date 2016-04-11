inline float uniform_pdf(float lower, float upper, float x) {
    bool in_range = (lower <= x) && (x <= upper);
    return in_range? (1 / (upper - lower)) : 0.0f;
}

inline float uniform_log(float lower, float upper, float x) {
    return native_log(uniform_pdf(lower, upper, x));
}

// ============= With params ========================================

inline float uniform_logpdf(__constant float* params, float* x) {
    return uniform_log(params[0], params[1], x[0]);
}
