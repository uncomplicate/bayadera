inline float uniform_pdf(__constant float *params, float x) {
    float lower = params[0];
    float upper = params[1];
    bool in_range = (lower <= x <= upper);
    return in_range? (1 / (upper - lower)) : 0.0f;
}

inline float uniform_logpdf(__constant float *params, float x) {
    return uniform_pdf(params, x);
}
