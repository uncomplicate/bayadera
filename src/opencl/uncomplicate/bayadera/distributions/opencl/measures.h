#ifndef M_SQRT_2PI_F
#define M_SQRT_2PI_F 0.9189385332046727f
#endif

// =============== Gaussian distribution ========================
inline float gaussian_logpdf(__constant float *params, float x) {
    float mu = params[0];
    float sigma = params[1];
    return (x - mu) * (x - mu) / (-2.0f * sigma * sigma)
        - native_log(sigma) - M_SQRT_2PI_F;
}

inline float gaussian_pdf(__constant float *params, float x) {
    float mu = params[0];
    float sigma = params[1];
    return native_exp((x - mu) * (x - mu) / (-2.0f * sigma * sigma))
        / (sigma * M_SQRT_2PI_F);
}

// =============== Uniform distribution ========================
inline float uniform_pdf(__constant float *params, float x) {
    float lower = params[0];
    float upper = params[1];
    bool in_range = (lower <= x <= upper);
    return in_range? (1 / (upper - lower)) : 0.0f;
}

inline float uniform_logpdf(__constant float *params, float x) {
    return uniform_pdf(params, x);
}
