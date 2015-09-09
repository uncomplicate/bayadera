#ifndef M_SQRT_2PI_F
#define M_SQRT_2PI_F 0.9189385332046727f
#endif

inline float gaussian_logpdf(__constant float *params, float x) {
    float mu = params[0];
    float sigma = params[1];
    return (x - mu) * (x - mu) / (-2.0f * sigma * sigma)
        - native_log(sigma) - M_SQRT_2PI_F;
}
