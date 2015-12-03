#ifndef M_SQRT_2PI_F
#define M_SQRT_2PI_F 0.9189385332046727f
#endif

#ifndef M_2PI_F
#define M_2PI_F 6.2831855f
#endif

//Sampling from the Gaussian distribution
inline float4 box_muller(float4 uniform) {
    return (float4)(native_sin(M_2PI_F * uniform.x)
                    * sqrt(-2.0f * native_log(uniform.y)),
                    native_cos(M_2PI_F * uniform.x)
                    * sqrt(-2.0f * native_log(uniform.y)),
                    native_sin(M_2PI_F * uniform.z)
                    * sqrt(-2.0f * native_log(uniform.w)),
                    native_cos(M_2PI_F * uniform.z)
                    * sqrt(-2.0f * native_log(uniform.w)));
}

inline float gaussian_logpdf(float mu, float sigma, float x) {
    return (x - mu) * (x - mu) / (-2.0f * sigma * sigma)
        - native_log(sigma) - M_SQRT_2PI_F;
}

inline float gaussian_pdf(float mu, float sigma, float x) {
    return native_exp((x - mu) * (x - mu) / (-2.0f * sigma * sigma))
        / (sigma * M_SQRT_2PI_F);
}
