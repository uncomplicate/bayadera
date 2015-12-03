#ifndef R123_0x1p_23f
#define R123_0x1p_23f 1.1920928955078125E-7f
#endif

// Sampling from the uniform distribution
inline float u01fpt_oo_32_24(uint i) {
    return (0.5f + (i >> 9)) * R123_0x1p_23f;
}

inline float4 u01fpt_oo_4x32_24(uint4 i) {
    return (float4)((0.5f + (i.x >> 9)) * R123_0x1p_23f,
                    (0.5f + (i.y >> 9)) * R123_0x1p_23f,
                    (0.5f + (i.z >> 9)) * R123_0x1p_23f,
                    (0.5f + (i.w >> 9)) * R123_0x1p_23f);
}


inline float uniform_pdf(float lower, float upper, float x) {
    bool in_range = (lower <= x <= upper);
    return in_range? (1 / (upper - lower)) : 0.0f;
}

inline float uniform_logpdf(float lower, float upper, float x) {
    return native_log(uniform_pdf(lower, upper, x));
}
