#ifndef M_2PI_F
#define M_2PI_F 6.2831855f
#endif

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
