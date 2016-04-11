inline float %s(__constant const float* params, float* x) {
    return %s(params, x) + %s(&params[%d], x);
}
