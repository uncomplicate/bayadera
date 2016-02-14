inline float %s(__constant float* params, float x) {
    return %s(params, x) + %s(&params[%d], x);
}
