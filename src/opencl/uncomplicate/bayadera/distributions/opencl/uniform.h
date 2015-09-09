inline float uniform_logpdf(__constant float *params, float x) {
    float low = params[0];
    float high = params[1];
    bool in_range (low <= x <= high);
    return in_range? 1 / high - low : 0.0;
}
