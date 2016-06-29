inline REAL gamma_log(const REAL alpha, const REAL beta, const REAL x) {
    return alpha * native_log(beta) + (alpha - 1) * native_log(x) - beta * x
        - lgamma(alpha);
}

// ============= With params ========================================

inline REAL gamma_logpdf(__constant const REAL* params, REAL* x) {
    const bool valid = (0.0f <= x[0]);
    return valid ? gamma_log(params[0], params[1], x[0]) : NAN;
}
