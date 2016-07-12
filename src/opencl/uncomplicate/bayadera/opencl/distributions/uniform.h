inline REAL uniform_pdf(REAL lower, REAL upper, REAL x) {
    bool in_range = (lower <= x) && (x <= upper);
    return in_range? (1 / (upper - lower)) : 0.0f;
}

inline REAL uniform_log(REAL lower, REAL upper, REAL x) {
    return native_log(uniform_pdf(lower, upper, x));
}

// ============= With params ========================================

REAL uniform_logpdf(__constant REAL* params, REAL* x) {
    return uniform_log(params[0], params[1], x[0]);
}
