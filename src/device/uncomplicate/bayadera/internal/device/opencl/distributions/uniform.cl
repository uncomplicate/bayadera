inline REAL uniform_pdf(const REAL lower, const REAL upper, const REAL x) {
    bool in_range = (lower <= x) && (x <= upper);
    return in_range? (1 / (upper - lower)) : 0.0f;
}

inline REAL uniform_log(const REAL lower, const REAL upper, const REAL x) {
    return native_log(uniform_pdf(lower, upper, x));
}

// ============= With params ========================================

inline REAL uniform_logpdf(const uint data_len, const uint params_len, const REAL* params,
                           const uint dim, const REAL* x) {
    return uniform_log(params[0], params[1], x[0]);
}
