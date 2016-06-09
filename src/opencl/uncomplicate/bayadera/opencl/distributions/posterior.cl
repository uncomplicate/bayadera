inline REAL %s(__constant const REAL* params, REAL* x) {
    return %s(params, x) + %s(&params[%d], x);
}
