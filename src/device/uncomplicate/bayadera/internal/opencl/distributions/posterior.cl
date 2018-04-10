inline REAL %s(const uint params_len, const REAL* params, const uint dim, const REAL* x) {
    const uint lik_params_len = %d;
    return %s(lik_params_len, params, params_len - lik_params_len, x) +
        %s(params_len - lik_params_len, &params[lik_params_len], dim, x);
}
