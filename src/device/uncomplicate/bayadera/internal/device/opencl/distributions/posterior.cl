inline REAL %s(const uint data_len, const uint hyperparams_len, const REAL* params,
               const uint dim, const REAL* x) {

    return %s(data_len, params, dim, x) +
        %s(data_len, hyperparams_len, &params[data_len], dim, x);
}
