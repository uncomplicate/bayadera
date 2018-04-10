extern "C" {

    #include <stdint.h>
    
    inline REAL %s(const uint32_t params_len, const REAL* params, const uint32_t dim, const REAL* x) {
        const uint32_t lik_params_len = %d;
        return %s(lik_params_len, params, dim, x) +
            %s(params_len - lik_params_len, &params[lik_params_len], dim, x);
    }

}
