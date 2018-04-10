extern "C" {

    #include <stdint.h>
    
    inline REAL %s(const uint32_t data_len, const uint32_t hyperparams_len, const REAL* params,
            const uint32_t dim, const REAL* x) {
        %s;
    }
}
