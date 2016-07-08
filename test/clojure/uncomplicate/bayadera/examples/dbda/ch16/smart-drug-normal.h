inline REAL smart_drug_logpdf(__constant const REAL* params, REAL* x) {

    return gaussian_log(params[0], params[1], x[0])
        + uniform_log(params[2], params[3], x[1]);

}
