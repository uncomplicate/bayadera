#ifndef DIST_NAME
#define DIST_NAME unknown
#endif

#ifndef DIST_PDF
#define DIST_PDF unknown
#endif


#ifndef DIST_LOGPDF
#define DIST_LOGPDF unknown
#endif

inline float logpdf(__constant float* params, float x){
    return DIST_LOGPDF(params, x);
}

inline float pdf(__constant float* params, float x){
    return DIST_PDF(params, x);
}
