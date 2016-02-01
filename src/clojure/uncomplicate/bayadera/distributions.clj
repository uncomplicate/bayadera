(ns uncomplicate.bayadera.distributions
  (:require [uncomplicate.neanderthal
             [math :refer [log exp pow floor sqrt]]
             [core :refer [scal! copy! copy freduce fmap!]]
             [real :refer [asum]]]
            [uncomplicate.bayadera.math
             :refer [log-factorial factorial log-binco log-multico
                     log-beta log-gamma gamma erf incomplete-gamma-l
                     regularized-beta regularized-gamma-q]]))

(defn probability? [^double p]
  (<= 0 p 1))

;; ==================== Bernoulli trials ====================
(defn bernoulli-log-trials
  ^double [^long n ^double p ^long k]
  (+ (* k (log p)) (* (- n k) (log (- 1 p)))))

;; ==================== Binomial distribution ====================

(defn binomial-check-args
  ([^double p]
   (probability? p))
  ([^long n ^long k]
   (<= 0 k n))
  ([^long n ^double p ^long k]
   (and (probability? p) (binomial-check-args n k))))

(defn binomial-log-pmf
  ^double [^long n ^double p ^long k]
  (+ (log-binco n k) (bernoulli-log-trials n p k)))

(defn binomial-pmf
  ^double [^long n ^double p ^long k]
  (exp (binomial-log-pmf n p k)))

(defn binomial-mean
  ^double [^long n ^double p]
  (* n p))

(defn binomial-variance
  ^double [^long n ^double p]
  (* n p (- 1.0 p)))

;; TODO Use normal approximation
(defn binomial-cdf
  ^double [^long n ^double p ^long k]
  (loop [i 0 res 0.0]
    (if (< k i)
      res
      (recur (inc i) (+ res (binomial-pmf n p i))))))

;; ==================== Geometric Distribution ====================

(defn geometric-check-args
  ([^double p]
   (probability? p))
  ([^double p ^long k]
   (and (probability? p) (< 0 k))))

(defn geometric-normalizer
  ^double [^long n ^double p]
  (/ 1.0 (- 1.0 (pow (- 1.0 p) n))))

(defn geometric-log-pmf
  ^double [^double p ^long k]
  (bernoulli-log-trials k p 1.0))

(defn geometric-pmf
  ^double [^double p ^long k]
  (exp (geometric-log-pmf p k)))

(defn geometric-mean
  ^double [^double p]
  (/ 1.0 p))

(defn geometric-variance
  ^double [^double p]
  (/ (- 1.0 p) (* p p)))

(defn geometric-cdf
  ^double [^double p ^long k]
  (- 1.0 (pow (- 1.0 p) k)))

;; ==================== Negative Binomial (Pascal) Distribution ================

(defn pascal-check-args
  ([^double p]
   (probability? p))
  ([^long k ^long n]
   (<= 0 k n))
  ([^long k ^double p ^long n]
   (and (probability? p) (pascal-check-args n k))))

(defn pascal-log-pmf
  ^double [^long k ^double p ^long n]
  (+ (log-binco (dec n) (dec k)) (bernoulli-log-trials n p k)))

(defn pascal-pmf
  ^double [^long k ^double p ^long n]
  (exp (pascal-log-pmf k p n)))

(defn pascal-mean
  ^double [^long k ^double p]
  (/ k p))

(defn pascal-variance
  ^double [^long k ^double p]
  (/ (* k (- 1 p)) (* p p)))

(defn pascal-cdf
  ^double [^long k ^double p ^long n]
  (if (< k n)
    (- 1 (regularized-beta p (inc k) (- n k)))
    (if (= n k)
      (pow p k)
      0.0)))

;; ==================== Hyper-geometric Distribution ================

(defn hypergeometric-check-args
  ([^long k ^long n]
   (<= 0 k n))
  ([^long N ^long K ^long n ^long k]
   (and (<= K N) (<= 0 k (min n (- N K)))
        (<= n N) (<= k K) (<= (- n k) (- N K)))))

(defn hypergeometric-log-pmf
  ^double [^long N ^long K ^long n ^long k]
  (- (+ (log-binco K k) (log-binco (- N K) (- n k)))
     (log-binco N n)))

(defn hypergeometric-pmf
  ^double [^long N ^long K ^long n ^long k]
  (exp (hypergeometric-log-pmf N K n k)))

(defn hypergeometric-mean
  ^double [^long N ^long K ^long n]
  (/ (* n K) N))

(defn hypergeometric-variance
  ^double [^long N ^long K ^long n]
  (/ (* n K (- N K) (- N n)) (* N N (dec N))))

(defn hypergeometric-cdf
  ^double [^long N ^long K ^long n ^long k]
  (loop [j (max 0 (- n (- N K))) res 0.0]
    (if (< k j)
      res
      (recur (inc j) (+ res (hypergeometric-pmf N K n j))))))

;; ==================== Poisson Distribution ================

(defn poisson-check-args
  [^double lambda ^long k]
  (and (< 0 lambda) (<= 0 k)))

(defn poisson-log-pmf
  ^double [^double lambda ^long k]
  (- (* k (log lambda)) lambda (log-factorial k)))

(defn poisson-pmf
  ^double [^double lambda ^long k]
  (exp (poisson-log-pmf lambda k)))

(defn poisson-mean
  ^double [^double lambda]
  lambda)

(defn poisson-variance
  ^double [^double lambda]
  lambda)

(defn poisson-cdf
  ^double [^double lambda ^long k]
  (regularized-gamma-q (inc k) lambda))

;; ==================== Exponential Distribution ================

(defn exponential-log-pdf
  ^double [^double lambda ^double x]
  (- (log lambda) (* lambda x)))

(defn exponential-pdf
  ^double [^double lambda ^double x]
  (* lambda (exp (- (* lambda x)))))

(defn exponential-mean
  ^double [^double lambda]
  (/ 1.0 lambda))

(defn exponential-variance
  ^double [^double lambda]
  (/ 1.0 (* lambda lambda)))

(defn exponential-cdf
  ^double [^double lambda ^double x]
  (- 1 (exp (- (* lambda x)))))

;; ==================== Erlang Distribution ================

(defn erlang-log-pdf
  ^double [^double lambda ^long k ^double x]
  (- (+ (* k (log lambda)) (* (dec k) (log x)))
     (* lambda x) (log-factorial (dec k))))

(defn erlang-pdf
  ^double [^double lambda ^long k ^double x]
  (exp (erlang-log-pdf lambda k x)))

(defn erlang-mean
  ^double [^double lambda ^long k]
  (/ k lambda))

(defn erlang-variance
  ^double [^double lambda ^long k]
  (/ k (* lambda lambda)))

(defn erlang-cdf
  ^double [^double lambda ^long k ^double x]
  (/ (incomplete-gamma-l k (* lambda x))
     (factorial (dec k))))

;; ==================== Uniform Distribution ================

(defn uniform-pdf
  ^double [^double a ^double b ^double x]
  (if (<= a x b)
    (/ 1.0 (- b a))
    0.0))

(defn uniform-log-pdf
  ^double [^double a ^double b ^double x]
  (log (uniform-pdf a b x)))

(defn uniform-mean
  ^double [^double a ^double b]
  (* (+ a b) 0.5))

(defn uniform-variance
  ^double [^double a ^double b]
  (/ (* (- b a) (- b a)) 12.0))

(defn uniform-cdf
  ^double [^double a ^double b ^double x]
  (cond
    (<= a x b) (/ (- x a) (- b a))
    (< x a) 0
    (> b x) 1
    :default 0))

;; ==================== Gaussian (Normal) Distribution ================

(let [sqrt-2pi (sqrt (* 2.0 Math/PI))
      log-sqrt-2pi (log sqrt-2pi)
      sqrt2 (sqrt 2.0)]

  (defn gaussian-log-pdf
    ^double [^double mu ^double sigma ^double x]
    (- (/ (* (- x mu) (- x mu)) (* -2.0 sigma sigma))
       (log sigma) log-sqrt-2pi))

  (defn gaussian-pdf
    ^double [^double mu ^double sigma ^double x]
    (/ (exp (/ (* (- x mu) (- x mu)) (* -2.0 sigma sigma)))
       (* sigma sqrt-2pi)))

  (defn gaussian-mean
    ^double [^double mu]
    mu)

  (defn gaussian-variance
    ^double [^double sigma]
    (* sigma sigma))

  (defn gaussian-cdf
    ^double [^double mu ^double sigma ^double x]
    (* 0.5 (+ 1.0 (erf (/ (- x mu) sigma sqrt2))))))

;; ==================== Beta Distribution ================

(defn beta-check-args
  ([^double a ^double b]
   (and (< 0.0 a) (< 0.0 b)))
  ([^double x]
   (< 0.0 x 1.0))
  ([^double a ^double b ^double x]
   (and (beta-check-args a b) (beta-check-args x))))

(defn beta-log-unscaled
  ^double [^double a ^double b ^double x]
  (+ (* (dec a) (log x)) (* (dec b) (log (- 1.0 x)))))

(defn beta-log-scale-factor
  ^double [^double a ^double b]
  (- (log-beta a b)))

(defn beta-log-pdf
  ^double [^double a ^double b ^double x]
  (- (beta-log-unscaled a b x) (log-beta a b)))

(defn beta-pdf
  ^double [^double a ^double b ^double x]
  (exp (beta-log-pdf a b x)))

(defn beta-mean
  ^double [^double a ^double b]
  (/ a (+ a b)))

(defn beta-variance
  ^double [^double a ^double b]
  (/ (* a b) (* (+ a b) (+ a b) (inc (+ a b)))))

(defn beta-cdf
  ^double [^double a ^double b ^double x]
  (regularized-beta x a b))

;; ==================== Gamma Distribution ================

(defn gamma-log-unscaled
  ^double [^double theta ^double k ^double x]
  (- (* (dec k) (log x)) (/ x theta)))

(defn gamma-log-scale-factor
  ^double [^double theta ^double k]
  (- (+ (log-gamma k) (* k (log theta)))))

(defn gamma-log-pdf
  ^double [^double theta ^double k ^double x]
  (+ (gamma-log-unscaled theta k x) (gamma-log-scale-factor theta k)))

(defn gamma-pdf
  ^double [^double theta ^double k ^double x]
  (exp (gamma-log-pdf theta k x)))

(defn gamma-mean
  ^double [^double theta ^double k]
  (* k theta))

(defn gamma-variance
  ^double [^double theta ^double k]
  (* k theta theta))

(defn gamma-cdf
  ^double [^double theta ^double k ^double x]
  (/ (incomplete-gamma-l k (/ x theta)) (gamma k)))

;; ==================== Multinomial Distribution ================

(defn multinomial-log-unscaled
  ^double [ps ks]
  (freduce (fn ^double [^double acc ^double p ^double k]
             (+ acc (* k (log p))))
           0.0
           ps ks))

(defn multinomial-log-pmf
  ^double [ps ks]
  (freduce (fn ^double [^double acc ^double p ^double k]
             (+ acc (- (* k (log p)) (log-factorial k))))
           (log-factorial (asum ks))
           ps ks))

(defn multinomial-pmf
  ^double [ps ks]
  (exp (multinomial-log-pmf ps ks)))

(defn multinomial-mean
  (^double [ps ks result]
   (scal! (asum ks) (copy! ps result)))
  (^double [ps ks]
   (scal! (asum ks) (copy ps))))

(defn multinomial-variance
  ^double [ps ks]
  (scal! (fmap! (fn ^double [^double p]
                  (* p (- 1.0 p)))
                (copy ps))))

(defn multinomial-cdf
  ^double [ps ks]
  (throw (java.lang.UnsupportedOperationException. "TODO")))
