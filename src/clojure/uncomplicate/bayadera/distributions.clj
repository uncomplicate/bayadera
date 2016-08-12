(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.distributions
  (:require [uncomplicate.commons.core :refer [double-fn]]
            [uncomplicate.fluokitten.core :refer [foldmap fmap!]]
            [uncomplicate.neanderthal
             [math :refer [log exp pow floor sqrt ceil f=]]
             [core :refer [scal! copy! copy dim raw]]
             [real :refer [asum entry]]]
            [uncomplicate.bayadera.math
             :refer [log-factorial factorial log-binco log-multico
                     log-beta log-gamma gamma erf incomplete-gamma-l
                     regularized-beta regularized-gamma-q]])
  (:import [org.apache.commons.math3.distribution TDistribution]))

(defn probability? [^double p]
  (<= 0.0 p 1.0))

;; ==================== Bernoulli trials ====================

(defn bernoulli-log-trials
  ^double [^long n ^double p ^long k]
  (+ (* k (log p)) (* (- n k) (log (- 1.0 p)))))

;; ==================== Binomial distribution ====================

(defn binomial-check
  ([^long n ^long p]
   (<= 0 n) (probability? p))
  ([^long n ^double p ^long k]
   (and (binomial-check n p) (<= 0 k n))))

(defn binomial-params
  [^long n ^double p]
  (when (and (<= 0 n) (probability? p))
    [n p]))

(defn binomial-log-pmf
  ^double [^long n ^double p ^long k]
  (+ (log-binco n k) (bernoulli-log-trials n p k)))

(defn binomial-pmf
  ^double [^long n ^double p ^long k]
  (exp (binomial-log-pmf n p k)))

(defn binomial-mean
  ^double [^long n ^double p]
  (* n p))

(defn binomial-mode
  ^long [^long n ^double p]
  (long (* (inc n) p)))

(defn binomial-median
  ^long [^long n ^double p]
  (long (* n p)))

(defn binomial-variance
  ^double [^long n ^double p]
  (* n p (- 1.0 p)))

(defn binomial-cdf
  ^double [^long n ^double p ^long k]
  (if (<= 0 k n)
    (- 1.0 (regularized-beta p (inc k) (- n k)))
    0.0))

;; ============ Binomial (Bernoulli) Likelihood ===================

(defn binomial-lik-params [^long n ^long k]
  (when (<= 0 k n)
    [n k]))

(defn binomial-log-lik
  ^double [^long n ^long k ^double p]
  (bernoulli-log-trials n p k))

(defn binomial-lik
  ^double [^long n ^long k ^double p]
  (exp (bernoulli-log-trials n p k)))

;; ==================== Geometric Distribution ====================

(defn geometric-check
  [^double p ^long k]
  (and (< 0 k) (probability? p)))

(defn geometric-params
  [^double p]
  (when (probability? p)
    [p]))

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

(defn geometric-mode
  ^long [^double p]
  1)

(defn geometric-median
  ^long [^double p]
  (long (ceil (/ -1.0 (log (- 1.0 p))))))

(defn geometric-variance
  ^double [^double p]
  (/ (- 1.0 p) (* p p)))

(defn geometric-cdf
  ^double [^double p ^long k]
  (- 1.0 (pow (- 1.0 p) k)))

;; ==================== Negative Binomial (Pascal) Distribution ================

(defn pascal-check
  ([^long k ^double p]
   (and (< 0 k) (probability? p)))
  ([^long k ^double p ^long n]
   (and (< 0 k) (<= 0 k n) (probability? p))))

(defn pascal-params
  [^long k ^double p]
  (when (pascal-check k p)
    [k p]))

(defn pascal-log-pmf
  ^double [^long k ^double p ^long n]
  (+ (log-binco (dec n) (dec k)) (bernoulli-log-trials n p k)))

(defn pascal-pmf
  ^double [^long k ^double p ^long n]
  (exp (pascal-log-pmf k p n)))

(defn pascal-mean
  ^double [^long k ^double p]
  (/ k p))

(defn pascal-mode
  ^long [^long k ^double p]
  (long (floor (inc (/ (dec k) p)))))

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
;;TODO

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
;; TODO

(defn poisson-check
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

(defn exponential-check
  ([^double lambda]
   (< 0.0 lambda))
  ([^double lambda ^double x]
   (and (< 0.0 lambda) (< 0.0 x))))

(defn exponential-params
  [^double lambda]
  (when (< 0.0 lambda)
    [lambda (log lambda)]))

(defn exponential-log-unscaled
  ^double [^double lambda ^double x]
  (- (* lambda x)))

(defn exponential-log-pdf
  ^double [^double lambda ^double x]
  (- (log lambda) (exponential-log-unscaled lambda x)))

(defn exponential-pdf
  ^double [^double lambda ^double x]
  (* lambda (exp (- (* lambda x)))))

(defn exponential-mean
  ^double [^double lambda]
  (/ 1.0 lambda))

(defn exponential-mode
  ^double [^double lambda]
  0.0)

(let [ln2 (log 2.0)]
  (defn exponential-median
    ^double [^double lambda]
    (/ ln2 lambda)))

(defn exponential-variance
  ^double [^double lambda]
  (/ 1.0 (* lambda lambda)))

(defn exponential-cdf
  ^double [^double lambda ^double x]
  (- 1.0 (exp (- (* lambda x)))))

;; ==================== Erlang Distribution ================

(defn erlang-check
  ([^double lambda ^long k]
   (and (< 0.0 lambda) (< 0 k)))
  ([^double lambda ^long k ^double x]
   (and (< 0.0 lambda) (< 0 k) (<= 0.0 x))))

(defn erlang-log-unscaled
  ^double [^double lambda ^long k ^double x]
  (- (* (dec k) (log x)) (* lambda x) ))

(defn erlang-log-scale
  ^double [^double lambda ^long k]
  (- (* k (log lambda)) (log-factorial (dec k))))

(defn erlang-params
  [^double lambda ^long k]
  (when (erlang-check lambda k)
    [lambda k (erlang-log-scale lambda k)]))

(defn erlang-log-pdf
  ^double [^double lambda ^long k ^double x]
  (+ (erlang-log-unscaled lambda k x) (erlang-log-scale lambda k)))

(defn erlang-pdf
  ^double [^double lambda ^long k ^double x]
  (exp (erlang-log-pdf lambda k x)))

(defn erlang-mean
  ^double [^double lambda ^long k]
  (/ k lambda))

(defn erlang-median
  ^double [^double lambda ^long k]
  (/ (/ k lambda) (- 1.0 (/ 1.0 (+ (* 3.0 k) 0.2)))))

(defn erlang-mode
  ^double [^double lambda ^long k]
  (if (<= 1.0 k)
    (/ (dec k) lambda)
    Double/NaN))

(defn erlang-variance
  ^double [^double lambda ^long k]
  (/ k (* lambda lambda)))

(defn erlang-cdf
  ^double [^double lambda ^long k ^double x]
  (/ (incomplete-gamma-l k (* lambda x))
     (factorial (dec k))))

;; ==================== Uniform Distribution ================

(defn uniform-check
  ([^double a ^double b]
   (< a b))
  ([^double a ^double b ^double x]
   (< a b)))

(defn uniform-params
  [^double a ^double b]
  (when (uniform-check a b)
    [a b]))

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

(def uniform-mode uniform-mean)

(def uniform-median uniform-mean)

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

  (defn gaussian-check
    ([^double mu ^double sigma]
     (< 0.0 sigma))
    ([^double mu ^double sigma ^double x]
     (< 0.0 sigma)))

  (defn gaussian-params
    [^double mu ^double sigma]
    (when (gaussian-check mu sigma)
      [mu sigma]))

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

  (def gaussian-mode gaussian-mean)

  (def gaussian-median gaussian-mean)

  (defn gaussian-variance
    ^double [^double sigma]
    (* sigma sigma))

  (defn gaussian-cdf
    ^double [^double mu ^double sigma ^double x]
    (* 0.5 (+ 1.0 (erf (/ (- x mu) sigma sqrt2))))))

;; ==================== Student's t distribution ================

(let [log-sqrt-pi (log (sqrt Math/PI))]

  (defn t-log-scale
    (^double [^double nu]
     (- (log-gamma (* 0.5 (inc nu)))
        (* 0.5 (log nu)) log-sqrt-pi (log-gamma (* 0.5 nu))))
    (^double [^double nu ^double sigma]
     (- (t-log-scale nu) (log sigma)))))

(defn t-check
  ([^double nu]
   (< 0.0 nu))
  ([^double nu ^double mu ^double sigma]
   (and (< 0.0 nu) (< 0.0 sigma)))
  ([^double nu ^double mu ^double sigma ^double x]
   (t-check nu mu sigma)))

(defn t-params
  ([^double nu ^double mu ^double sigma]
   (when (t-check nu mu sigma)
     [nu mu sigma (t-log-scale nu sigma)]))
  ([^double nu]
   (t-params nu 0.0 1.0)))

(defn t-log-unscaled
  (^double [^double nu ^double x]
   (- (* 0.5 (inc nu) (log (inc (/ (* x x) nu))))))
  (^double [^double nu ^double mu ^double sigma ^double x]
   (t-log-unscaled nu (/ (- x mu) sigma))))

(defn t-log-pdf
  (^double [^double nu ^double x]
   (+ (t-log-unscaled nu x) (t-log-scale nu)))
  (^double [^double nu ^double mu ^double sigma ^double x]
   (+ (t-log-unscaled nu mu sigma x) (t-log-scale nu sigma))))

(defn t-pdf
  (^double [^double nu ^double x]
   (exp (t-log-pdf nu x)))
  (^double [^double nu ^double mu ^double sigma ^double x]
   (exp (t-log-pdf nu mu sigma x))))

(defn t-mean
  (^double [^double nu]
   (t-mean nu 0.0))
  (^double [^double nu ^double mu]
   (if (< 1.0 nu) mu Double/NaN)))

(def t-mode t-mean)

(def t-median t-mean)

(defn t-variance
  (^double [^double nu]
   (t-variance nu 1.0))
  (^double [^double nu ^double sigma]
   (cond
     (< 2.0 nu) (* sigma sigma (/ nu (- nu 2.0)))
     (and (< 1.0 nu) (<= nu 2)) Double/POSITIVE_INFINITY
     :default Double/NaN)))

(defn t-cdf
  (^double [^double nu ^double x]
   (t-cdf nu 0.0 1.0 x))
  (^double [^double nu ^double mu ^double sigma ^double x]
   (if (= mu x)
     0.5
     (let [nusigma2 (* nu sigma sigma)
           x-mu (- x mu)]
       (if (< x mu)
         (* 0.5 (regularized-beta (/ nusigma2 (+ nusigma2 (* x-mu x-mu)))
                                  (* 0.5 nu) 0.5))
         (* 0.5 (+ 1.0 (regularized-beta (/ (* x-mu x-mu) (+ nusigma2 (* x-mu x-mu)))
                                         0.5 (* 0.5 nu)))))))))

;; ==================== Beta Distribution ================

(defn beta-check
  ([^double a ^double b]
   (and (< 0.0 a) (< 0.0 b)))
  ([^double a ^double b ^double x]
   (and (beta-check a b) (<= 0.0 x 1.0))))

(defn beta-log-unscaled
  ^double [^double a ^double b ^double x]
  (+ (* (dec a) (log x)) (* (dec b) (log (- 1.0 x)))))

(defn beta-log-scale
  ^double [^double a ^double b]
  (- (log-beta a b)))

(defn beta-params [^double a ^double b]
  (when (beta-check a b)
    [a b (beta-log-scale a b)]))

(defn beta-log-pdf
  ^double [^double a ^double b ^double x]
  (- (beta-log-unscaled a b x) (log-beta a b)))

(defn beta-pdf
  ^double [^double a ^double b ^double x]
  (exp (beta-log-pdf a b x)))

(defn beta-mean
  ^double [^double a ^double b]
  (/ a (+ a b)))

(defn beta-median
  ^double [^double a ^double b]
  (cond
    (= a b) 0.5
    (and (= 1.0 a) (< 0.0 b)) (- 1.0 (pow 2.0 (- (/ 1.0 b))))
    (and (= 1.0 b) (< 0.0 a)) (pow 2.0 (- (/ 1.0 a)))
    (and (= 3.0 a) (= 2.0 b)) 0.6142724318676105
    (and (= 2.0 a) (= 3.0 b)) 0.38572756813238945
    (and (< 1.0 a) (< 1.0 b)) (/ (- a (/ 1.0 3.0)) (- (+ a b) (/ 2.0 3.0)))
    :default Double/NaN))

(defn beta-mode
  ^double [^double a ^double b]
  (if (and (< 1.0 a) (< 1.0 b))
    (/ (dec a) (- (+ a b) 2.0))
    Double/NaN))

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

(defn gamma-log-scale
  ^double [^double theta ^double k]
  (- (+ (log-gamma k) (* k (log theta)))))

(defn gamma-check
  ([^double theta ^double k]
   (and (< 0.0 theta) (< 0.0 k)))
  ([^double theta ^double k ^double x]
   (and (< 0.0 theta) (< 0.0 k) (< 0.0 x))))

(defn gamma-params
  [^double theta ^double k]
  (when (gamma-check theta k)
    [theta k (gamma-log-scale theta k)]))

(defn gamma-log-pdf
  ^double [^double theta ^double k ^double x]
  (+ (gamma-log-unscaled theta k x) (gamma-log-scale theta k)))

(defn gamma-pdf
  ^double [^double theta ^double k ^double x]
  (exp (gamma-log-pdf theta k x)))

(defn gamma-mean
  ^double [^double theta ^double k]
  (* k theta))

(defn gamma-mode
  ^double [^double theta ^double k]
  (if (<= 1.0 k)
    (* (dec k) theta)))

(defn gamma-variance
  ^double [^double theta ^double k]
  (* k theta theta))

(defn gamma-cdf
  ^double [^double theta ^double k ^double x]
  (/ (incomplete-gamma-l k (/ x theta)) (gamma k)))

;; ==================== Multinomial Distribution ================

(def ^:private p+ (double-fn +) )

(defn multinomial-log-unscaled
  ^double [ps ks]
  (foldmap p+
           (fn ^double [^double p ^double k]
             (* k (log p)))
           ps ks))

(defn multinomial-log-pmf
  ^double [ps ks]
  (foldmap p+
           (log-factorial (asum ks))
           (fn ^double [^double p ^double k]
             (- (* k (log p)) (log-factorial k)))
           ps ks))

(defn multinomial-check
  ([ps]
   (let [n (dim ps)]
     (loop [i 0 sum 0.0]
       (if (and (< i n) (probability? (entry ps i)))
         (recur (inc i) (+ sum (entry ps i)))
         (and (= i n) (f= (float sum) (float 1.0) (sqrt n)))))))
  ([ps ks]
   (and (= (dim ps) (dim ks))
        (multinomial-check ps)
        (let [n (dim ks)]
          (loop [i 0]
            (if (and (< i n) (<= 0 (entry ks i)))
              (recur (inc i))
              (= i n)))))))

(defn multinomial-pmf
  ^double [ps ks]
  (exp (multinomial-log-pmf ps ks)))

(defn multinomial-mean
  ([ps ^long n result]
   (scal! n (copy! ps result)))
  ([ps ^long n]
   (scal! n (copy ps))))

(defn multinomial-mode
  ([ps ^long n result]
   (scal! (inc n) (copy! ps result)))
  ([ps ^long n]
   (scal! (inc n) (copy ps))))

(def multinomial-median multinomial-mean)

(defn multinomial-variance
  ([ps ^long n result]
   (fmap! (fn ^double [^double p]
            (* n p (- 1.0 p)))
          (copy! ps result)))
  ([ps ^long n]
   (multinomial-variance ps n (raw ps))))

(defn multinomial-cdf
  ^double [ps ks]
  (throw (java.lang.UnsupportedOperationException. "TODO")))
