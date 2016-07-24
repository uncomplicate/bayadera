(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.math
  (:require [uncomplicate.fluokitten.core :refer [fold]]
            [uncomplicate.neanderthal
             [core :refer [asum]]
             [math :refer [exp round? pow magnitude]]])
  (:import [org.apache.commons.math3.special Gamma Beta Erf]))

(defn log-gamma
  "Natural logarithm of the Gamma function:
  http://en.wikipedia.org/wiki/Gamma_function#The_log-gamma_function"
  ^double [^double x]
  (Gamma/logGamma x))

(defn gamma
  "Gamma function: http://en.wikipedia.org/wiki/Gamma_function"
  ^double [^double x]
  (Gamma/gamma x))

(defn regularized-gamma-q
  ^double [^double a ^double x]
  (Gamma/regularizedGammaQ a x))

(defn regularized-gamma-p
  ^double [^double a ^double x]
  (Gamma/regularizedGammaP a x))

(defn incomplete-gamma-l
  ^double [^double s ^double x]
  (* (gamma s) (regularized-gamma-p s x)))

(defn incomplete-gamma-u
  ^double [^double s ^double x]
  (* (gamma s) (regularized-gamma-q s x)))

(let [table-size (double 1000)
      log-factorial-table (double-array
                           (map (fn [^double x]
                                  (log-gamma (inc x)))
                                (range 0 (inc table-size))))]
  (defn log-factorial
    "Natural logarithm of a factorial of a positive real x."
    ^double [^double x]
    (if (< x 0)
      (throw (IllegalArgumentException.
              (format "x have to be positive, but is %f." x)))
      (if (and (< x table-size) (round? x))
        (aget log-factorial-table x)
        (log-gamma (inc x))))))

(let [factorial-table (double-array
                       (reduce (fn [acc ^double x]
                                 (conj acc (* (double (peek acc)) x)))
                               [1.0]
                               (range 1 171)))]
  (defn factorial
    "Factorial function:
  Computes the product of all natural numbers from 0 to x.
  If x is a real number, computes the exponent of log-factorial."
    ^double [^double x]
    (if (and (<= 0 x 170) (round? x))
      (aget factorial-table x)
      (exp (log-factorial x)))))

(defn log-beta
  "Natural logarithm of the beta function."
  ^double [^double a ^double b]
  (- (+ (log-gamma a) (log-gamma b)) (log-gamma (+ a b))))

(defn beta
  "Beta function of a and b."
  ^double [^double a ^double b]
  (exp (log-beta a b)))

(defn regularized-beta
  ^double [^double x ^double a ^double b]
  (Beta/regularizedBeta x a b))

(defn log-binco
  "Natural logarithm of the binomial coefficient of n and k:
  Computes the number of ways to choose k items out of n items."
  ^double [^double n ^double k]
  (- (log-factorial n) (log-factorial k) (log-factorial (- n k))))

(defn binco
  "Binomial coefficient of n and k:
  Computes the number of ways to choose k items out of n items."
  ^double [^double n ^double k]
  (if (and (<= 0 n 170) (round? n))
    (/ (factorial n) (factorial k) (factorial (- n k)))
    (exp (log-binco n k))))

(defn log-multico
  "Natural logarithm of multinomial coefficient of a RealVector xks:
  Computes the number of ways of partitioning N object
  into k groups of size x1, x2, ... , xk;
  where k = (dim xks) and N = (sum xks)"
  ^double [xks]
  (fold (fn ^double [^double acc ^double xk]
          (- acc (log-factorial xk)))
        (log-factorial (asum xks))
        xks))

(defn multico
  "Multinomial coefficient of a RealVector xks:
  Computes the number of ways of partitioning N object
  into k groups of size x1, x2, ... , xk;
  where k = (dim xks) and N = (sum xks)"
  ^double [xks]
  (exp (log-multico xks)))

(defn erfc
  "The complementary error function: erfc(x) = 1 - erf(x)"
  ^double [^double x]
  (Erf/erfc x))

(defn erf
  "Error function: erf(x) = 2/√π 0∫x e-t2dt"
  ^double [^double x]
  (Erf/erf x))
