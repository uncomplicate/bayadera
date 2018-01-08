;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.math
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [asum sum]]
             [math :refer [exp round? magnitude gamma lgamma]]
             [vect-math :refer [linear-frac linear-frac! lgamma!]]])
  (:import [org.apache.commons.math3.special Beta Gamma]))

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

(defn log-factorial
  "Natural logarithm of a factorial of a positive real x."
  ^double [^double x]
  (lgamma (inc x)))

(defn factorial
  "Factorial function:
  Computes the product of all natural numbers from 0 to x.
  If x is a real number, computes the exponent of log-factorial."
  ^double [^double x]
  (gamma (inc x)))

(defn log-beta
  "Natural logarithm of the beta function."
  ^double [^double a ^double b]
  (- (+ (lgamma a) (lgamma b)) (lgamma (+ a b))))

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
  (exp (log-binco n k)))

(defn log-multico!
  "Destructive version of [[log-multico]] that uses `xks` as working memory.."
  ^double [xks]
  (- ^double (log-factorial (asum xks)) ^double (sum (lgamma! (linear-frac! xks 1.0)))))

(defn log-multico
  "Natural logarithm of multinomial coefficient of a RealVector xks:
  Computes the number of ways of partitioning N object
  into k groups of size x1, x2, ... , xk;
  where k = (dim xks) and N = (sum xks)"
  ^double [xks]
  (with-release [xks-copy (linear-frac xks)]
    (log-multico! xks-copy)))

(defn multico
  "Destructive version of [[log-multico]] that uses `xks` as working memory.."
  ^double [xks]
  (exp (log-multico! xks)))

(defn multico
  "Multinomial coefficient of a RealVector xks:
  Computes the number of ways of partitioning N object
  into k groups of size x1, x2, ... , xk;
  where k = (dim xks) and N = (sum xks)"
  ^double [xks]
  (exp (log-multico xks)))
