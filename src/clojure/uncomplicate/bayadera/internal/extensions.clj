;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.extensions
  (:require [uncomplicate.commons.core
             :refer [with-release let-release Releaseable release releaseable?]]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.neanderthal.internal.api :as na]
            [uncomplicate.neanderthal
             [math :refer [sqrt sqr]]
             [vect-math :refer [sqr! sqrt!]]
             [core :refer [dim mrows ncols raw axpy! imax row col rows
                           scal! mv! copy! rk! vctr native!]]
             [real :refer [entry! sum nrm2]]]
            [uncomplicate.bayadera.protocols :refer :all])
  (:import [clojure.lang Sequential]
           [uncomplicate.neanderthal.internal.api Vector Matrix]))

(defn vector-variance! [x work]
  (let [n (dim x)]
    (if (< 0 n)
      (let [x-mean (axpy! x (entry! work (- (/ (sum x) n))))]
        (/ (sqr (nrm2 x-mean)) n))
      0.0)))

(extend-type Vector
  Location
  (mean [x]
    (let [n (dim x)]
      (if (< 0 n)
        (/ (sum x) n)
        Double/NaN)))
  Spread
  (variance [x]
    (if (< 0 (dim x))
      (with-release [work (raw x)]
        (vector-variance! x work))
      0.0))
  (sd [x]
    (sqrt (variance x))))

(defn matrix-mean! [a ones res]
  (if (and (< 0 (dim a)))
    (mv! (/ 1.0 (ncols a)) a ones res)
    (entry! res Double/NaN)))

(defn matrix-variance! [a ones work res]
  (let [n (ncols a)]
    (if (< 0 (dim a))
      (let [row-sum (mv! a ones res)
            a-mean (rk! (/ -1.0 n) row-sum ones (copy! a work))]
        (scal! (/ 1 n) (mv! (sqr! a-mean) ones res)))
      (entry! res 0))))

(extend-type Matrix
  Dataset
  (data [this]
    this)
  Location
  (mean
    ([a]
     (let-release [res (na/create-vector (na/factory a) (mrows a) false)]
       (mean a res)))
    ([a res]
     (with-release [ones (entry! (raw (row a 0)) 1.0)]
       (matrix-mean! a ones res))))
  Spread
  (variance
    ([a]
     (let-release [res (na/create-vector (na/factory a) (mrows a) false)]
       (variance a res)))
    ([a res]
     (with-release [ones (entry! (raw (row a 0)) 1.0)
                    work (raw a)]
       (matrix-variance! a ones work res))))
  (sd [a]
    (sqrt! (variance a))))

(extend-type Sequential
  Location
  (mean [this]
    (if (first this)
      (if (number? (first this))
        (loop [n 0 acc 0.0 s (seq this)]
          (if s
            (recur (inc n) (+ acc (double (first s))) (next s))
            (/ acc n)))
        (let [c (count (first this))
              acc (double-array c)]
          (if (< 0 c)
            (loop [n 0 s (seq this)]
              (if s
                (do
                  (loop [j 0 es (seq (first s))]
                    (when es
                      (aset acc j (+ (aget acc j) (double (first es))))
                      (recur (inc j) (next es))))
                  (recur (inc n) (next s)))
                (into [] (map (fn ^double [^double sum] (/ sum n))) acc)))
            Double/NaN)))
      Double/NaN))
  Spread
  (variance [this]
    (if (first this)
      (if (number? (first this))
        (loop [i 1 mu 0.0 m2 0.0 x (first this) s (next this)]
          (if x
            (let [delta (- (double x) mu)
                  new-mu (+ mu (/ delta i))]
              (recur (inc i)
                     new-mu
                     (+ m2 (* delta (- (double x) new-mu)))
                     (first s)
                     (next s)))
            (if (< i 3)
              0.0
              (/ m2 (dec i)))))
        (let [c (count (first this))
              mus (double-array c)
              m2s (double-array c)]
          (if (< 0 c)
            (loop [i 1 xs (first this) s (next this)]
              (if xs
                (do
                  (loop [j 0 es (seq xs)]
                    (when es
                      (let [e (double (first es))
                            mu (aget mus j)
                            delta (- e mu)
                            new-mu (+ mu (/ delta i))]
                        (aset mus j new-mu)
                        (aset m2s j (+ (aget m2s j) (* delta (- e new-mu)))))
                      (recur (inc j) (next es))))
                  (recur (inc i) (first s) (next s)))
                (if (< i 3)
                  (vec (take c (repeat 0.0)))
                  (into [] (map (fn (^double [^double m2] (/ m2 (dec i))))) m2s))))
            0.0)))
      0.0))
  (sd [x]
    (if (number? x)
      (sqrt (variance x))
      (fmap sqrt (variance x)))))
