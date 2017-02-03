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
            [uncomplicate.fluokitten.core :refer [fmap! fmap]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt pow]]
             [core :refer [dim mrows ncols raw zero axpy! imax row col rows
                           scal! mv trans copy rank! native
                           create-raw create create-vector]]
             [real :refer [entry! entry sum nrm2]]]
            [uncomplicate.bayadera.protocols :refer :all])
  (:import [clojure.lang Sequential]
           [uncomplicate.neanderthal.protocols RealVector RealMatrix]))

(extend-type RealVector
  Location
  (mean [x]
    (let [n (dim x)]
      (if (< 0 n)
        (/ (sum x) n)
        Double/NaN)))
  Spread
  (variance [x]
    (let [n (dim x)]
      (if (< 0 n)
        (with-release [x-mean (axpy! x (entry! (raw x) (- (/ (sum x) n))))]
          (/ (- (pow (nrm2 x-mean) 2.0) (/ (pow (sum x-mean) 2.0) n)) n))
        0.0)))
  (sd [x]
    (sqrt (variance x))))

(extend-type RealMatrix
  Dataset
  (data [this]
    this)
  Location
  (mean [a]
    (let [m (mrows a)
          n (ncols a)]
      (if (and (< 0 m) (< 0 n))
        (if (< 1 m)
          (with-release [ones (entry! (raw (row a 0)) 1)]
            (scal! (/ 1.0 n) (native (mv a ones))))
          (mean (row a 0)))
        (entry! (create-raw (np/native-factory a) m) Double/NaN))))
  Spread
  (variance [a]
    (let [m (mrows a)
          n (ncols a)]
      (if (and (< 0 m) (< 0 n))
        (if (< 1 m)
          (with-release [ones (entry! (raw (row a 0)) 1)
                         sums (mv a ones)
                         a-mean (rank! (/ -1.0 n) sums ones (copy a))
                         sum-sqr (fmap! (pow 2.0) (native (mv a-mean ones)))]
            (let-release [res (raw sum-sqr)]
              (dotimes [i m]
                (entry! res i (pow (nrm2 (row a-mean i)) 2.0)))
              (scal! (/ 1 n) (axpy! (/ -1.0 n) sum-sqr res))))
          (create-vector (np/native-factory a) (variance (row a 0))))
        (create (np/native-factory a) m))))
  (sd [a]
    (fmap! sqrt (variance a))))

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
