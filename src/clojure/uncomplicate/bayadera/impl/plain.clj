(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.impl.plain
  (:require [uncomplicate.commons.core :refer [with-release let-release]]
            [uncomplicate.fluokitten.core :refer [fmap! fmap]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt pow]]
             [core :refer [dim mrows ncols raw zero axpy! imax row col rows
                           scal! mv trans copy rank! native
                           create-raw create create-vector]]
             [real :refer [entry! entry sum nrm2]]]
            [uncomplicate.bayadera.protocols :refer :all])
  (:import [clojure.lang IPersistentCollection]
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

(extend-type IPersistentCollection
  Location
  (mean [this]
    (loop [n 0 sum 0.0 s (seq this)]
      (if s
        (recur (inc n) (+ sum (double (first s))) (next s))
        (/ sum n))))
  Spread
  (variance [this]
    (loop [i 1 mu 0.0 m2 0.0 x (first this) s (next this)]
      (if x
        (let [delta (- (double x) mu)]
          (recur (inc i)
                 (+ mu (/ delta i))
                 (+ m2 (* delta (- (double x) mu)))
                 (first s)
                 (next s)))
        (if (< i 3)
          0.0
          (/ m2 (- i 1))))))
  (sd [x]
    (sqrt (variance x))))
