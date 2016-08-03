(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.impl.plain
  (:require [uncomplicate.commons.core :refer [with-release let-release]]
            [uncomplicate.fluokitten.core :refer [fmap! fmap]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt pow]]
             [core :refer [dim mrows ncols raw zero axpy! imax row col rows
                           scal! mv trans copy rank native create-raw create]]
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
          (/ (- (pow (nrm2 x-mean) 2.0) (/ (pow (sum x-mean) 2.0) n)) (dec n)))
        0.0)))
  (sd [x]
    (sqrt (variance x))))

(extend-type RealMatrix
  Location
  (mean [a]
    (let [n (mrows a)]
      (if (< 0 n)
        (with-release [ones (entry! (raw (col a 0)) 1)]
          (scal! (/ 1.0 n) (native (mv (trans a) ones))))
        (entry! (create-raw (np/native-factory a) (ncols a)) Double/NaN))))
  Spread
  (variance [a]
    (let [n (mrows a)]
      (if (< 0 n)
        (with-release [ones (entry! (raw (col a 0)) 1)
                       sums (mv (trans a) ones)
                       a-mean (rank (/ -1.0 n) ones sums a)
                       sum-sqr (fmap! (pow 2.0) (native (mv (trans a-mean) ones)))]
          (let-release [res (raw sum-sqr)]
            (dotimes [i (ncols a)]
              (entry! res i (pow (nrm2 (col a-mean i)) 2.0)))
            (scal! (/ 1 (dec n)) (axpy! (/ -1.0 n) sum-sqr res))))
        (create (np/factory a) (ncols a)))))
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
          (/ m2 (- i 2))))))
  (sd [x]
    (sqrt (variance x))))
