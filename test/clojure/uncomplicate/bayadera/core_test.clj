(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.core-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [math :refer [log exp sqrt]]
             [core :refer [col axpy! mrows ge]]
             [real :refer [sum]]]
            [uncomplicate.bayadera.core :refer :all]
            [uncomplicate.bayadera.internal
             [protocols :as p]]))

(defmacro roughly100 [exp]
  `(let [v# (double ~exp)]
     (roughly v# (/ v# 100.0)) ))

(defn test-dataset [bayadera-factory]
  (let [data-size (* 31 (long (Math/pow 2 16)))]
    (with-release [data-matrix (ge bayadera-factory 22 data-size (repeatedly (* 22 data-size) rand))
                   data-set (dataset bayadera-factory data-matrix)]
      (facts
       "Test histogram"
       (/ (sum (col (:pdf (histogram data-set)) 4)) (mrows (:pdf (histogram data-set)))) => (roughly 1.0))
      (facts
       "Test variance"
       (sum (axpy! -1 (variance (p/data data-set)) (variance data-set))) => (roughly 0 0.003))
      (facts
       "Test mean"
       (sum (axpy! -1 (mean (p/data data-set)) (mean data-set))) => (roughly 0 0.003)))))
