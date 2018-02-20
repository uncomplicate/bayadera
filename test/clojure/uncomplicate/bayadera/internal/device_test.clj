(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.device-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [ge vctr row col ncols native axpy!]]
             [real :refer [sum entry]]
             [opencl :refer [opencl-float]]]
            [uncomplicate.bayadera.core :refer :all]
            [uncomplicate.bayadera.internal.protocols :as p]))


(defn test-dataset [bayadera-factory]
  (let [data-size (* 31 (long (Math/pow 2 16)))]
    (with-release [data-matrix (ge bayadera-factory 22 data-size (repeatedly (* 22 data-size) rand))
                   data-set (dataset bayadera-factory data-matrix)]

      (facts
       "Test histogram"
       (/ (sum (col (:pdf (histogram data-set)) 4)) (.WGS bayadera-factory)) => (roughly 1.0))

      (facts
       "Test variance"
       (sum (axpy! -1 (variance (p/data data-set)) (p/variance data-set))) => (roughly 0 0.003)))))

(defn test-mcmc [bayadera-factory gaussian-model]
  (let [walker-count (* 2 256 44)
        a 8.0
        mcmc-engine-factory (p/mcmc-factory bayadera-factory gaussian-model)]
    (with-release [cl-params (vctr bayadera-factory [200 1])
                   limits (ge bayadera-factory 2 1 [180.0 220.0])]
      (let [engine (p/mcmc-sampler mcmc-engine-factory walker-count cl-params)]
        (facts
         "Test MCMC stretch engine."
         (p/init-position! engine 123 limits)
         (p/init! engine 1243)
         (p/burn-in! engine 5120 a)
         (< 0.45 (p/acc-rate! engine a) 0.5)  => true
         (entry (:tau (:autocorrelation (run-sampler! engine 67 a))) 0) => (roughly 5.757689952850342)
         (with-release [xs (p/sample! engine walker-count)]
           (/ (sum (row xs 0)) (ncols xs)) => (roughly 200.0)))))))
