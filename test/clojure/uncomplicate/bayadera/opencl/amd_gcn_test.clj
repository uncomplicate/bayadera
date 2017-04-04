(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl.amd-gcn-test
  (:require [midje.sweet :refer :all]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.commons.core :refer [with-release]]
            [clojure.java.io :as io]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [core :refer [ge vctr row col ncols]]
             [real :refer [sum entry]]
             [opencl :refer [opencl-float]]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.opencl
             [models :refer [distributions]]
             [amd-gcn :refer :all]]))

(with-release [dev (first (sort-by-cl-version (devices (first (platforms)))))
               ctx (context [dev])
               cqueue (command-queue ctx dev)]

  (let [data-size (* 44 (long (Math/pow 2 16)))]
    (with-release [neanderthal-factory (opencl-float ctx cqueue)
                   dataset-engine (gcn-dataset-engine ctx cqueue)
                   data-matrix (ge neanderthal-factory 22 data-size (repeatedly (* 22 data-size) rand))]
      (facts
       "Test histogram"
       (/ (sum (col (:pdf (histogram dataset-engine data-matrix)) 4)) 256) => (roughly 1.0))))

  (let [walker-count (* 2 256 44)
        a 8.0]
    (with-release [neanderthal-factory (opencl-float ctx cqueue)
                   mcmc-engine-factory (gcn-stretch-factory ctx cqueue neanderthal-factory
                                                            (:gaussian distributions))
                   cl-params (vctr neanderthal-factory [200 1])
                   limits (ge neanderthal-factory 2 1 [180.0 220.0])
                   engine (mcmc-sampler mcmc-engine-factory walker-count cl-params)]
      (facts
       "Test MCMC stretch engine."
       (init-position! engine 123 limits)
       (init! engine 1243)
       (burn-in! engine 5120 a)
       (< 0.45 (acc-rate! engine a) 0.5)  => true
       (entry (:tau (:autocorrelation (run-sampler! engine 67 a))) 0) => (roughly 7.1493382453)
       (with-release [xs (sample! engine walker-count)]
         (/ (sum (row xs 0)) (ncols xs)) => (roughly 200.0))))))
