(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl.mcmc-stretch-test
  (:require [midje.sweet :refer :all]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.commons.core :refer [with-release wrap-float]]
            [uncomplicate.clojurecl.core :refer [enq-read!]]
            [clojure.java.io :as io]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [core :refer [create-vector row ncols]]
             [real :refer [sum]]
             [native :refer [sv sge]]
             [opencl :refer [opencl-single]]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.opencl
             [models :refer [gaussian-model]]
             [amd-gcn-stretch :refer :all]]))

(with-release [dev (first (sort-by-cl-version (devices (first (platforms)))))
               ctx (context [dev])
               cqueue (command-queue ctx dev)]

  (let [walker-count (* 2 256 44)
        a 8.0]
    (with-release [neanderthal-factory (opencl-single ctx cqueue)
                   mcmc-engine-factory (gcn-stretch-factory
                                        ctx cqueue neanderthal-factory
                                        gaussian-model)
                   cl-params (create-vector neanderthal-factory [200 1])
                   limits (sge 2 1 [180.0 220.0])
                   engine (mcmc-sampler mcmc-engine-factory walker-count cl-params)]
      (facts
       "Test for MCMC stretch engine."
       (init-position! engine 123 limits)
       (init! engine 1243)
       (burn-in! engine 5120 a)
       (< 0.45 (acc-rate! engine a) 0.5)  => true
       (:tau (:autocorrelation (run-sampler! engine 67 a))) => (sv 7.1493382453)
       (with-release [xs (sample! engine walker-count)]
         (/ (sum (row xs 0)) (ncols xs)) => (roughly 200.0))))))
