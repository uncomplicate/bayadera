(ns uncomplicate.bayadera.opencl.mcmc-stretch-test
  (:require [midje.sweet :refer :all]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.commons.core :refer [with-release wrap-float]]
            [uncomplicate.clojurecl.core :refer [enq-read!]]
            [clojure.java.io :as io]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [core :refer [dim create]]
             [real :refer [sum]]
             [native :refer [sv]]
             [block :refer [buffer]]
             [opencl :refer [gcn-single]]]
            [uncomplicate.neanderthal.opencl :refer [with-gcn-engine sv-cl]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.opencl
             [generic :refer [gaussian-model]]
             [amd-gcn-stretch :refer :all]]))

(with-release [dev (first (sort-by-cl-version (devices (first (platforms)))))
               ctx (context [dev])
               cqueue (command-queue ctx dev)]
  (with-gcn-engine cqueue
    (facts
     "Test for MCMC stretch engine."
     (let [walker-count (* 2 256 44)
           a (wrap-float 8.0)
           xs (sv walker-count)
           run-cnt 140]
       (with-release [neanderthal-factory (gcn-single ctx cqueue)
                      mcmc-engine-factory (gcn-stretch-1d-factory
                                           ctx cqueue neanderthal-factory
                                           gaussian-model)
                      cl-params (sv-cl [200 1])
                      engine (mcmc-sampler mcmc-engine-factory walker-count
                                          cl-params 190 210)]
         (init! engine 1243)
         (set-position! engine 123)
         (< 0.45 (time (burn-in! engine 512 a)) 0.5)  => true
         (init! engine 567)
         (time (:tau (run-sampler! engine 67 a))) => (float 5.51686)
         (enq-read! cqueue (.cl-xs engine) (buffer xs)) => cqueue
         (/ (sum xs) (dim xs)) => (roughly 200.0))))))
