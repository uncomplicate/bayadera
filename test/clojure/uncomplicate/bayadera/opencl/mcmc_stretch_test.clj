(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl.mcmc-stretch-test
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
             [opencl :refer [gcn-single sv-cl]]]
            [uncomplicate.neanderthal.opencl :refer [with-gcn-engine]]
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
           run-cnt 140]
       (with-release [neanderthal-factory (gcn-single ctx cqueue)
                      mcmc-engine-factory (gcn-stretch-factory
                                           ctx cqueue neanderthal-factory
                                           gaussian-model)
                      xs (sv walker-count)
                      cl-params (sv-cl [200 1])
                      lower (sv 190.0)
                      upper (sv 210.0)
                      engine (mcmc-sampler mcmc-engine-factory walker-count
                                           cl-params lower upper)]
         (init! engine 1243)
         (set-position! engine 123)
         (< 0.45 (time (burn-in! engine 512 a)) 0.5)  => true
         (init! engine 567)
         (time (:tau (run-sampler! engine 67 a))) => (sv 5.5218153)
         (enq-read! cqueue (.cl-xs engine) (buffer xs)) => cqueue
         (/ (sum xs) (dim xs)) => (roughly 200.0))))))
