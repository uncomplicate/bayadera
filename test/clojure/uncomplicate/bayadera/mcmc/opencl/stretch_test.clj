(ns uncomplicate.bayadera.mcmc.opencl.stretch-test
  (:require [midje.sweet :refer :all]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [core :refer [dim sum]]
             [native :refer [sv]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.opencl :refer [clv read! write! gcn-single]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.mcmc.opencl.amd-gcn-stretch :refer :all]))

(with-release [dev (first (devices (first (platforms))))
               ctx (context [dev])
               cqueue (command-queue ctx dev)]
  (facts
   "Test for MCMC stretch engine."
   (let [walker-count (* 2 256 44)
         params (sv [200 1])
         a (float-array [8.0])
         xs (sv walker-count)
         run-cnt 140]
     (with-release [mcmc-engine-factory (gcn-stretch-1d-engine-factory
                                         ctx cqueue "gaussian")
                    engine (mcmc-engine mcmc-engine-factory walker-count params 190 210)]
       (set-position! engine (int-array [123]))
       (init! engine (int-array [1243]))
       (< 0.45 (time (burn-in! engine 512 a)) 0.5)  => true
       (init! engine (int-array [567]))
       (time (:tau (run-sampler! engine 67 a))) => (float 5.51686)
       (enq-read! cqueue (.cl-xs engine) (buffer xs)) => cqueue
       (/ (sum xs) (dim xs)) => (roughly 200.0)))))
