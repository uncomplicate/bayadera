(ns uncomplicate.bayadera.mcmc.opencl.stretch-test
  (:require [midje.sweet :refer :all]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.clojurecl
             [core :refer :all]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.mcmc.opencl.amd-gcn-stretch :refer :all]))

(with-release [dev (first (devices (first (platforms))))
               ctx (context [dev])
               cqueue (command-queue ctx dev :profiling)]
  (facts
   "Test for MCMC stretch engine."
   (let [walker-count (* 2 256 44)
         params (float-array [200 1])
         xs (float-array walker-count)
         run-cnt 140]
     (with-release [mcmc-engine-factory (gcn-stretch-1d-engine-factory ctx cqueue)
                    engine (mcmc-engine mcmc-engine-factory walker-count params)]
       (init-walkers! engine 100)
       (time (burn-in! engine (+ 64 1024))) => :burn-in
       (time (run-sampler! engine 140)) => :autocorrelation-map
       (enq-read! cqueue (.cl-xs engine) xs) => cqueue

       ;;(frequencies (map #(Math/round (* % 10)) xs)) => :Xk
       (< 0.75 (acc-rate engine) 0.80) => true

       ))))
