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
       (init-walkers! engine 125)
       (time (do (burn-in! engine 16) (finish! cqueue))) => :burn-in

       (time (run-sampler! engine 256)) => :run-sampler
       (enq-read! cqueue (.cl-xs engine) xs) => cqueue
       (/ (reduce + xs) walker-count) => (roughly 200.0)
       ;;(frequencies (map #(Math/round (* % 10)) xs)) => :Xk
       (< 0.75(acc-rate engine) 0.80) => true

       ))))
