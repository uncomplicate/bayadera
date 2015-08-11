(ns uncomplicate.bayadera.mcmc.opencl.stretch-test
  (:require [midje.sweet :refer :all]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.clojurecl
             [core :refer :all]]
            [uncomplicate.bayadera.mcmc.opencl.amd-gcn-stretch :refer :all]))

(with-release [dev (first (devices (first (platforms))))
               ctx (context [dev])
               cqueue (command-queue ctx dev :profiling)]
  (facts
   "Test for MCMC stretch engine."
   (let [walker-count (long (Math/pow 2 13))
         params (float-array [200 1])
         xs (float-array walker-count)]
     (with-release [mcmc-engine-factory (gcn-stretch-1d-engine-factory ctx cqueue)
                    engine (mcmc-engine mcmc-engine-factory walker-count params)]
       (doto engine (init!) (burn-in! 1000))
       (enq-read! cqueue
                  (move! engine)
                  xs) => cqueue
                  ;;(frequencies (map #(Math/round (* % 10)) xs)) => :Xk
                  (< 0.75 (acc-rate engine) 0.80) => true

                  ))))
