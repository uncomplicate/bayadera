(ns uncomplicate.bayadera.mcmc.opencl.stretch-test
  (:require [midje.sweet :refer :all]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [core :refer [dim sum]]
             [native :refer [sv]]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.mcmc.opencl.amd-gcn-stretch :refer :all]))

(with-release [dev (first (devices (first (platforms))))
               ctx (context [dev])
               cqueue (command-queue ctx dev :profiling)]
  (facts
   "Test for MCMC stretch engine."
   (let [walker-count (* 2 256 44)
         params (sv [200 1])
         a (float-array [8.0])
         xs (sv walker-count)
         run-cnt 140]
     (with-release [mcmc-engine-factory (gcn-stretch-1d-engine-factory ctx cqueue)
                    engine (mcmc-engine mcmc-engine-factory walker-count params 190 210)]
       (set-position! engine (int-array [123]))
       (init! engine (int-array [1243]))
       (time (burn-in! engine 512 a))  => :burn-in
       (init! engine (int-array [567]))
       (time (run-sampler! engine 67 a)) => :run-sampler
       (enq-read! cqueue (.cl-xs engine) (.buffer xs)) => cqueue
       (Math/abs (- 200 (/ (sum xs) (dim xs)))) => 0.0;;(roughly 200.0)
       ;;(frequencies (map #(Math/round (* % 10)) xs)) => :Xk


       ))))
