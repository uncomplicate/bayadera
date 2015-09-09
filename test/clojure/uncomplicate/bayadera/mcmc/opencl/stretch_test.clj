(ns uncomplicate.bayadera.mcmc.opencl.stretch-test
  (:require [midje.sweet :refer :all]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [core :refer [dim sum]]
             [native :refer [sv]]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.mcmc.opencl.amd-gcn-stretch :refer :all]
            [uncomplicate.bayadera.distributions.opencl.amd-gcn :refer :all]
            [uncomplicate.neanderthal.opencl :refer [clv read! write!]]
            [uncomplicate.neanderthal.opencl.amd-gcn :refer [gcn-single]]))

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
     (with-release [mcmc-engine-factory (gcn-stretch-1d-engine-factory
                                         ctx cqueue (logpdf-source "gaussian"))
                    engine (mcmc-engine mcmc-engine-factory walker-count params 190 210)]
       (set-position! engine (int-array [123]))
       (init! engine (int-array [1243]))
       (< 0.45 (time (burn-in! engine 512 a)) 0.5)  => true
       (init! engine (int-array [567]))
       (time (run-sampler! engine 67 a)) => :run-sampler
       (enq-read! cqueue (.cl-xs engine) (.buffer xs)) => cqueue
       (/ (sum xs) (dim xs)) => (roughly 200.0)
       ;;(frequencies (map #(Math/round (* % 10)) xs)) => :Xk


       ))))

(with-release [dev (first (devices (first (platforms))))
               ctx (context [dev])
               cqueue (command-queue ctx dev :profiling)]
  (facts
   "Test for distribution engines."
   (let [sample-count (* 2 256 44)
         params (sv [200 10])
         xs (sv sample-count)]
     (with-release [dist-engine-factory (gcn-distribution-engine-factory
                                         ctx cqueue "gaussian")
                    engine (random-sampler dist-engine-factory)
                    neanderthal-engine (gcn-single ctx cqueue)
                    cl-xs (clv neanderthal-engine sample-count)
                    cl-params (write! (clv neanderthal-engine 2) params)]
       (do
         (time (sample! engine (float-array [1238798]) sample-count
                        (.buffer cl-params) (.buffer cl-xs)))
         (read! cl-xs xs)
         (/ (sum xs) (dim xs)))
       => (roughly 200.0)

       ))))
