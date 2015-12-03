(ns uncomplicate.bayadera.mcmc.opencl.stretch-test
  (:require [midje.sweet :refer :all]
            [clojure.core.async :refer [chan <!!]]
            [clojure.java.io :as io]
            [uncomplicate.clojurecl
             [core :refer :all]
             [toolbox :refer [wrap-int wrap-float]]]
            [uncomplicate.neanderthal
             [core :refer [dim sum transfer! create]]
             [native :refer [sv]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.opencl :refer [gcn-single]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.opencl.amd-gcn :refer [->CLDistributionModel]]
            [uncomplicate.bayadera.mcmc.opencl.amd-gcn-stretch :refer :all]))

(with-release [dev (first (sort-by-cl-version (devices (first (platforms)))))
               ctx (context [dev])
               cqueue (command-queue ctx dev)]
  (facts
   "Test for MCMC stretch engine."
   (let [walker-count (* 2 256 44)
         params (sv [200 1])
         a (wrap-float 8.0)
         xs (sv walker-count)
         run-cnt 140
         gaussian-model
         (->CLDistributionModel "gaussian_mcmc"
                                (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/gaussian.h"))
                                (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/gaussian.cl")))]
     (with-release [mcmc-engine-factory (gcn-stretch-1d-engine-factory
                                         ctx cqueue gaussian-model)
                    engine (mcmc-engine mcmc-engine-factory walker-count params 190 210)]
       (set-position! engine (wrap-int 123))
       (init! engine (wrap-int 1243))
       (< 0.45 (time (burn-in! engine 512 a)) 0.5)  => true
       (init! engine (wrap-int 567))
       (time (:tau (run-sampler! engine 67 a))) => (float 5.51686)
       (enq-read! cqueue (.cl-xs engine) (buffer xs)) => cqueue
       (/ (sum xs) (dim xs)) => (roughly 200.0)))))
