(ns uncomplicate.bayadera.distributions.gaussian-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [core :refer [dim sum]]
             [native :refer [sv]]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.distributions.opencl.amd-gcn :refer :all]
            [uncomplicate.neanderthal.opencl :refer [clv read! write!]]
            [uncomplicate.neanderthal.opencl.amd-gcn :refer [gcn-single]]))

(with-release [dev (first (devices (first (platforms))))
               ctx (context [dev])
               cqueue (command-queue ctx dev)]
  (facts
   "Test for gaussian sampling engine."
   (let [sample-count (* 2 256 44)
         params (sv [200 10])]
     (with-release [dist-engine-factory (gcn-distribution-engine-factory
                                         ctx cqueue "gaussian")
                    sampler (random-sampler dist-engine-factory)
                    neanderthal-engine (gcn-single ctx cqueue)
                    cl-xs (clv neanderthal-engine sample-count)
                    cl-params (write! (clv neanderthal-engine 2) params)]
       (do
         (time (sample! sampler (float-array [123]) sample-count
                        (.buffer cl-params) (.buffer cl-xs)))
         (/ (sum cl-xs) (dim cl-xs)))
       => (roughly 200.0)))))

(with-release [dev (first (devices (first (platforms))))
               ctx (context [dev])
               cqueue (command-queue ctx dev)]
  (facts
   "Test for gaussian distribution engine."
   (let [sample-count (* 1024 2 256 44)
         params (sv [-13 20])
         xs (sv sample-count)
         pdf (sv sample-count)]
     (with-release [dist-engine-factory (gcn-distribution-engine-factory
                                         ctx cqueue "uniform")
                    sampler (random-sampler dist-engine-factory)
                    dist-engine (distribution-engine dist-engine-factory)
                    neanderthal-engine (gcn-single ctx cqueue)
                    cl-xs (clv neanderthal-engine sample-count)
                    cl-pdf (clv neanderthal-engine sample-count)
                    cl-params (write! (clv neanderthal-engine 2) params)]
       (do
         (sample! sampler (float-array [123]) sample-count
                  (.buffer cl-params) (.buffer cl-xs))
         (time (logpdf! dist-engine sample-count (.buffer cl-params)
                        (.buffer cl-xs) (.buffer cl-pdf))))

       (/ (sum cl-xs) (dim cl-xs)) => (roughly 3.5)
       (/ (sum cl-pdf) (dim cl-pdf)) => (roughly (/ 1.0 33))))))
