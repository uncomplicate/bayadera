(ns uncomplicate.bayadera.distributions.core-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [core :refer [dim sum nrm2]]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [core :refer :all]
             [impl :refer :all]]
            [uncomplicate.bayadera.opencl.amd-gcn :refer [gcn-engine-factory]]))

(with-release [dev (first (devices (first (platforms))))
               ctx (context [dev])
               cqueue (command-queue ctx dev)]
  (facts
   "Core functions for gaussian distribution."
   (let [sample-count (* 2 256 44)]
     (with-release [dist-engine (gcn-engine-factory ctx cqueue)
                    dist (gaussian dist-engine 200000000 10)
                    cl-sample (sample dist sample-count)]

       (mean dist) => 200.0
       (sd dist) => 10.0
       (mean cl-sample) => (roughly 200.0)
       (sd cl-sample) => (roughly 10.0)

       ))))
