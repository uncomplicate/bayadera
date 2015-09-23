(ns uncomplicate.bayadera.distributions.core-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [core :refer [dim sum nrm2]]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [core :refer :all]
             [impl :refer :all]]
            [uncomplicate.bayadera.opencl.amd-gcn :refer [gcn-engine-factory]]))

(with-release [dev (first (devices (first (platforms))))
               ctx (context [dev])
               cqueue (command-queue ctx dev)]
  (facts
   "Core functions for gaussian distribution."
   (let [sample-count (* 2 256 44000)
         mu 200.0
         sigma 10.0
         temp-reduction-acc (double-array (/ sample-count 256))]
     (with-release [dist-engine (gcn-engine-factory ctx cqueue)
                    dist (gaussian dist-engine mu sigma)
                    cl-sample (time (sample dist sample-count))]

       (mean dist) => mu
       (sd dist) => sigma
       ;;(mean cl-sample) => mu
       (first (variance cl-sample)) => mu
       (Math/sqrt (second (variance cl-sample))) => sigma
       ;;(enq-read! cqueue (.reduce-acc (.engine (p/data cl-sample))) temp-reduction-acc)
       ;;(vec temp-reduction-acc) => :x
       ;;(enq-read! cqueue (.reduction-acc (.engine cl-sample)) temp-reduction-acc)
       ;;(vec temp-reduction-acc) => :m2n
       ))))
