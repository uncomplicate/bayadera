(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl.amd-gcn-test
  (:require [midje.sweet :refer :all]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.commons.core :refer [with-release wrap-float]]
            [uncomplicate.clojurecl.core :refer [enq-read!]]
            [clojure.java.io :as io]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [core :refer [dim create-ge-matrix row col imax submatrix entry ncols transfer cols]]
             [real :refer [sum]]
             [native :refer [sv]]
             [block :refer [buffer]]
             [opencl :refer [opencl-single clge with-engine]]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.opencl.amd-gcn :refer :all]))


(with-release [dev (first (sort-by-cl-version (devices (first (platforms)))))
               ctx (context [dev])
               cqueue (command-queue ctx dev)]

  (let [data-size (* 44 (long (Math/pow 2 16)))]
    (with-release [neanderthal-factory (opencl-single ctx cqueue)
                   dataset-engine (gcn-dataset-engine ctx cqueue)
                   data-matrix (create-ge-matrix neanderthal-factory
                                                 22 data-size
                                                 (repeatedly (* 22 data-size) rand))]
      (facts
       "Test histogram"
       (/ (sum (col (:pdf (histogram dataset-engine data-matrix)) 4)) 256) => (roughly 1.0)))))
