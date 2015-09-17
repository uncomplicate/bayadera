(ns uncomplicate.bayadera.distributions.opencl.generic
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [protocols :refer [BlockCreator]]
             [core :refer [sum dim]]
             [native :refer [sv]]
             [opencl :refer [clv write!]]]
            [uncomplicate.bayadera.protocols :refer :all])
  (:import [uncomplicate.neanderthal.protocols Vector] ))

(extend-type Vector
  Location
  (mean [x]
    (/ (sum x) (dim x))))

(deftype CLGaussianDistribution [dist-eng samp vect-factory cl-params
                                 ^float mu ^float sigma]
  Releaseable
  (release [_]
    (release dist-eng)
    (release samp)
    (release cl-params))
  Distribution
  (parameters [_]
    cl-params)
  BlockCreator
  (create-block [_ n]
    (clv vect-factory n))
  SamplerProvider
  (sampler [_]
    samp)
  DistributionEngineProvider
  (distribution-engine [_]
    dist-eng)
  Location
  (mean [_]
    mu)
  Spread
  (sd [_]
    sigma))

(defn gaussian [dist-factory ^double mu ^double sigma]
  (let [vect-factory (vector-factory dist-factory)]
    (->CLGaussianDistribution (dist-engine dist-factory "gaussian")
                              (random-sampler dist-factory "gaussian")
                              vect-factory
                              (write! (clv vect-factory 2) (sv mu sigma))
                              mu sigma)))
