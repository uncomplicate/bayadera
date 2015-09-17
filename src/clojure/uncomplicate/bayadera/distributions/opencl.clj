(ns uncomplicate.bayadera.distributions.opencl
  (:require [clojure.java.io :as io]
            [me.raynes.fs :as fsc]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [protocols :refer [BlockCreator]]
             [native :refer [sv]]]
            [uncomplicate.bayadera.protocols :refer :all])
  (:import [uncomplicate.neanderthal.protocols Vector Block] ))

;;TODO move this to a generic namespace
(extend Vector
  Location
  (mean [x]
        (/ (sum x) (dim x))))
move to package opencl.generic
(deftype CLGaussianParameters [cl-params ^float mu ^float sigma ]
  Releaseable
  (release [_]
    (release params))
  Block
  (buffer
    cl-params)
  Location
  (mean [_]
    mu)
  Spread
  (sd [_]
    sigma))

(deftype CLUnivariateDistribution [dist-eng samp neand-factory params]
  Releaseable
  (release [_]
    (release dist-eng)
    (release samp)
    (release params))
  Distribution
  (parameters [_]
    params)
  BlockCreator
  (create-block [_ n]
    (clv neand-factory n))
  SamplerProvider
  (sampler [_]
    samp)
  DistributionEngineProvider
  (distribution-engine [_]
    dist-eng))

(defn gaussian [dist-factory mu sigma]
  (->UnivariateDistribution (distribution-engine dist-factory "gaussian")
                            (sampler dist-factory "gaussian")
                            (->GaussianCLParameters (cl-buffer ctx (* Float)))
                          mu sigma))
