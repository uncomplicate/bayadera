(ns uncomplicate.bayadera.impl
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecl.core :refer [Releaseable release]]
            [uncomplicate.neanderthal
             [protocols :refer [Group zero create-block write!]]
             [math :refer [sqrt]]
             [core :refer [sum dim nrm2]]
             [native :refer [sv]]]
            [uncomplicate.bayadera.protocols :refer :all]))

(declare univariate-dataset)

(defrecord UnivariateDataSet [engine-factory engine data-vect]
  Releaseable
  (release [_]
    (release engine)
    (release data-vect))
  Group
  (zero [_]
    (univariate-dataset engine-factory (zero data-vect)))
  DataSet
  (data [_]
    data-vect)
  Location
  (mean [_]
    (/ (sum data-vect) (dim data-vect)))
  Spread
  (variance [this]
    (variance engine this))
  (sd [this]
    (sqrt (variance this))))

(defn univariate-dataset [engine-factory n]
  (let [data (create-block engine-factory n)]
    (->UnivariateDataSet engine-factory (dataset-engine engine-factory data) data)))

(deftype GaussianDistribution [eng-factory eng samp params
                               ^double mu ^double sigma]
  Releaseable
  (release [_]
    (release eng)
    (release samp)
    (release params))
  DataSetCreator
  (create-dataset [_ n]
    (univariate-dataset eng-factory n))
  Distribution
  (parameters [_]
    params)
  SamplerProvider
  (sampler [_]
    samp)
  EngineProvider
  (engine [_]
    eng)
  Location
  (mean [_]
    mu)
  Spread
  (variance [_]
    (* sigma sigma))
  (sd [_]
    sigma))

(defn gaussian [eng-factory ^double mu ^double sigma]
  (->GaussianDistribution eng-factory
                          (distribution-engine eng-factory "gaussian")
                          (random-sampler eng-factory "gaussian")
                          (write! (create-block eng-factory 2) (sv mu sigma))
                          mu sigma))
