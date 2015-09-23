(ns uncomplicate.bayadera.impl
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecl.core :refer [Releaseable release]]
            [uncomplicate.neanderthal
             [protocols :refer [Group zero create-block write!]]
             [math :refer [sqrt]]
             [core :refer [sum dim nrm2]]
             [real :refer [entry]]
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
  (mean-variance [this]
    (mean-variance engine this))
  (variance [this]
    (entry (mean-variance engine this) 1))
  (sd [this]
    (sqrt (variance this))))

(defn univariate-dataset [engine-factory n]
  (let [data (create-block engine-factory n)]
    (->UnivariateDataSet engine-factory (dataset-engine engine-factory data) data)))

(deftype GaussianDistribution [eng-factory eng samp ^double mu ^double sigma]
  Releaseable
  (release [_]
    (release eng)
    (release samp))
  DataSetCreator
  (create-dataset [_ n]
    (univariate-dataset eng-factory n))
  Distribution
  (parameters [_]
    (sv mu sigma))
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
  (mean-variance [_]
    :TODO)
  (variance [_]
    (* sigma sigma))
  (sd [_]
    sigma))

(defn gaussian [eng-factory ^double mu ^double sigma]
  (let [params (sv mu sigma)]
    (->GaussianDistribution eng-factory
                            (distribution-engine eng-factory "gaussian" params)
                            (random-sampler eng-factory "gaussian" params)
                            mu sigma)))
