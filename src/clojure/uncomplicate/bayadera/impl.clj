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
    (let [d (zero data-vect)]
      (univariate-dataset engine-factory (dataset-engine engine-factory d) d)))
  DataSet
  (data [_]
    data-vect)
  Location
  (mean [_]
    (/ (sum data-vect) (dim data-vect)))
  Spread
  (mean-variance [this]
    (mean-variance engine))
  (variance [this]
    (entry (mean-variance engine) 1))
  (sd [this]
    (sqrt (variance this))))

(defn univariate-dataset [engine-factory n]
  (let [data (create-block engine-factory n)]
    (->UnivariateDataSet engine-factory (dataset-engine engine-factory data) data)))

(defrecord GaussianParameters [^double mu ^double sigma]
  Location
  (mean [_]
    mu)
  Spread
  (mean-variance [x]
    (sv mu (variance x)))
  (variance [_]
    (* sigma sigma))
  (sd [_]
    sigma))

(deftype UnivariateDistribution [eng-factory eng samp params]
  Releaseable
  (release [_]
    (release eng)
    (release samp))
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
    (mean params))
  Spread
  (mean-variance [_]
    (mean-variance params))
  (variance [_]
    (variance params))
  (sd [_]
    (sd params)))

(defn gaussian [eng-factory ^double mu ^double sigma]
  (let [params (sv mu sigma)]
    (->UnivariateDistribution eng-factory
                            (distribution-engine eng-factory "gaussian" params)
                            (random-sampler eng-factory "gaussian" params)
                            (->GaussianParameters mu sigma))))
