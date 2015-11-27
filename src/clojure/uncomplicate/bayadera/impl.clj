(ns uncomplicate.bayadera.impl
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecl.core :refer [Releaseable release]]
            [uncomplicate.neanderthal
             [protocols :refer [Container zero raw  FactoryProvider factory]]
             [math :refer [sqrt]]
             [core :refer [dim create]]
             [real :refer [entry sum]]
             [native :refer [sv]]]
            [uncomplicate.bayadera.protocols :refer :all]))

(declare univariate-dataset)

(defrecord UnivariateDataSet [dataset-eng data-vect]
  Releaseable
  (release [_]
    (and
     (release dataset-eng)
     (release data-vect)))
  Container
  (zero [_]
    (univariate-dataset dataset-eng (zero data-vect)))
  (raw [_]
    (univariate-dataset dataset-eng (raw data-vect)))
  DataSet
  (data [_]
    data-vect)
  MeasureProvider
  (measures [this]
    this)
  Location
  (mean [_]
    (/ (sum data-vect) (dim data-vect)))
  Spread
  (mean-variance [this]
    (mean-variance dataset-eng data-vect))
  (variance [this]
    (entry (mean-variance this) 1))
  (sd [this]
    (sqrt (variance this))))

(deftype UnivariateDistribution [dist-eng samp dataset-eng params]
  Releaseable
  (release [_] true)
  DataSetCreator
  (create-dataset [_ n]
    (univariate-dataset dataset-eng n))
  SamplerProvider
  (sampler [_]
    samp)
  EngineProvider
  (engine [_]
    dist-eng)
  MeasureProvider
  (measures [_]
    params))

(defrecord GaussianParameters [^double mu ^double sigma]
  Location
  (mean [_]
    mu)
  Spread
  (mean-variance [this]
    (sv mu (variance this)))
  (variance [_]
    (* sigma sigma))
  (sd [_]
    sigma))

(defrecord UniformParameters [^double a ^double b]
  Location
  (mean [_]
    (/ (+ a b) 2.0))
  Spread
  (mean-variance [this]
    (sv (mean this) (variance this)))
  (variance [_]
    (/ (* (- b a) (- b a)) 12.0))
  (sd [this]
    (sqrt (variance this))))
