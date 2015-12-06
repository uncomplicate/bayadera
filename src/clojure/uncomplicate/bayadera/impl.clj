(ns uncomplicate.bayadera.impl
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecl.core :refer [Releaseable release]]
            [uncomplicate.neanderthal
             [protocols :refer [Container zero raw]]
             [math :refer [sqrt]]
             [core :refer [dim create]]
             [real :refer [entry sum]]
             [native :refer [sv]]]
            [uncomplicate.bayadera.protocols :refer :all]))

(declare univariate-dataset)

(defrecord UnivariateDataSet [dataset-eng data-vect]
  Releaseable
  (release [_]
    (release data-vect))
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

(deftype UnivariateDistribution [bayadera-factory dist-eng samp meas params]
  Releaseable
  (release [_]
    (release params))
  Distribution
  (parameters [_]
    params)
  FactoryProvider
  (factory [_]
    bayadera-factory)
  SamplerProvider
  (sampler [_]
    samp)
  EngineProvider
  (engine [_]
    dist-eng)
  MeasureProvider
  (measures [_]
    meas))

(deftype DirectSampler [neand-factory samp seed params]
  Releaseable
  (release [_]
    true)
  RandomSampler
  (sample! [_ n]
    (let [res (create neand-factory n)]
      (sample! samp seed params res)
      res)))

(deftype GaussianDistribution [bayadera-factory dist-eng samp params ^double mu ^double sigma]
  Releaseable
  (release [_]
    (release params))
  FactoryProvider
  (factory [_]
    bayadera-factory)
  SamplerProvider
  (sampler [_]
    samp)
  EngineProvider
  (engine [_]
    dist-eng)
  MeasureProvider
  (measures [this]
    this)
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

(defrecord GaussianMeasures [^double mu ^double sigma]
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

(defrecord UniformMeasures [^double a ^double b]
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

(defrecord BetaMeasures [^double alpha ^double beta]
  Location
  (mean [_]
    (/ alpha (+ alpha beta)))
  Spread
  (mean-variance [this]
    (sv (mean this) (variance this)))
  (variance [_]
    (/ (* alpha beta) (* (+ alpha beta) (+ alpha beta) (+ alpha beta 1.0))))
  (sd [this]
    (sqrt (variance this))))
