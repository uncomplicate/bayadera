(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.impl
  (:require [clojure.java.io :as io]
            [uncomplicate.commons.core :refer [Releaseable release wrap-float]]
            [uncomplicate.fluokitten.core :refer [op fmap!]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [ecount transfer]]
             [native :refer [sge]]]
            [uncomplicate.bayadera
             [protocols :refer :all]
             [math :refer [log-beta]]
             [distributions :refer :all]])
  (:import [clojure.lang IFn]))

(def ^:private INVALID_PARAMS_MESSAGE
  "Invalid params dimension. Must be %s, but is %s.")

(def ^:private USE_SAMPLE_MSG
  "This distribution's %s is a random variable. Please draw a sample to estimate it.")

(defrecord DataSetImpl [dataset-eng data-matrix]
  Releaseable
  (release [_]
    (release data-matrix))
  DataSet
  (data [_]
    data-matrix)
  Location
  (mean [this]
    (means dataset-eng data-matrix))
  Spread
  (variance [this]
    (variances dataset-eng data-matrix))
  (sd [this]
    (fmap! sqrt (variance this))))

(deftype DirectSampler [samp-engine params]
  Releaseable
  (release [_]
    true)
  RandomSampler
  (sample! [_ n]
    (sample! samp-engine (rand-int Integer/MAX_VALUE) params n)))

;; ==================== Distributions ====================

(deftype GaussianDistribution [bayadera-factory dist-eng params
                               ^double mu ^double sigma]
  Releaseable
  (release [_]
    (release params))
  SamplerProvider
  (sampler [_]
    (->DirectSampler (gaussian-sampler bayadera-factory) params))
  Distribution
  (parameters [_]
    params)
  EngineProvider
  (engine [_]
    dist-eng)
  ModelProvider
  (model [_]
    (model dist-eng))
  Location
  (mean [_]
    mu)
  Spread
  (variance [_]
    (gaussian-variance sigma))
  (sd [_]
    sigma))

(deftype UniformDistribution [bayadera-factory dist-eng params ^double a ^double b]
  Releaseable
  (release [_]
    (release params))
  SamplerProvider
  (sampler [_]
    (->DirectSampler (uniform-sampler bayadera-factory) params))
  Distribution
  (parameters [_]
    params)
  EngineProvider
  (engine [_]
    dist-eng)
  ModelProvider
  (model [_]
    (model dist-eng))
  Location
  (mean [_]
    (uniform-mean a b))
  Spread
  (variance [_]
    (uniform-variance a b))
  (sd [_]
    (sqrt (uniform-variance a b))))

(defn ^:private prepare-mcmc-sampler
  [sampler-factory walkers params lower-limit upper-limit
   & {:keys [warm-up iterations a]
      :or {warm-up 512
           iterations 64
           a 2.0}}]
  (let [a (wrap-float a)
        samp (mcmc-sampler sampler-factory walkers params lower-limit upper-limit)]
    (set-position! samp (rand-int Integer/MAX_VALUE))
    (init! samp (rand-int Integer/MAX_VALUE))
    (burn-in! samp warm-up a)
    (init! samp (rand-int Integer/MAX_VALUE))
    (run-sampler! samp iterations a)
    samp))

(deftype BetaDistribution [bayadera-factory dist-eng params ^double a ^double b]
  Releaseable
  (release [_]
    (release params))
  SamplerProvider
  (sampler [this options]
    (let [walkers (or (:walkers options)
                      (* (long (processing-elements bayadera-factory)) 32))
          beta-model (model this)]
      (apply prepare-mcmc-sampler (beta-sampler bayadera-factory)
             walkers params (lower beta-model) (upper beta-model) options)))
  (sampler [this]
    (sampler this nil))
  Distribution
  (parameters [_]
    params)
  EngineProvider
  (engine [_]
    dist-eng)
  ModelProvider
  (model [_]
    (model dist-eng))
  Location
  (mean [_]
    (beta-mean a b))
  Spread
  (variance [_]
    (beta-variance a b))
  (sd [_]
    (sqrt (beta-variance a b))))

(deftype DistributionImpl [bayadera-factory dist-eng sampler-factory params dist-model]
  Releaseable
  (release [_]
    (release params))
  SamplerProvider
  (sampler [_ options]
    (let [walkers (or (:walkers options)
                      (* (long (processing-elements bayadera-factory)) 32))]
      (apply prepare-mcmc-sampler sampler-factory walkers params
             (lower dist-model) (upper dist-model) options)))
  (sampler [this]
    (sampler this nil))
  Distribution
  (parameters [_]
    params)
  EngineProvider
  (engine [_]
    dist-eng)
  ModelProvider
  (model [_]
    dist-model)
  Location
  (mean [_]
    (throw (UnsupportedOperationException. (format USE_SAMPLE_MSG "mean"))))
  Spread
  (variance [_]
    (throw (UnsupportedOperationException. (format USE_SAMPLE_MSG "variance"))))
  (sd [_]
    (throw (UnsupportedOperationException. (format USE_SAMPLE_MSG "standard deviation")))))

(deftype DistributionCreator [bayadera-factory dist-eng sampler-factory dist-model]
  Releaseable
  (release [_]
    (and
     (release dist-eng)
     (release sampler-factory)))
  IFn
  (invoke [_ params]
    (if (= (params-size dist-model) (ecount params))
      (->DistributionImpl bayadera-factory dist-eng sampler-factory
                          (transfer (np/factory bayadera-factory) params)
                          dist-model)
      (throw (IllegalArgumentException.
              (format INVALID_PARAMS_MESSAGE (params-size dist-model)
                      (ecount params))))))
  (invoke [this data hyperparams]
    (.invoke this (op data hyperparams)))
  ModelProvider
  (model [_]
    dist-model))

(deftype PosteriorCreator [^IFn dist-creator hyperparams]
  Releaseable
  (release [_]
    (and (release hyperparams)
         (release dist-creator)))
  IFn
  (invoke [_ data]
    (.invoke dist-creator data hyperparams)))
