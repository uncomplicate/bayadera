(ns uncomplicate.bayadera.impl
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecl
             [core :refer [Releaseable release]]
             [toolbox :refer [wrap-float]]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [dim create subvector copy! transfer!]]
             [real :refer [entry sum]]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :refer :all]
             [math :refer [log-beta]]
             [distributions :refer :all]])
  (:import [clojure.lang IFn]))

(defrecord UnivariateDataSet [dataset-eng data-vect]
  Releaseable
  (release [_]
    (release data-vect))
  DataSet
  (data [_]
    data-vect)
  (data-count [_]
    (dim data-vect))
  Location
  (mean [_]
    (/ (sum data-vect) (dim data-vect)))
  Spread
  (mean-variance [this]
    (mean-variance dataset-eng data-vect))
  (variance [this]
    (entry (mean-variance this) 1)))

(deftype DirectSampler [samp-engine params]
  Releaseable
  (release [_]
    true)
  RandomSampler
  (sample! [_ res]
    (sample! samp-engine (rand-int Integer/MAX_VALUE) params res)))

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
  (mean-variance [this]
    (sv mu (gaussian-variance sigma)))
  (variance [_]
    (gaussian-variance sigma)))

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
  (mean-variance [this]
    (sv (uniform-mean a b) (uniform-variance a b)))
  (variance [_]
    (uniform-variance a b)))

(defn ^:private prepare-mcmc-sampler
  [sampler-factory walkers params lower-limit upper-limit
   & {:keys [warm-up iterations a]
      :or {warm-up 512
           iterations 64
           a 2.0}}]
  (let [a (wrap-float a)
        samp (mcmc-sampler sampler-factory walkers params lower-limit upper-limit)];;TODO make low/high optional in MCMC-stretch
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
  (sampler [_ options]
    (let [walkers (or (:walkers options)
                      (* (long (processing-elements bayadera-factory)) 32))]
      (apply prepare-mcmc-sampler (beta-sampler bayadera-factory)
             walkers params 0.0 1.0 options)))
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
  (mean-variance [this]
    (sv (beta-mean a b) (beta-variance a b)))
  (variance [_]
    (beta-variance a b)))

;;TODO Sort out whether params are on the host or on the GPU!
(deftype UnivariateDistribution [bayadera-factory dist-eng sampler-factory params dist-model]
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
    dist-model))

(deftype UnivariateDistributionCreator [bayadera-factory dist-eng sampler-factory dist-model]
  Releaseable
  (release [_]
    (release dist-eng)
    (release sampler-factory))
  IFn
  (invoke [_ params];;Use GPU params instead of host later
    (->UnivariateDistribution
     bayadera-factory dist-eng sampler-factory
     (transfer! params (create (np/factory bayadera-factory) (dim params)))
     dist-model))
  (invoke [this data hyperparams]
    (let [params (sv (+ (dim data) (dim hyperparams)))]
      (do
        (copy! data (subvector params 0 (dim data)))
        (copy! hyperparams (subvector params (dim data) (dim hyperparams)))
        (.invoke this params))))
  ModelProvider
  (model [_]
    dist-model))
