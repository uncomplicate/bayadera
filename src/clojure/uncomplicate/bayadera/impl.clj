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
  Location
  (mean [_]
    (uniform-mean a b))
  Spread
  (mean-variance [this]
    (sv (uniform-mean a b) (uniform-variance a b)))
  (variance [_]
    (uniform-variance a b)))

(deftype BetaDistribution [bayadera-factory dist-eng params ^double a ^double b]
  Releaseable
  (release [_]
    (release params))
  SamplerProvider
  (sampler [_]
    (let [samp (mcmc-sampler (beta-sampler bayadera-factory) (* 44 256 32)
                            params 0 1)] ;; TODO don't hardcode this
      (set-position! samp (rand-int Integer/MAX_VALUE))
      (init! samp (rand-int Integer/MAX_VALUE))
      (burn-in! samp 512 (wrap-float 2.0))
      (init! samp (rand-int Integer/MAX_VALUE))
      (run-sampler! samp 64 (wrap-float 2.0))
      samp))
  Distribution
  (parameters [_]
    params)
  EngineProvider
  (engine [_]
    dist-eng)
  Location
  (mean [_]
    (beta-mean a b))
  Spread
  (mean-variance [this]
    (sv (beta-mean a b) (beta-variance a b)))
  (variance [_]
    (beta-variance a b)))

;;TODO Sort out whether params are on the host or on the GPU!
(deftype UnivariateDistribution [dist-eng sampler-factory params model]
  Releaseable
  (release [_]
    (release params))
  SamplerProvider
  (sampler [_];;TODO make low/high optional in MCMC-stretch, and also introduce training options in this method
    (let [samp (mcmc-sampler sampler-factory (* 44 256 32)
                             params (lower model) (upper model))]
      (set-position! samp (rand-int Integer/MAX_VALUE))
      (init! samp (rand-int Integer/MAX_VALUE))
      (burn-in! samp 512 (wrap-float 2.0))
      (init! samp (rand-int Integer/MAX_VALUE))
      (run-sampler! samp 64 (wrap-float 2.0))
      samp))
  Distribution
  (parameters [_]
    params)
  EngineProvider
  (engine [_]
    dist-eng))

(deftype UnivariateDistributionCreator [factory dist-eng sampler-factory model]
  Releaseable
  (release [_]
    (release dist-eng)
    (release sampler-factory))
  IFn
  (invoke [_ params];;Use GPU params instead of host later
    (->UnivariateDistribution
     dist-eng sampler-factory
     (transfer! params (create (np/factory factory) (dim params)))
     model))
  (invoke [this data hyperparams]
    (let [params (sv (+ (dim data) (dim hyperparams)))]
      (do
        (copy! data (subvector params 0 (dim data)))
        (copy! hyperparams (subvector params (dim data) (dim hyperparams)))
        (.invoke this params)))))
