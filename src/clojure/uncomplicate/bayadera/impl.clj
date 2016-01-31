(ns uncomplicate.bayadera.impl
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecl
             [core :refer [Releaseable release]]
             [toolbox :refer [wrap-int wrap-float]]]
            [uncomplicate.neanderthal
             ;;       [protocols :refer [Container zero raw]]
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [dim create subvector copy! transfer!]]
             [real :refer [entry sum]]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :refer :all]
             [math :refer [log-beta]]])
  (:import [clojure.lang IFn]))

(declare univariate-dataset)

(defrecord UnivariateDataSet [neanderthal-factory dataset-eng data-vect]
  Releaseable
  (release [_]
    (release data-vect))
  ;;Container ;;TODO Probably not needed
  ;;(zero [_]
  ;;  (univariate-dataset dataset-eng (zero data-vect)))
  ;;(raw [_]
  ;;  (univariate-dataset dataset-eng (raw data-vect)))
  DataSet
  (data [_]
    data-vect)
  (raw-result [_]
    (create neanderthal-factory (dim data-vect)))
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

(deftype DirectSampler [neand-factory samp-engine seed params]
  Releaseable
  (release [_]
    true)
  RandomSampler
  (sample! [_ n]
    (let [res (create neand-factory n)]
      (sample! samp-engine seed params res)
      res)))

(deftype GaussianDistribution [bayadera-factory dist-eng params ^double mu ^double sigma]
  Releaseable
  (release [_]
    (release params))
  SamplerProvider
  (sampler [_]
    (->DirectSampler (uncomplicate.neanderthal.protocols/factory bayadera-factory)
                     (gaussian-sampler bayadera-factory)
                     (rand-int Integer/MAX_VALUE) params))
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
    (sv mu (variance this)))
  (variance [_]
    (* sigma sigma)))

(deftype UniformDistribution [bayadera-factory dist-eng params ^double a ^double b]
  Releaseable
  (release [_]
    (release params))
  SamplerProvider
  (sampler [_]
    (->DirectSampler (uncomplicate.neanderthal.protocols/factory bayadera-factory)
                     (uniform-sampler bayadera-factory)
                     (rand-int Integer/MAX_VALUE) params))
  Distribution
  (parameters [_]
    params)
  EngineProvider
  (engine [_]
    dist-eng)
  Location
  (mean [_]
    (/ (+ a b) 2.0))
  Spread
  (mean-variance [this]
    (sv (mean this) (variance this)))
  (variance [_]
    (/ (* (- b a) (- b a)) 12.0)))

(deftype BetaDistribution [bayadera-factory dist-eng params ^double a ^double b]
  Releaseable
  (release [_]
    (release params))
  SamplerProvider
  (sampler [_]
    (let [samp (mcmc-engine (beta-sampler bayadera-factory) (* 44 256 32)
                            params 0 1)]
      (set-position! samp (wrap-int (rand-int Integer/MAX_VALUE)))
      (init! samp (wrap-int (rand-int Integer/MAX_VALUE)))
      (burn-in! samp 512 (wrap-float 2.0))
      (init! samp (wrap-int (rand-int Integer/MAX_VALUE)))
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
    (/ a (+ a b)))
  Spread
  (mean-variance [this]
    (sv (mean this) (variance this)))
  (variance [_]
    (/ (* a b) (* (+ a b) (+ a b) (+ a b 1.0)))))

;;TODO Sort out whether params are on the host or on the GPU!
;;MCMC engine should use all-gpu params, similarily to DirectSampler
(deftype UnivariateDistribution [dist-eng sampler-factory params model]
  Releaseable
  (release [_]
    (release params))
  SamplerProvider
  (sampler [_];;TODO make low/high optional in MCMC-stretch, and also introduce training options in this method
    (let [samp (mcmc-engine sampler-factory (* 44 256 32)
                            params (:lower model) (:upper model))]
      (set-position! samp (wrap-int (rand-int Integer/MAX_VALUE)))
      (init! samp (wrap-int (rand-int Integer/MAX_VALUE)))
      (burn-in! samp 512 (wrap-float 2.0))
      (init! samp (wrap-int (rand-int Integer/MAX_VALUE)))
      (run-sampler! samp 64 (wrap-float 2.0))
      samp))
  Distribution
  (parameters [_]
    params)
  EngineProvider
  (engine [_]
    dist-eng))

(deftype UnivariateDistributionType [factory dist-eng sampler-factory model]
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
