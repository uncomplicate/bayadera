(ns uncomplicate.bayadera.impl
  (:require [clojure.java.io :as io]
            [uncomplicate.commons.core :refer [Releaseable release wrap-float]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [dim create-vector compatible? vect?]]
             [real :refer [entry sum]]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :refer :all]
             [math :refer [log-beta]]
             [distributions :refer :all]])
  (:import [clojure.lang IFn]))

(def ^:private INVALID_PARAMS_MESSAGE
  "Invalid params dimension. Must be %s, but is %s.")

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

(deftype UnivariateDistribution [bayadera-factory dist-eng
                                 sampler-factory params dist-model]
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

(deftype UnivariateDistributionCreator [bayadera-factory dist-eng
                                        sampler-factory dist-model]
  Releaseable
  (release [_]
    (and
     (release dist-eng)
     (release sampler-factory)))
  IFn
  (invoke [_ params]
    (if (= (params-size dist-model)
           (if (vect? params) (dim params) (count params)))
      (->UnivariateDistribution bayadera-factory dist-eng sampler-factory
                                (create-vector (np/factory bayadera-factory) params)
       dist-model)
      (throw (IllegalArgumentException.
              (format INVALID_PARAMS_MESSAGE (params-size dist-model)
                      (dim data))))))
  (invoke [this data hyperparams]
    (this (op data hyperparams)))
  ModelProvider
  (model [_]
    dist-model))

(deftype UnivariatePosteriorPriorCreator [bayadera-factory dist-eng
                                          sampler-factory hyperparams dist-model]
  Releaseable
  (release [_]
    (and (release hyperparams)
         (release dist-eng)
         (release sampler-factory)))
  IFn
  (invoke [_ data]
    (let [expected-dim (- (long (params-size dist-model)) (dim hyperparams))
          data-dim (if (vect? data) (dim data) (count data))]
      (if (= expected-dim data-dim)
        (let [params (if (compatible? hyperparams data)
                       (op data hyperparams)
                       (op (create-vector (np/factory bayadera-factory) data)
                           hyperparams))]
          (->UnivariateDistribution bayadera-factory dist-eng sampler-factory
                                    params dist-model))
        (throw (IllegalArgumentException.
                (format INVALID_PARAMS_MESSAGE expected-dim data-dim))))))
  ModelProvider
  (model [_]
    dist-model))

(defn univariate-distribution-creator [factory model]
  (->UnivariateDistributionCreator factory
                                   (distribution-engine factory model)
                                   (mcmc-factory factory model)
                                   model))

(defn univariate-posterior-creator [factory model]
  (->UnivariateDistributionCreator factory
                                   (posterior-engine factory model)
                                   (mcmc-factory factory model)
                                   model))

(defn univariate-posterior-prior-creator [factory params model]
  (->UnivariatePosteriorPriorCreator factory
                                     (posterior-engine factory model)
                                     (mcmc-factory factory model)
                                     params
                                     model))
