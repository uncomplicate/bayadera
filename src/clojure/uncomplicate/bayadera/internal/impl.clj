;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.impl
  (:require [clojure.java.io :as io]
            [uncomplicate.commons.core
             :refer [Releaseable release with-release let-release wrap-float wrap-int]]
            [uncomplicate.fluokitten.core :refer [op fmap!]]
            [uncomplicate.neanderthal
             [math :refer [sqrt abs]]
             [core :refer [transfer ge dim]]]
            [uncomplicate.neanderthal.internal.api :as na]
            [uncomplicate.bayadera
             [protocols :refer :all]
             [math :refer [log-beta]]
             [distributions :refer :all]
             [util :refer [srand-int]]
             [mcmc :as mcmc]])
  (:import [clojure.lang IFn]))

(def ^:private INVALID_PARAMS_MESSAGE
  "Invalid params dimension. Must be %s, but is %s.")

(def ^:private USE_SAMPLE_MSG
  "This distribution's %s is a random variable. Please draw a sample to estimate it.")

(defrecord DatasetImpl [dataset-eng data-matrix]
  Releaseable
  (release [_]
    (release data-matrix))
  Dataset
  (data [_]
    data-matrix)
  Location
  (mean [this]
    (data-mean dataset-eng data-matrix))
  Spread
  (variance [this]
    (data-variance dataset-eng data-matrix))
  (sd [this]
    (fmap! sqrt (variance this)))
  EstimateEngine
  (histogram [this]
    (histogram dataset-eng data-matrix)))

(deftype DirectSampler [samp-engine params sample-count ^ints fseed]
  Releaseable
  (release [_]
    true)
  RandomSampler
  (init! [_ seed]
    (aset fseed 0 (int seed)))
  (sample [_ n]
    (sample samp-engine (aget fseed 0) params n))
  (sample [this]
    (sample this sample-count))
  (sample! [this n]
    (init! this (aset fseed 0 (inc (aget fseed 0))))
    (sample this n))
  (sample! [this]
    (sample! this sample-count)))

;; ==================== Distributions ====================

(deftype GaussianDistribution [bayadera-factory dist-eng params
                               ^double mu ^double sigma]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? bayadera-factory o))
  SamplerProvider
  (sampler [_ options]
    (->DirectSampler (direct-sampler bayadera-factory :gaussian) params
                     (processing-elements bayadera-factory) (wrap-int (or (:seed options) (srand-int)))))
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
    mu)
  (mode [_]
    mu)
  (median [_]
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
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? bayadera-factory o))
  SamplerProvider
  (sampler [_ options]
    (->DirectSampler (direct-sampler bayadera-factory :uniform) params
                     (processing-elements bayadera-factory) (wrap-int (or (:seed options) (srand-int)))))
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
    (uniform-mean a b))
  (mode [_]
    (uniform-mode a b))
  (median [_]
    (uniform-median a b))
  Spread
  (variance [_]
    (uniform-variance a b))
  (sd [_]
    (sqrt (uniform-variance a b))))

(deftype TDistribution [bayadera-factory dist-eng params ^double nu ^double mu ^double sigma]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? bayadera-factory o))
  SamplerProvider
  (sampler [this options]
    (let [walkers (or (:walkers options) (* (long (processing-elements bayadera-factory)) 32))
          std (sqrt (t-variance nu sigma))
          m (t-mean nu mu)
          seed (int (or (:seed options) (srand-int)))]
      (with-release [limits (ge (na/native-factory bayadera-factory)
                                2 1 [(- m (* 10 std)) (+ m (* 10 std))])]
        (let-release [samp (mcmc-sampler (mcmc-factory bayadera-factory :t) walkers params)]
          (init-position! samp seed limits)
          (init! samp (inc seed))
          (burn-in! samp (max 0 (long (or (:warm-up options) 128))) 8.0)
          samp))))
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
    (t-mean nu mu))
  (mode [_]
    (t-mode nu mu))
  (median [_]
    (t-median nu mu))
  Spread
  (variance [_]
    (t-variance nu sigma))
  (sd [_]
    (sqrt (t-variance nu sigma))))

(deftype BetaDistribution [bayadera-factory dist-eng params ^double a ^double b]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? bayadera-factory o))
  SamplerProvider
  (sampler [this options]
    (let [walkers (or (:walkers options) (* (long (processing-elements bayadera-factory)) 32))
          seed (int (or (:seed options) (srand-int)))]
      (with-release [limits (ge (na/native-factory bayadera-factory) 2 1 [0 1])]
        (let-release [samp (mcmc-sampler (mcmc-factory bayadera-factory :beta) walkers params)]
          (init-position! samp seed limits)
          (init! samp (inc seed))
          (burn-in! samp (max 0 (long (or (:warm-up options) 128))) 8.0)
          samp))))
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
  (median [_]
    (beta-median a b))
  (mode [_]
    (beta-mode a b))
  Spread
  (variance [_]
    (beta-variance a b))
  (sd [_]
    (sqrt (beta-variance a b))))

(deftype GammaDistribution [bayadera-factory dist-eng params ^double theta ^double k]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? bayadera-factory o))
  SamplerProvider
  (sampler [this options]
    (let [walkers (or (:walkers options) (* (long (processing-elements bayadera-factory)) 32))
          seed (int (or (:seed options) (srand-int)))]
      (with-release [limits (ge (na/native-factory bayadera-factory) 2 1 [0 (+ (gamma-mean theta k)
                                           (* 2 (sqrt (gamma-variance theta k))))])]
        (let-release [samp (mcmc-sampler (mcmc-factory bayadera-factory :gamma) walkers params)]
          (init-position! samp seed limits)
          (init! samp (inc seed))
          (burn-in! samp (max 0 (long (or (:warm-up options) 128))) 8.0)
          samp))))
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
    (gamma-mean theta k))
  (mode [_]
    (gamma-mode theta k))
  (median [_]
    (throw (java.lang.UnsupportedOperationException. "No closed form for gamma median.")))
  Spread
  (variance [_]
    (gamma-variance theta k))
  (sd [_]
    (sqrt (gamma-variance theta k))))

(deftype ExponentialDistribution [bayadera-factory dist-eng params ^double lambda]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? bayadera-factory o))
  SamplerProvider
  (sampler [_ options]
    (->DirectSampler (direct-sampler bayadera-factory :exponential) params
                     (processing-elements bayadera-factory) (wrap-int (or (:seed options) (srand-int)))))
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
    (exponential-mean lambda))
  (mode [_]
    (exponential-mode lambda))
  (median [_]
    (exponential-median lambda))
  Spread
  (variance [_]
    (exponential-variance lambda))
  (sd [_]
    (sqrt (exponential-variance lambda))))

(deftype ErlangDistribution [bayadera-factory dist-eng params
                             ^double lambda ^long k]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? bayadera-factory o))
  SamplerProvider
  (sampler [_ options]
    (->DirectSampler (direct-sampler bayadera-factory :erlang) params
                     (processing-elements bayadera-factory) (wrap-int (or (:seed options) (srand-int)))))
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
    (erlang-mean lambda k))
  (mode [_]
    (erlang-mode lambda k))
  (median [_]
    (erlang-median lambda k))
  Spread
  (variance [_]
    (erlang-variance lambda k))
  (sd [_]
    (sqrt (erlang-variance lambda k))))

(deftype DistributionImpl [bayadera-factory dist-eng sampler-factory params dist-model]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? bayadera-factory o))
  SamplerProvider
  (sampler [_ options]
    (let [walkers (or (:walkers options) (* (long (processing-elements bayadera-factory)) 2))]
      (let-release [samp (mcmc-sampler sampler-factory walkers params)]
        (init! samp (or (:seed options) (srand-int)))
        (when-let [limits (or (:limits options) (limits dist-model))]
          (init-position! samp (srand-int) limits))
        (when-let [positions (:positions options)]
          (init-position! samp positions))
        samp)))
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
    (release dist-eng)
    (release sampler-factory))
  IFn
  (invoke [_ params]
    (if (= (params-size dist-model) (dim params))
      (->DistributionImpl bayadera-factory dist-eng sampler-factory
                          (transfer (na/factory bayadera-factory) params) dist-model)
      (throw (IllegalArgumentException.
              (format INVALID_PARAMS_MESSAGE (params-size dist-model) (dim params))))))
  (invoke [this data hyperparams]
    (.invoke this (op data hyperparams)))
  ModelProvider
  (model [_]
    dist-model))

(deftype PosteriorCreator [^IFn dist-creator hyperparams]
  Releaseable
  (release [_]
    (release hyperparams)
    (release dist-creator))
  IFn
  (invoke [_ data]
    (.invoke dist-creator data hyperparams)))
