;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.impl
  (:require [uncomplicate.commons
             [core :refer [Releaseable release with-release let-release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.neanderthal
             [math :refer [sqrt]]
             [vect-math :refer [sqrt!]]
             [core :refer [transfer ge dim vspace?]]
             [block :refer [column?]]]
            [uncomplicate.neanderthal.internal.api :as na]
            [uncomplicate.bayadera
             [distributions :refer :all]
             [util :refer [srand-int]]
             [mcmc :as mcmc]]
            [uncomplicate.bayadera.internal.protocols :refer :all])
  (:import [clojure.lang IFn]))

(defrecord DatasetImpl [dataset-eng data-matrix]
  Releaseable
  (release [_]
    (release data-matrix))
  na/FactoryProvider
  (factory [_]
    (na/factory data-matrix))
  (native-factory [_]
    (na/native-factory data-matrix))
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
    (sqrt! (variance this)))
  EstimateEngine
  (histogram [this]
    (histogram dataset-eng data-matrix)))

(defn ^:private inc-long ^long [^long x]
  (inc x))

(deftype DirectSampler [samp-engine params sample-count seed-vol]
  Releaseable
  (release [_]
    true)
  RandomSampler
  (init! [_ seed]
    (vreset! seed-vol seed))
  (sample [_ n]
    (sample samp-engine @seed-vol params n))
  (sample [this]
    (sample this sample-count))
  (sample! [this n]
    (vswap! seed-vol inc-long)
    (sample this n))
  (sample! [this]
    (sample! this sample-count)))

;; ==================== Distributions ====================

(deftype GaussianDistribution [factory dist-eng params ^double mu ^double sigma]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [_ options]
    (->DirectSampler (direct-sampler factory :gaussian) params (processing-elements factory)
                     (volatile! (or (:seed options) (srand-int)))))
  (sampler [this]
    (sampler this nil))
  ParameterProvider
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

(deftype UniformDistribution [factory dist-eng params ^double a ^double b]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [_ options]
    (->DirectSampler (direct-sampler factory :uniform) params (processing-elements factory)
                     (volatile! (or (:seed options) (srand-int)))))
  (sampler [this]
    (sampler this nil))
  ParameterProvider
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

(deftype StudentTDistribution [factory dist-eng params ^double nu ^double mu ^double sigma]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [this options]
    (let [walkers (or (:walkers options) (* ^long (processing-elements factory) 256))
          std (sqrt (student-t-variance nu sigma))
          m (student-t-mean nu mu)
          seed (int (or (:seed options) (srand-int)))]
      (with-release [limits (ge (na/native-factory factory) 2 1 [(- m (* 10 std)) (+ m (* 10 std))])]
        (let-release [samp (mcmc-sampler (mcmc-factory factory :student-t) walkers params)]
          (init-position! samp seed limits)
          (init! samp (inc seed))
          (burn-in! samp (max 0 (long (or (:warm-up options) 128))) 8.0)
          samp))))
  (sampler [this]
    (sampler this nil))
  ParameterProvider
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
    (student-t-mean nu mu))
  (mode [_]
    (student-t-mode nu mu))
  (median [_]
    (student-t-median nu mu))
  Spread
  (variance [_]
    (student-t-variance nu sigma))
  (sd [_]
    (sqrt (student-t-variance nu sigma))))

(deftype BetaDistribution [factory dist-eng params ^double a ^double b]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [this options]
    (let [walkers (or (:walkers options) (* ^long (processing-elements factory) 256))
          seed (int (or (:seed options) (srand-int)))]
      (with-release [limits (ge (na/native-factory factory) 2 1 [0 1])]
        (let-release [samp (mcmc-sampler (mcmc-factory factory :beta) walkers params)]
          (init-position! samp seed limits)
          (init! samp (inc seed))
          (burn-in! samp (max 0 (long (or (:warm-up options) 128))) 8.0)
          samp))))
  (sampler [this]
    (sampler this nil))
  ParameterProvider
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

(deftype GammaDistribution [factory dist-eng params ^double theta ^double k]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [this options]
    (let [walkers (or (:walkers options) (* ^long (processing-elements factory) 256))
          seed (int (or (:seed options) (srand-int)))]
      (with-release [limits (ge (na/native-factory factory) 2 1 [0 (+ (gamma-mean theta k)
                                           (* 2 (sqrt (gamma-variance theta k))))])]
        (let-release [samp (mcmc-sampler (mcmc-factory factory :gamma) walkers params)]
          (init-position! samp seed limits)
          (init! samp (inc seed))
          (burn-in! samp (max 0 (long (or (:warm-up options) 128))) 8.0)
          samp))))
  (sampler [this]
    (sampler this nil))
  ParameterProvider
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

(deftype ExponentialDistribution [factory dist-eng params ^double lambda]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [_ options]
    (->DirectSampler (direct-sampler factory :exponential) params (processing-elements factory)
                     (volatile! (or (:seed options) (srand-int)))))
  (sampler [this]
    (sampler this nil))
  ParameterProvider
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

(deftype ErlangDistribution [factory dist-eng params ^double lambda ^long k]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [_ options]
    (->DirectSampler (direct-sampler factory :erlang) params
                     (processing-elements factory) (volatile! (or (:seed options) (srand-int)))))
  (sampler [this]
    (sampler this nil))
  ParameterProvider
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

(deftype LikelihoodImpl [factory lik-eng lik-model]
  Releaseable
  (release [_]
    (release lik-eng))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  EngineProvider
  (engine [_]
    lik-eng)
  ModelProvider
  (model [_]
    lik-model))

(deftype DistributionImpl [factory dist-eng sampler-factory params dist-model]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [_ options]
    (let [walkers (or (:walkers options) (* ^long (processing-elements factory) 2))]
      (let-release [samp (mcmc-sampler sampler-factory walkers params)]
        (init! samp (or (:seed options) (srand-int)))
        (when-let [limits (or (:limits options) (limits dist-model))]
          (init-position! samp (srand-int) limits))
        (when-let [positions (:positions options)]
          (init-position! samp positions))
        samp)))
  (sampler [this]
    (sampler this nil))
  ParameterProvider
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
    (dragan-says-ex "Please estimate mean from a sample."))
  Spread
  (variance [_]
    (dragan-says-ex "Please estimate variance from a sample."))
  (sd [_]
    (dragan-says-ex "Please estimate standard deviation from a sample.")))

(defmacro ^:private with-params-check [model params & body]
  `(if (= (params-size ~model) (if (vspace? ~params) (dim ~params) (count ~params)))
     ~@body
     (dragan-says-ex "Dimension of params must match the distribution."
                     {:required (params-size ~model) :supplied (dim ~params)})))

(deftype DistributionCreator [factory dist-eng sampler-factory dist-model]
  Releaseable
  (release [_]
    (release dist-eng)
    (release sampler-factory))
  IFn
  (invoke [_ params]
    (with-params-check dist-model params
      (->DistributionImpl factory dist-eng sampler-factory (transfer factory params) dist-model)))
  (invoke [this data hyperparams]
    (with-release [params (op data hyperparams)]
      (with-params-check dist-model hyperparams
        (->DistributionImpl factory dist-eng sampler-factory (transfer factory params) dist-model))))
  ModelProvider
  (model [_]
    dist-model))

(deftype PosteriorCreator [factory dist-eng sampler-factory dist-model hyperparams]
  Releaseable
  (release [_]
    (release hyperparams)
    (release dist-eng)
    (release sampler-factory))
  IFn
  (invoke [_ params-data]
    (let-release [params (if (na/compatible? hyperparams params-data)
                           (op params-data hyperparams)
                           (op (transfer factory params-data) hyperparams))]
      (with-params-check dist-model hyperparams
        (->DistributionImpl factory dist-eng sampler-factory params dist-model))))
  ModelProvider
  (model [_]
    dist-model))

(defn posterior-creator [factory model hyperparams]
  (with-params-check model hyperparams
    (->PosteriorCreator factory (distribution-engine factory model)
                        (mcmc-factory factory model) model (transfer factory hyperparams))))
