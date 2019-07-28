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
             [core :refer [Releaseable release let-release info]]
             [utils :refer [dragan-says-ex generate-seed]]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.neanderthal
             [vect-math :refer [sqrt!]]
             [core :refer [transfer transfer! dim vspace? subvector vctr ge]]]
            [uncomplicate.neanderthal.internal.api :as na]
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
  (sample! [this]
    (sample! this sample-count))
  (sample! [this n-or-res]
    (vswap! seed-vol inc-long)
    (if (integer? n-or-res)
      (let-release [res (ge params 1 n-or-res {:raw true})];;TODO support dim>1. Infer dim from params (use ge for params in library instead of vctr). I can't do that, since params might have odd entries! Use (dimension model)
        (sample samp-engine @seed-vol params res))
      (if (na/compatible? params n-or-res)
        (sample samp-engine @seed-vol params n-or-res)
        (dragan-says-ex "Data matrix is incompatible with the sampler's data accessor."
                        {:data (info n-or-res)
                         :accessor (na/data-accessor params)})))))

(defn create-direct-sampler [samp-engine params sample-count seed]
  (->DirectSampler samp-engine params sample-count (volatile! seed)))

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

(deftype DistributionImpl [factory dist-eng sampler-factory params]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [_ options]
    (let [walkers (or (:walkers options) (* ^long (processing-elements factory) 2))]
      (let-release [samp (create-sampler sampler-factory (or (:seed options) (generate-seed)) walkers params)]
        (when-let [limits (or (:limits options) (limits (model dist-eng)))]
          (init-position! samp (generate-seed) limits))
        (when-let [positions (:positions options)]
          (init-position! samp positions))
        samp)))
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
    (dragan-says-ex "Please estimate mean from a sample."))
  Spread
  (variance [_]
    (dragan-says-ex "Please estimate variance from a sample."))
  (sd [_]
    (dragan-says-ex "Please estimate standard deviation from a sample.")))

(defmacro ^:private with-params-check [model params & body]
  `(if (= (params-size ~model) (if (vspace? ~params) (dim ~params) (count ~params)))
     (do ~@body)
     (dragan-says-ex "Dimension of params must match the distribution."
                     {:required (params-size ~model) :supplied (dim ~params)})))

(deftype DistributionCreator [factory dist-eng sampler-factory]
  Releaseable
  (release [_]
    (release dist-eng)
    (release sampler-factory))
  IFn
  (invoke [_ params]
    (with-params-check (model dist-eng) params
      (->DistributionImpl factory dist-eng sampler-factory (transfer factory params))))
  (invoke [this data hyperparams]
    (let-release [params (vctr factory (+ (dim data) (dim hyperparams)))]
      (with-params-check (model dist-eng) hyperparams
        (transfer! data (subvector params 0 (dim data)))
        (transfer! hyperparams (subvector params (dim data) (dim hyperparams)))
        (->DistributionImpl factory dist-eng sampler-factory params))))
  ModelProvider
  (model [_]
    (model dist-eng)))

(deftype PosteriorCreator [factory dist-eng sampler-factory hyperparams]
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
      (with-params-check (model dist-eng) hyperparams
        (->DistributionImpl factory dist-eng sampler-factory params))))
  ModelProvider
  (model [_]
    (model dist-eng)))

(defn posterior-creator [factory model hyperparams]
  (with-params-check model hyperparams
    (->PosteriorCreator factory (distribution-engine factory model)
                        (mcmc-factory factory model) (transfer factory hyperparams))))
