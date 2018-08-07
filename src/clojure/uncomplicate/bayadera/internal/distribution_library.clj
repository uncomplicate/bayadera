;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.distribution-library
  (:require [uncomplicate.commons
             [core :refer [Releaseable release with-release let-release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [math :refer [sqrt]]
             [core :refer [ge]]]
            [uncomplicate.neanderthal.internal.api :as na]
            [uncomplicate.bayadera
             [distributions :refer :all]
             [util :refer [srand-int]]]
            [uncomplicate.bayadera.internal
             [protocols :refer :all]
             [impl :refer [create-direct-sampler]]]))

;; ==================== Distributions ====================

(deftype GaussianDistribution [factory samp dist-eng params ^double mu ^double sigma]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [_ options]
    (create-direct-sampler @samp params (processing-elements factory) (or (:seed options) (srand-int))))
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

(deftype UniformDistribution [factory samp dist-eng params ^double a ^double b]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [_ options]
    (create-direct-sampler @samp params (processing-elements factory) (or (:seed options) (srand-int))))
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

(deftype StudentTDistribution [factory mcmc dist-eng params ^double nu ^double mu ^double sigma]
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
        (let-release [samp (create-sampler @mcmc seed walkers params)]
          (init-position! samp (inc seed) limits)
          (burn-in! samp (max 0 (long (or (:warm-up options) 128))) 8.0)
          samp))))
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

(deftype BetaDistribution [factory mcmc dist-eng params ^double a ^double b]
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
        (let-release [samp (create-sampler @mcmc seed walkers params)]
          (init-position! samp (inc seed) limits)
          (burn-in! samp (max 0 (long (or (:warm-up options) 128))) 8.0)
          samp))))
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

(deftype GammaDistribution [factory mcmc dist-eng params ^double theta ^double k]
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
        (let-release [samp (create-sampler @mcmc seed walkers params)]
          (init-position! samp (inc seed) limits)
          (burn-in! samp (max 0 (long (or (:warm-up options) 128))) 8.0)
          samp))))
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

(deftype ExponentialDistribution [factory samp dist-eng params ^double lambda]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [_ options]
    (create-direct-sampler @samp params (processing-elements factory) (or (:seed options) (srand-int))))
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

(deftype ErlangDistribution [factory samp dist-eng params ^double lambda ^long k]
  Releaseable
  (release [_]
    (release params))
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  SamplerProvider
  (sampler [_ options]
    (create-direct-sampler @samp params (processing-elements factory) (or (:seed options) (srand-int))))
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

(deftype LibraryLikelihood [factory lik-eng lik-model]
  na/MemoryContext
  (compatible? [_ o]
    (na/compatible? factory o))
  EngineProvider
  (engine [_]
    lik-eng)
  ModelProvider
  (model [_]
    lik-model))
