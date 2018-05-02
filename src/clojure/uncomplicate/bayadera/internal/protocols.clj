;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.protocols
  (:require [uncomplicate.commons.core :refer [Releaseable release]]))

(defrecord Histogram [limits pdf bin-ranks]
  Releaseable
  (release [this]
    (release limits)
    (release pdf)
    (release bin-ranks)))

(defprotocol Location
  (mean [this])
  (median [this])
  (mode [this]))

(defprotocol Spread
  (hdi [this])
  (variance [this])
  (sd [this]))

(defprotocol Association
  (cov [x y])
  (corr [x y]))

(defprotocol Dataset
  (data [_]))

;; ==================== Models ======================================

(defprotocol DistributionModel
  (params-size [this])
  (mcmc-logpdf [this])
  (logpdf [this])
  (dimension [this])
  (limits [this]))

(defprotocol LikelihoodModel
  (loglik [this]))

(defprotocol PriorModel
  (posterior-model [prior name likelihood]))

(defprotocol ModelProvider
  (model [this]))

(defprotocol ParameterProvider
  (parameters [_]))

;; ==================== Device Model ===============================

(defprotocol DeviceModel
  (dialect [this])
  (source [this])
  (sampler-source [this]))

(defprotocol ModelFactory
  (likelihood-model [this src args])
  (distribution-model [this src args]))

;; ==================== Engines ====================

(defprotocol DatasetEngine
  (data-mean [engine data-matrix])
  (data-variance [engine data-matrix]))

(defprotocol DensityEngine
  (log-density [_ params x])
  (density [_ params x]))

(defprotocol LikelihoodEngine
  (evidence [_ params x]))

(defprotocol EstimateEngine
  (histogram! [engine n])
  (histogram [engine] [engine x]))

(defprotocol AcorEngine
  (acor [this data-matrix]))

;; ==================== Samplers ====================

(defprotocol RandomSampler
  (init! [this seed])
  (sample [this] [this n] [this seed params n])
  (sample! [this] [this n]))

(defprotocol MCMC
  (init-position! [this position] [this seed limits])
  (burn-in! [this n a])
  (anneal! [this schedule n a])
  (acc-rate! [this a])
  (run-sampler! [this n a]))

(defprotocol MCMCStretch
  (init-move! [this cl-means-acc a])
  (move! [this])
  (move-bare! [this])
  (set-temperature! [this t]))

(defrecord Autocorrelation [tau mean sigma ^long steps ^long lag]
  Releaseable
  (release [_]
    (release tau)
    (release mean)
    (release sigma)))

(defprotocol MCMCFactory
  (mcmc-sampler [this walkers params]))

;; ==================== Factories and Providers  ====================

(defprotocol SamplerProvider
  (sampler [_] [_ options]))

(defprotocol EngineProvider
  (engine [_]))

(defprotocol FactoryProvider
  (factory [_]))

(defprotocol EngineFactory;;TODO fuse with samplerfactory?
  (distribution-engine [this model])
  (likelihood-engine [this model])
  (dataset-engine [this]))

(defprotocol SamplerFactory
  (direct-sampler [this model])
  (mcmc-factory [this model])
  (processing-elements [this]))

(defprotocol Library
  (get-source [this id])
  (get-distribution-model [this id])
  (get-likelihood-model [this id])
  (get-distribution-engine [this id])
  (get-likelihood-engine [this id])
  (get-direct-sampler [this id])
  (get-mcmc-factory [this id]))
