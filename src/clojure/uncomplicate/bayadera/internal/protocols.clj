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

(defprotocol Distribution
  (parameters [_]))

;; ==================== Models ======================================

(defprotocol Model
  (params-size [this]))

(defprotocol DistributionModel
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

;; ==================== Engines ====================

(defprotocol DatasetEngine
  (data-mean [engine data-matrix])
  (data-variance [engine data-matrix]))

(defprotocol DistributionEngine
  (log-pdf [_ params x])
  (pdf [_ params x])
  (evidence [_ params x]))

(defprotocol EstimateEngine
  (histogram! [engine n])
  (histogram [engine] [engine x]))

;; ==================== Samplers ====================

(defprotocol RandomSampler
  (init! [this seed])
  (sample [this] [this n] [this seed params n])
  (sample! [this] [this n]))

(defprotocol MCMC
  (info [this])
  (init-position! [this position] [this seed limits])
  (burn-in! [this n a])
  (anneal! [this schedule n a])
  (acc-rate! [this a])
  (run-sampler! [this n a]))

(defprotocol MCMCStretch
  (init-move! [this cl-means-acc a])
  (move! [this])
  (move-bare! [this])
  (set-temperature! [this t])
  (acor [this sample]))

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

(defprotocol DistributionEngineFactory
  (distribution-engine [this model])
  (posterior-engine [this model]))

(defprotocol SamplerFactory
  (direct-sampler [this id])
  (mcmc-factory [this model])
  (processing-elements [this]))

(defprotocol DatasetFactory
  (dataset-engine [this]))
