(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.protocols)

(defrecord Autocorrelation [tau mean sigma
                            ^long size ^long steps ^long walkers
                            ^long lag ^float acc-rate])

(defrecord Histogram [limits pdf sorted-bins])

(defprotocol Location
  (mean [x])
  (median [x]))

(defprotocol Spread
  (total-range [x])
  (interquartile-range [x])
  (variance [x])
  (sd [x]))

(defprotocol Association
  (cov [x y])
  (corr [x y]))

(defprotocol DataSet
  (data [_]))

(defprotocol Distribution
  (parameters [_]))

;; ==================== Models ====================

(defprotocol Model
  (params-size [this]))

(defprotocol DistributionModel
  (mcmc-logpdf [this])
  (logpdf [this])
  (dimension [this])
  (lower [this])
  (upper [this]))

(defprotocol LikelihoodModel
  (loglik [this]))

(defprotocol ModelProvider
  (model [this]))

(defprotocol PriorModel
  (posterior-model [prior name likelihood]))

;; ==================== Engines ====================
(defprotocol DatasetEngine
  (means [engine x])
  (variances [engine x])
  (histogram [engine x])
  (sort-data [engine x]))

(defprotocol DistributionEngine
  (log-pdf [_ params x])
  (pdf [_ params x])
  (evidence [_ params x]))

;; ==================== Samplers ====================

(defprotocol RandomSampler
  (init! [this seed])
  (sample! [this n] [this seed params n]))

(defprotocol MCMC
  (set-position! [this position])
  (burn-in! [this n a])
  (run-sampler! [this n a]))

(defprotocol MCMCStretch
  (move! [this])
  (move-bare! [this])
  (acc-rate [this])
  (acor [this sample]))

(defprotocol MCMCFactory
  (mcmc-sampler [this walkers params low high]))

;; ==================== Factories and Providers  ====================

(defprotocol SamplerProvider
  (sampler [_] [_ options]))

(defprotocol EngineProvider
  (engine [_]))

(defprotocol DistributionEngineFactory
  (gaussian-engine [this])
  (uniform-engine [this])
  (binomial-engine [this])
  (beta-engine [this])
  (distribution-engine [this model])
  (posterior-engine [this model]))

(defprotocol SamplerFactory
  (gaussian-sampler [this])
  (uniform-sampler [this])
  (binomial-sampler [this])
  (beta-sampler [this])
  (mcmc-factory [this model])
  (processing-elements [this]))

(defprotocol DataSetFactory
  (dataset-engine [this]))
