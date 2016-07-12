(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.protocols)

(defrecord Histogram [limits pdf bin-ranks])

(defprotocol Location
  (mean [x])
  (median [x])
  (mode [x]))

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

(defprotocol ModelProvider
  (model [this]))

(defprotocol PriorModel
  (posterior-model [prior name likelihood]))

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

(defrecord Autocorrelation [tau mean sigma ^long steps ^long lag])

(defprotocol MCMCFactory
  (mcmc-sampler [this walkers params]))

;; ==================== Factories and Providers  ====================

(defprotocol SamplerProvider
  (sampler [_] [_ options]))

(defprotocol EngineProvider
  (engine [_]))

(defprotocol DistributionEngineFactory
  (uniform-engine [this])
  (binomial-engine [this])
  (gaussian-engine [this])
  (t-engine [this])
  (beta-engine [this])
  (gamma-engine [this])
  (exponential-engine [this])
  (distribution-engine [this model])
  (posterior-engine [this model]))

(defprotocol SamplerFactory
  (uniform-sampler [this])
  (binomial-sampler [this])
  (gaussian-sampler [this])
  (t-sampler [this])
  (beta-sampler [this])
  (gamma-sampler [this])
  (exponential-sampler [this])
  (mcmc-factory [this model])
  (processing-elements [this]))

(defprotocol DataSetFactory
  (dataset-engine [this]))
