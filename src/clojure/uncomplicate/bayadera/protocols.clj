(ns uncomplicate.bayadera.protocols)

(defrecord Autocorrelation [^float tau ^float mean ^float sigma ^long size
                            ^long steps ^long walkers ^long lag ^float acc-rate])


(defprotocol Location
  (mean [this])
  (median [this]))

(defprotocol Spread
  (total-range [x])
  (interquartile-range [x])
  (mean-variance [x] [eng data])
  (variance [x]))

(defprotocol Association
  (cov [x y])
  (corr [x y]))

(defprotocol DataSet
  (data [_])
  (data-count [_]))

(defprotocol Distribution
  (parameters [_]))

;; ==================== Models ====================

(defprotocol DistributionModel
  (mcmc-logpdf [this])
  (logpdf [this])
  (dimension [this])
  (lower [this])
  (upper [this]))

(defprotocol LikelihoodModel
  (loglik [this]))

(defprotocol CLModel
  (params-size [this])
  (source [this])
  (sampler-source [this]))

(defprotocol PriorModel
  (posterior [prior likelihood model-name]))

;; ==================== Engines ====================

(defprotocol DistributionEngine
  (model [_])
  (logpdf! [_ params x res])
  (pdf! [_ params x res])
  (evidence [_ params x]))

;; ==================== Samplers ====================

(defprotocol RandomSampler
  (init! [this seed])
  (sample! [this res] [this params res] [this seed params res]))

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
  (mcmc-sampler [this walker-count cl-params low high]))

;; ==================== Factories and Providers  ====================

(defprotocol SamplerProvider
  (sampler [_]))

(defprotocol EngineProvider
  (engine [_]))

(defprotocol DistributionEngineFactory
  (gaussian-engine [this])
  (uniform-engine [this])
  (binomial-engine [this])
  (beta-engine [this])
  (custom-engine [this model]))

(defprotocol SamplerFactory
  (gaussian-sampler [this])
  (uniform-sampler [this])
  (binomial-sampler [this])
  (beta-sampler [this])
  (mcmc-factory [this model]))

(defprotocol DataSetFactory
  (dataset-engine [this]))
