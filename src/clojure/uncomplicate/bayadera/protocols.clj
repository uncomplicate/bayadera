(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.protocols)

(defrecord Autocorrelation [tau mean sigma
                            ^long size ^long steps ^long walkers
                            ^long lag ^float acc-rate])

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

(defprotocol ModelProvider
  (model [this]))

(defmulti posterior-model (fn [name likelihood prior]
                            [(class likelihood) (class prior)]))


(defmulti posterior (fn
                      ([factory model]
                       [(dimension model) (class model)])
                      ([factory name likelihood prior]
                       [(dimension (model prior)) (class likelihood) (class prior)])))

;; ==================== Engines ====================

(defprotocol DistributionEngine
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
