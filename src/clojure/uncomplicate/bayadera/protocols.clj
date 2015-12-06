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
  (raw-result [_])
  (data-count [_]))

(defprotocol Distribution
  (parameters [_]))

(defprotocol DistributionEngine
  (logpdf! [_ params x res])
  (pdf! [_ params x res]))

(defprotocol RandomSampler
  (init! [this seed])
  (sample! [this n] [this seed params res]))

(defprotocol MCMCEngineFactory
  (mcmc-engine [this walker-count cl-params low high]))

(defprotocol MCMC
  (set-position! [this position])
  (burn-in! [this n a])
  (run-sampler! [this n a]))

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
  (mcmc-sampler [this model]))

(defprotocol DataSetFactory
  (dataset-engine [this]))
