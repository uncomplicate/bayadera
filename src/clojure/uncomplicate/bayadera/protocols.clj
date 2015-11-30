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
  (variance [x])
  (sd [x]))

(defprotocol Association
  (cov [x y])
  (corr [x y]))

(defprotocol Distribution
  (parameters [_]))

(defprotocol DataSet
  (data [_]))

(defprotocol DistributionEngine
  (logpdf! [_ params x res])
  (pdf! [_ params x res]))

(defprotocol RandomSampler
  (sample! [this seed params res] [this seed params]))

(defprotocol MeasureProvider
  (measures [this]))

(defprotocol MCMC
  (set-position! [this position])
  (init! [this seed])
  (burn-in! [this n a])
  (run-sampler! [this n a]))

(defprotocol SamplerProvider
  (sampler [_]))

(defprotocol EngineProvider
  (engine [_]))

(defprotocol FactoryProvider
  (factory [_]))

(defprotocol DistributionEngineFactory
  (gaussian-engine [this])
  (uniform-engine [this]))

(defprotocol SamplerFactory
  (gaussian-sampler [this])
  (uniform-sampler [this]))

(defprotocol DataSetFactory
  (dataset-engine [this]))
