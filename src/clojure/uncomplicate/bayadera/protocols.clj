(ns uncomplicate.bayadera.protocols)

(defrecord Autocorrelation [^float tau ^float mean ^float sigma ^long size
                            ^long steps ^long walkers ^long lag ^float acc-rate])

(defprotocol Location
  (mean [this])
  (median [this]))

(defprotocol Spread
  (total-range [x])
  (interquartile-range [x])
  (variance [x] [engine x])
  (sd [x]))

(defprotocol Association
  (cov [x y])
  (corr [x y]))

(defprotocol Distribution
  (parameters [_]))

(defprotocol DataSet
  (data [_]))

(defprotocol DataSetCreator
  (create-dataset [_ n]))

(defprotocol DistributionEngine
  (logpdf! [_ dist x res])
  (pdf! [_ dist x res]))

(defprotocol RandomSampler
  (sample!
    [this seed params res]
    [this seed params]))

(defprotocol MCMC
  (set-position! [this position])
  (init! [this seed])
  (burn-in! [this n a])
  (run-sampler! [this n a]))

(defprotocol SamplerProvider
  (sampler [_]))

(defprotocol EngineProvider
  (engine [_]))

(defprotocol EngineFactory
  (random-sampler [_ name])
  (distribution-engine [_ name])
  (dataset-engine [_ data]))
