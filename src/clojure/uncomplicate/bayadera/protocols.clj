(ns uncomplicate.bayadera.protocols)

(defrecord Autocorrelation [^float tau ^float mean ^float sigma ^long size
                            ^long steps ^long walkers ^long lag ^float acc-rate])

(defprotocol MCMC
  (set-position! [this position])
  (init! [this seed])
  (burn-in! [this n a])
  (run-sampler! [this n a]))

(defprotocol DistributionEngine
  (logpdf! [_ dist x res])
  (pdf! [_ dist x res]))

(defprotocol RandomSampler
  (sample! [this seed dist res]))

(defprotocol Distribution
  (parameters [_]))

(defprotocol Location
  (mean [this])
  (median [this]))

(defprotocol Spread
  (total-range [x])
  (interquartile-range [x])
  (variance [x])
  (sd [x]))

(defprotocol Association
  (cov [x y])
  (corr [x y]))

(defprotocol SamplerProvider
  (sampler [_]))

(defprotocol DistributionEngineProvider
  (distribution-engine [_]))

(defprotocol DistributionEngineFactory
  (vector-factory [_])
  (random-sampler [_ name])
  (dist-engine [_ name]))
