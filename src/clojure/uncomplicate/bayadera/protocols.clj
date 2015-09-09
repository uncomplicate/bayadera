(ns uncomplicate.bayadera.protocols)

(defrecord Autocorrelation [^float tau ^float mean ^float sigma ^long size
                            ^long steps ^long walkers ^long lag ^float acc-rate])

(defprotocol MCMC
  (set-position! [this position])
  (init! [this seed])
  (burn-in! [this n a])
  (run-sampler! [this n a]))

(defprotocol DistributionEngine
  (logpdf! [_ n params x res])
  (pdf! [_ n params x res]))

(defprotocol RandomSampler
  (sample! [this seed res] [this seed n params res]))
