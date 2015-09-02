(ns uncomplicate.bayadera.protocols)

(defrecord Autocorrelation [^float time ^float mean ^float std ^long size d])

(defprotocol MCMC
  (init! [this] [this seed])
  (reset-counters! [_])
  (move! [this])
  (run-sampler! [_ n])
  (acc-rate [_])
  (acor [_ sample]))
