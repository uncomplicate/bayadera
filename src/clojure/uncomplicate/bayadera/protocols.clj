(ns uncomplicate.bayadera.protocols)

(defrecord Autocorrelation [^float tau ^float mean ^float sigma ^long size
                            ^long steps ^long walkers ^long lag ^float acc-rate])

(defprotocol MCMC
  (init-walkers! [this seed] [this seed cl-walkers])
  (init! [this] [this seed] [this seed walkers])
  (burn-in! [this n a])
  (reset-counters! [_])
  (move! [this])
  (move-bare! [this])
  (run-sampler! [_ n a])
  (acc-rate [_])
  (acor [_ sample]))
