(ns uncomplicate.bayadera.protocols)

(defprotocol MCMC
  (init! [_] [_ walkers])
  (reset-counters! [_])
  (move! [this])
  (run-sampler! [this n])
  (acc-rate [this])
  (acor [this]))
