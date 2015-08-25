(ns uncomplicate.bayadera.protocols)

(defprotocol MCMC
  (init! [_] [_ walkers])
  (move! [this])
  (run! [this n])
  (acc-rate [this])
  (acor [this]))
