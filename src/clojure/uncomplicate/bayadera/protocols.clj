(ns uncomplicate.bayadera.protocols)

(defprotocol MCMC
  (init! [_] [_ walkers])
  (move! [this])
  (burn-in! [this n])
  (acc-rate [this]))
