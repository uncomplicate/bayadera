(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.models-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.bayadera.internal.device.models :refer :all]))

(facts
 "Test source library"
 (let [source-lib (source-library "uncomplicate/bayadera/internal/opencl/%s.cl")]
   (apply hash-set (keys source-lib)) => #{:posterior :distribution :uniform :gaussian :student-t
                                           :beta :exponential :erlang :gamma :binomial
                                           :uniform-sampler :gaussian-sampler :exponential-sampler
                                           :erlang-sampler}
   (string? (deref (source-lib :gaussian))) => true))
