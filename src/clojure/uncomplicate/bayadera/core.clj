(ns uncomplicate.bayadera.core
  (:require [uncomplicate.neanderthal
             [protocols :as np]
             [core :refer [zero dim]]]
            [uncomplicate.bayadera.protocols :as p]
            [uncomplicate.bayadera.distributions.opencl.generic :refer :all]))

(defn mean [x]
  (p/mean x))

(defn sd [x]
  (p/sd x))

(defn sample! [dist result]
  (p/sample! (p/sampler dist) (rand-int Integer/MAX_VALUE) (p/parameters dist) result))

(defn sample [dist n]
  (let [result (np/create-block dist n)]
    (sample! dist result)
    result))

(defn pdf! [dist xs result]
  (p/pdf! (p/distribution-engine dist) (p/parameters dist) xs result))

(defn pdf [dist xs]
  (let [result (zero xs)]
    (pdf! dist xs result)
    result))
