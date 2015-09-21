(ns uncomplicate.bayadera.core
  (:require [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [zero dim]]]
            [uncomplicate.bayadera.protocols :as p]
            [uncomplicate.bayadera.impl :refer :all]))

(defn mean [x]
  (p/mean x))

(defn sd [x]
  (p/sd x))

(defn sample! [dist result]
  (p/sample! (p/sampler dist) (rand-int Integer/MAX_VALUE) (p/parameters dist) (p/data result)))

(defn sample [dist n]
  (let [result (p/create-dataset dist n)]
    (sample! dist result)
    result))

(defn pdf! [dist xs result]
  (p/pdf! (p/engine dist) (p/parameters dist) xs result))

(defn pdf [dist xs]
  (let [result (zero xs)]
    (pdf! dist xs result)
    result))
