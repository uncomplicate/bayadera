(ns uncomplicate.bayadera.core
  (:require [uncomplicate.neanderthal
             [math :refer [sqrt]]
             [core :refer [raw dim alter!]]
             [real :refer [entry]]]
            [uncomplicate.bayadera.protocols :as p]
            [uncomplicate.bayadera.impl :refer :all]))

(defn mean-variance [x]
  (p/mean-variance x))

(defn mean-sd [x]
  (alter! (p/mean-variance x) 1 sqrt))

(defn mean [x]
  (p/mean x))

(defn variance [x]
  (p/variance x))

(defn sd [x]
  (p/sd x))

(defn parameters [dist]
  (p/parameters dist))

(defn sample! [dist result]
  (p/sample! (p/sampler dist) (rand-int Integer/MAX_VALUE) (p/data result)))

(defn sample [dist n]
  (let [result (p/create-dataset dist n)]
    (sample! dist result)
    result))

(defn pdf! [dist xs result]
  (p/pdf! (p/engine dist) xs result))

(defn pdf [dist xs]
  (let [result (raw xs)]
    (pdf! dist xs result)
    result))
