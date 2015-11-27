(ns uncomplicate.bayadera.core
  (:require [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [raw dim alter! create]]
             [real :refer [entry]]]
            [uncomplicate.bayadera.protocols :as p]
            [uncomplicate.bayadera.impl :refer :all]))

(defn dataset [factory n]
  (->UnivariateDataSet (p/dataset-engine factory)
                       (create (np/factory factory) n)))

(defn gaussian [factory ^double mu ^double sigma]
  (->UnivariateDistribution (p/gaussian-engine factory)
                            (p/gaussian-sampler factory)
                            (p/dataset-engine factory)
                            (->GaussianParameters mu sigma)))

(defn uniform [factory ^double a ^double b]
  (->UnivariateDistribution (p/uniform-engine factory)
                            (p/uniform-sampler factory)
                            (p/dataset-engine factory)
                            (->UniformParameters a b)))

(defn mean-variance [x]
  (p/mean-variance (p/measures x)))

(defn mean-sd [x]
  (alter! (p/mean-variance (p/measures x)) 1 sqrt))

(defn mean [x]
  (p/mean (p/measures x)))

(defn variance [x]
  (p/variance (p/measures x)))

(defn sd [x]
  (p/sd (p/measures x)))

(defn parameters [dist]
  (p/parameters dist))

(defn sample! [dist result]
  (p/sample! (p/sampler dist) (rand-int Integer/MAX_VALUE) (p/data result)))

(defn sample [dist n]
  (let [result (p/create-dataset dist n)]
    (sample! dist result)
    result))

;;TODO later
#_(defn pdf! [dist xs result]
  (p/pdf! (p/engine dist) xs result))

#_(defn pdf [dist xs]
  (let [result (raw xs)]
    (pdf! dist xs result)
    result))
