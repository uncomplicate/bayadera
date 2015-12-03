(ns uncomplicate.bayadera.core
  (:require [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [raw dim alter! create transfer!]]
             [real :refer [entry]]]
            [uncomplicate.bayadera.protocols :as p]
            [uncomplicate.bayadera.impl :refer :all]))

(defn dataset [factory n]
  (->UnivariateDataSet (p/dataset-engine factory)
                       (create (np/factory factory) n)))

(defn gaussian [factory ^double mu ^double sigma]
  (->UnivariateDistribution factory
                            (p/gaussian-engine factory)
                            (p/gaussian-sampler factory)
                            (->GaussianMeasures mu sigma)
                            (transfer! [mu sigma] (create (np/factory factory) 2))))

(defn uniform [factory ^double a ^double b]
  (->UnivariateDistribution factory
                            (p/uniform-engine factory)
                            (p/uniform-sampler factory)
                            (->UniformMeasures a b)
                            (transfer! [a b] (create (np/factory factory) 2))))

(defn build-model [factory model]
  (let [engine (p/custom-engine factory model)
        sampler (p/mcmc-sampler factory model)]
    (fn [params]
      (->UnivariateDistribution factory
                                engine sampler nil
                                (transfer! params (create (np/factory factory) (dim params)))))))

#_(defn beta [factory ^double a ^double b]
  (let [params (transfer! [a b] (create (np/factory factory) 2))]
    (->UnivariateDistribution factory
                              (p/beta-engine factory)
                              (p/mcmc-engine (p/beta-sampler factory) params)
                              (->BetaMeasures a b)
                              params)))

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

#_(defn mcmc-sampler [dist options]
  (p/mcmc-engine (p/mcmc-factory dist) options))

(defn sample! [dist result]
  (p/sample! (p/sampler dist) (rand-int Integer/MAX_VALUE) (p/parameters dist) (p/data result)))

#_(defn sample! [sampler result]
  (p/sample! sampler (p/data result)))

(defn sample [dist n]
  (let [result (dataset (p/factory dist) n)]
    (sample! dist result)
    result))

;;TODO later
#_(defn pdf! [dist xs result]
  (p/pdf! (p/engine dist) xs result))

#_(defn pdf [dist xs]
  (let [result (raw xs)]
    (pdf! dist xs result)
    result))
