(ns uncomplicate.bayadera.core
  (:require [uncomplicate.clojurecl.toolbox :refer [wrap-int wrap-float]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [raw dim alter! create transfer! vect? raw]]
             [native :refer [sv]]
             [real :refer [entry]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [impl :refer :all]
             [math :refer [log-beta]]]))

(defn dataset [factory src]
  (->UnivariateDataSet (p/dataset-engine factory)
                       (cond (number? src)
                             (create (np/factory factory) src)
                             (vect? src) src)))

(defn gaussian [factory ^double mu ^double sigma]
  (let [params (transfer! [mu sigma] (create (np/factory factory) 2))]
    (->GaussianDistribution factory (p/gaussian-engine factory) params mu sigma)))

(defn uniform [factory ^double a ^double b]
  (let [params (transfer! [a b] (create (np/factory factory) 2))]
    (->UniformDistribution factory (p/uniform-engine factory) params a b)))

(defn beta [factory ^double a ^double b]
  (let [params (transfer! [a b (log-beta a b)] (create (np/factory factory) 3))]
    (->BetaDistribution factory (p/beta-engine factory) params a b)))

(defn udist [factory model]
  (->UnivariateDistributionCreator factory
                                   (p/custom-engine factory model)
                                   (p/mcmc-sampler factory model)
                                   model))

(defn mean-variance [x]
  (p/mean-variance x))

(defn mean-sd [x]
  (alter! (p/mean-variance x) 1 sqrt))

(defn mean [x]
  (p/mean x))

(defn variance [x]
  (p/variance x))

(defn sd [x]
  (sqrt (variance x)))

(defn sampler [dist]
  (p/sampler dist))

(defn sample [sampler n]
  (p/sample! sampler n))

;;TODO Rename this to density and pmf to mass, and use pdf and pmf to acquire the functions themselves
(defn pdf! [dist xs result]
  (p/pdf! (p/engine dist) (p/parameters dist) (p/data xs) result))

(defn pdf [dist xs]
  (let [result (raw (p/data xs))]
    (pdf! dist xs result)
    result))
