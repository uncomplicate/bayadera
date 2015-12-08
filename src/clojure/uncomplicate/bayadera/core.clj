(ns uncomplicate.bayadera.core
  (:require [uncomplicate.clojurecl.toolbox :refer [wrap-int wrap-float]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [raw dim alter! create transfer! vect? subvector copy!]]
             [native :refer [sv]]
             [real :refer [entry]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [impl :refer :all]
             [special :refer [lnbeta]]]))

(defn dataset [factory src]
  (->UnivariateDataSet (np/factory factory)
                       (p/dataset-engine factory)
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
  (let [params (transfer! [a b (lnbeta a b)] (create (np/factory factory) 3))]
    (->BetaDistribution factory (p/beta-engine factory) params a b)))

(defn univariate [factory model data hyperparams];;Use GPU params instead of host later
  (let [params-size (+ (dim data) (dim hyperparams))
        params (sv params-size)]
        (do
          (copy! data (subvector params 0 (dim data)))
          (copy! hyperparams (subvector params (dim data) (dim hyperparams)))
          (->UnivariateDistribution factory (p/custom-engine factory model)
                                    params params model))))

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

(defn pdf! [dist xs result]
  (p/pdf! (p/engine dist) (p/parameters dist) (p/data xs) result))

(defn pdf [dist xs]
  (let [result (p/raw-result xs)]
    (pdf! dist xs result)
    result))
