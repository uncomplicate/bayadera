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

;;TODO This should probably go to opencl.clj, similarly to neanderthal
(defn distribution [factory model]
  (if (= 1 (:dimension model))
    (->UnivariateDistributionCreator factory
                                     (p/custom-engine factory model)
                                     (p/mcmc-factory factory model)
                                     model)
    (throw (UnsupportedOperationException. "TODO"))))

(defn posterior [likelihood prior]
  (p/posterior prior likelihood))

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

(defn sample [sampler n-or-result]
  (p/sample! sampler n-or-result))

(defn pdf! [dist xs result]
  (p/pdf! (p/engine dist) (p/parameters dist) (p/data xs) result))

(defn pdf [dist xs]
  (let [result (raw (p/data xs))];;TODO This works only for univariate xs
    (pdf! dist xs result)
    result))

;;(defn pdf* ^double [dist ^double x]
;;  (p/pdf* dist x))
