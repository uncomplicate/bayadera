(ns uncomplicate.bayadera.core
  (:require [uncomplicate.clojurecl.core :refer [release]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [dim alter! create-vector vect? raw scal!]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [impl :refer :all]
             [math :refer [log-beta]]]))

(def ^:dynamic *bayadera-factory*)

(defmacro with-bayadera
  [factory-fn params & body]
  `(binding [*bayadera-factory*
             (~factory-fn ~@params)]
     (try ~@body
          (finally (release *bayadera-factory*)))))

;; =============================================================================

(defn dataset
  ([src]
   (dataset *bayadera-factory* src))
  ([factory src]
   (->UnivariateDataSet (p/dataset-engine factory)
                        (cond (number? src)
                              (create-vector (np/factory factory) src)
                              (vect? src) src))))

;; =============================================================================

(defn gaussian-params [mu sigma]
  [mu sigma])

(defn gaussian
  ([^double mu ^double sigma]
   (gaussian *bayadera-factory* mu sigma))
  ([factory ^double mu ^double sigma]
   (->GaussianDistribution
    factory (p/gaussian-engine factory)
    (create-vector (np/factory factory) (gaussian-params mu sigma))
    mu sigma)))

(defn uniform-params [a b]
  [a b])

(defn uniform
  ([^double a ^double b]
   (uniform *bayadera-factory* a b))
  ([factory ^double a ^double b]
   (->UniformDistribution
    factory (p/uniform-engine factory)
    (create-vector (np/factory factory) (uniform-params a b))
    a b)))

(defn beta-params [a b]
  [a b (log-beta a b)])

(defn beta
  ([^double a ^double b]
   (beta *bayadera-factory* a b))
  ([factory ^double a ^double b]
   (->BetaDistribution
    factory (p/beta-engine factory)
    (create-vector (np/factory factory) (beta-params a b))
    a b)))

(defn binomial-lik-params [n k]
  [n k])

;; =============================================================================

(defn distribution
  ([model]
   (distribution *bayadera-factory* model))
  ([factory model]
   (if (= 1 (p/dimension model))
     (univariate-distribution-creator factory model)
     (throw (UnsupportedOperationException. "TODO")))))

(defn posterior-model
  ([name likelihood prior]
   (p/posterior-model name likelihood prior))
  ([likelihood prior]
   (p/posterior-model "posterior" likelihood prior)))

(defn posterior
  ([model]
   (p/posterior *bayadera-factory* model))
  ([factory model]
   (p/posterior factory model))
  ([^String name likelihood prior]
   (p/posterior *bayadera-factory* name likelihood prior))
  ([factory ^String name likelihood prior]
   (p/posterior factory name likelihood prior)))

;; =============================================================================

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

(defn evidence [dist xs]
  (p/evidence (p/engine dist) (p/parameters dist) (p/data xs)))

;;(defn pdf* ^double [dist ^double x]
;;  (p/pdf* dist x))
