(ns uncomplicate.bayadera.core
  (:require [uncomplicate.clojurecl.core :refer [release]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [raw dim alter! create transfer! vect? raw scal!]]]
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

(defn dataset
  ([src]
   (dataset *bayadera-factory* src))
  ([factory src]
   (->UnivariateDataSet (p/dataset-engine factory)
                        (cond (number? src)
                              (create (np/factory factory) src)
                              (vect? src) src))))

(defn gaussian
  ([^double mu ^double sigma]
   (gaussian *bayadera-factory* mu sigma))
  ([factory ^double mu ^double sigma]
   (let [params (transfer! [mu sigma] (create (np/factory factory) 2))]
     (->GaussianDistribution factory (p/gaussian-engine factory) params mu sigma))))

(defn uniform
  ([^double a ^double b]
   (uniform *bayadera-factory* a b))
  ([factory ^double a ^double b]
   (let [params (transfer! [a b] (create (np/factory factory) 2))]
     (->UniformDistribution factory (p/uniform-engine factory) params a b))))

(defn beta
  ([^double a ^double b]
   (beta *bayadera-factory* a b))
  ([factory ^double a ^double b]
   (let [params (transfer! [a b (log-beta a b)] (create (np/factory factory) 3))]
     (->BetaDistribution factory (p/beta-engine factory) params a b))))

;;TODO This should probably go to opencl.clj, similarly to neanderthal
(defn distribution
  ([model]
   (distribution *bayadera-factory* model))
  ([factory model]
   (if (= 1 (p/dimension model))
     (->UnivariateDistributionCreator factory
                                      (p/distribution-engine factory model)
                                      (p/mcmc-factory factory model)
                                      model)
     (throw (UnsupportedOperationException. "TODO")))))

(defn posterior-model
  ([likelihood prior name]
   (p/posterior prior likelihood name))
  ([likelihood prior]
   (posterior-model likelihood prior "posterior")))

(defn posterior
  ([model]
   (posterior *bayadera-factory* model))
  ([factory model]
   (if (= 1 (p/dimension model))
     (->UnivariateDistributionCreator factory
                                      (p/posterior-engine factory model)
                                      (p/mcmc-factory factory model)
                                      model)
     (throw (UnsupportedOperationException. "TODO"))))
  ([likelihood prior name]
   (posterior *bayadera-factory* likelihood prior name))
  ([factory likelihood prior ^String name]
   (let [post-model (p/posterior (p/model prior) likelihood name)]
     (posterior factory post-model))))

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
