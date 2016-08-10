(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.core
  (:require [uncomplicate.commons.core :refer [release with-release let-release double-fn]]
            [uncomplicate.fluokitten.core :refer [fmap! foldmap]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [transfer create-vector]]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [distributions :refer [beta-log-scale gamma-log-scale t-log-scale
                                    uniform-params gaussian-params t-params
                                    beta-params gamma-params exponential-params]]
             [util :refer [srand-int]]]
            [uncomplicate.bayadera.internal
             [extensions :as extensions]
             [impl :refer :all]]))

(def ^:dynamic *bayadera-factory*)

(defmacro with-bayadera [factory-fn params & body]
  `(binding [*bayadera-factory* (~factory-fn ~@params)]
     (try ~@body
          (finally (release *bayadera-factory*)))))

(defn ^:private compatible [factory x]
  (= (np/factory factory) (np/factory x)))

;; =================== Dataset =================================================

(defn dataset
  ([data-matrix]
   (dataset *bayadera-factory* data-matrix))
  ([factory data-matrix]
   (if (compatible factory data-matrix)
     (->DatasetImpl (p/dataset-engine factory) data-matrix)
     (throw (IllegalArgumentException. (format "Illegal data source: %s." data-matrix))))))

;; =================== Distributions ===========================================

(defn uniform
  ([^double a ^double b]
   (uniform *bayadera-factory* a b))
  ([factory ^double a ^double b]
   (->UniformDistribution factory (p/distribution-engine factory :uniform)
                          (create-vector (np/factory factory)
                                         (uniform-params a b))
                          a b)))

(defn gaussian
  ([^double mu ^double sigma]
   (gaussian *bayadera-factory* mu sigma))
  ([factory ^double mu ^double sigma]
   (->GaussianDistribution factory (p/distribution-engine factory :gaussian)
                           (create-vector (np/factory factory)
                                          (gaussian-params mu sigma))
                           mu sigma)))

(defn t
  ([^double nu ^double mu ^double sigma]
   (t *bayadera-factory* nu mu sigma))
  ([^double nu]
   (t nu 0.0 1.0))
  ([factory ^double nu ^double mu ^double sigma]
   (->TDistribution factory (p/distribution-engine factory :t)
                    (create-vector (np/factory factory) (t-params nu mu sigma))
                    nu mu sigma))
  ([factory ^double nu]
   (t factory nu 0.0 1.0)))

(defn beta
  ([^double a ^double b]
   (beta *bayadera-factory* a b))
  ([factory ^double a ^double b]
   (->BetaDistribution factory (p/distribution-engine factory :beta)
                       (create-vector (np/factory factory) (beta-params a b)) a b)))

(defn gamma
  ([^double theta ^double k]
   (beta *bayadera-factory* theta k))
  ([factory ^double theta ^double k]
   (->BetaDistribution factory (p/distribution-engine factory :gamma)
                       (create-vector (np/factory factory) (gamma-params theta k))
                       theta k)))

(defn exponential
  ([^double lambda]
   (exponential *bayadera-factory* lambda))
  ([factory ^double lambda]
   (->ExponentialDistribution factory (p/distribution-engine factory :exponential)
                              (create-vector (np/factory factory)
                                             (exponential-params lambda))
                              lambda)))

;; ====================== Distribution =========================================

(defn distribution
  ([model]
   (distribution *bayadera-factory* model))
  ([factory model]
   (->DistributionCreator factory (p/distribution-engine factory model)
                          (p/mcmc-factory factory model) model)))

(defn posterior-model
  ([name likelihood prior]
   (p/posterior-model (p/model prior) name likelihood))
  ([likelihood prior]
   (posterior-model (str (gensym "posterior")) likelihood prior)))

(defn posterior
  ([model]
   (posterior *bayadera-factory* model))
  ([factory model]
   (->DistributionCreator factory (p/posterior-engine factory model)
                          (p/mcmc-factory factory model) model))
  ([^String name likelihood prior]
   (posterior *bayadera-factory* name likelihood prior))
  ([factory ^String name likelihood prior]
   (let-release [dist-creator
                 (posterior factory (posterior-model name likelihood prior))]
     (if (satisfies? p/Distribution prior)
       (->PosteriorCreator dist-creator (transfer (p/parameters prior)))
       dist-creator))))

;; ====================== Measures =============================================

(defn mean [x]
  (p/mean x))

#_(defn mode [x]
  (p/mode x))

(defn variance [x]
  (p/variance x))

(defn sd [x]
  (p/sd x))

(defn pdf [dist xs]
  (p/pdf (p/engine dist) (p/parameters dist) (p/data xs)))

(defn log-pdf [dist xs]
  (p/log-pdf (p/engine dist) (p/parameters dist) (p/data xs)))

(defn evidence ^double [dist xs]
  (p/evidence (p/engine dist) (p/parameters dist) (p/data xs)))

;;(defn pdf1 ^double [dist ^double x]
;;  (p/pdf1 dist x))

;; ================= Estimation ===============================================

(defn sampler
  ([dist]
   (p/sampler dist))
  ([dist options]
   (p/sampler dist options)))

(defn sample!
  ([sampler]
   (p/sample! sampler))
  ([sampler n]
   (p/sample! sampler n)))

(defn sample
  ([sampler]
   (p/sample sampler))
  ([sampler n]
   (p/sample sampler n)))

(defn init!
  ([samp seed]
   (p/init! samp seed))
  ([samp]
   (p/init! samp (srand-int))))

(defn histogram!
  ([estimator n]
   (p/histogram! estimator n)))

(defn histogram
  ([estimator]
   (p/histogram estimator)))

;; ============================================================================
