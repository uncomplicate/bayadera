(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.core
  (:require [uncomplicate.commons.core :refer [release with-release let-release double-fn]]
            [uncomplicate.fluokitten.core :refer [fmap! foldmap]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [transfer]]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [impl :refer :all]
             [math :refer [log-beta]]]))

(def ^:dynamic *bayadera-factory*)

(defmacro with-bayadera [factory-fn params & body]
  `(binding [*bayadera-factory* (~factory-fn ~@params)]
     (try ~@body
          (finally (release *bayadera-factory*)))))

;; =============================================================================

(defn dataset
  ([data-matrix]
   (dataset *bayadera-factory* data-matrix))
  ([factory data-matrix]
   (if (= (np/factory factory) (np/factory data-matrix));;TODO check compatibility of data-matrix factory and dataset factory in a separate method
     (->DataSetImpl (p/dataset-engine factory) data-matrix)
     (throw (IllegalArgumentException. (format "Illegal data source: %s." data-matrix))))))

;; =============================================================================

(defn gaussian-params [^double mu ^double sigma]
  (sv mu sigma))

(defn gaussian
  ([^double mu ^double sigma]
   (gaussian *bayadera-factory* mu sigma))
  ([factory ^double mu ^double sigma]
   (with-release [params (gaussian-params mu sigma)]
     (->GaussianDistribution factory (p/gaussian-engine factory)
                             (transfer (np/factory factory) params)
                              mu sigma))))

(defn uniform-params [^double a ^double b]
  (sv a b))

(defn uniform
  ([^double a ^double b]
   (uniform *bayadera-factory* a b))
  ([factory ^double a ^double b]
   (with-release [params (uniform-params a b)]
     (->UniformDistribution factory (p/uniform-engine factory)
                            (transfer (np/factory factory) params) a b))))

(defn beta-params [^double a ^double b]
  (sv a b (log-beta a b)))

(defn beta
  ([^double a ^double b]
   (beta *bayadera-factory* a b))
  ([factory ^double a ^double b]
   (with-release [params (beta-params a b)]
     (->BetaDistribution factory (p/beta-engine factory)
                         (transfer (np/factory factory) params) a b))))

(defn binomial-lik-params [^double n ^double k]
  (sv n k))

;; =============================================================================

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

;; =============================================================================

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

(defn evidence [dist xs]
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

(defn histogram!
  ([estimator n]
   (p/histogram! estimator n)))

(defn histogram
  ([estimator]
   (p/histogram estimator)))

;; ============================================================================
