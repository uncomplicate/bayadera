;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.core
  (:require [uncomplicate.commons
             [core :refer [release with-release let-release info]]
             [utils :refer [dragan-says-ex cond-into]]]
            [uncomplicate.neanderthal
             [core :refer [vctr native! matrix-type compatible?]]
             [block :refer [column?]]]
            [uncomplicate.bayadera.util :refer [srand-int]]
            [uncomplicate.bayadera.internal
             [protocols :as p]
             [extensions]
             [impl :refer [->DatasetImpl ->LikelihoodImpl ->DistributionCreator posterior-creator]]]))

(def ^:dynamic *bayadera-factory*)

(defmacro with-bayadera [factory-fn params & body]
  `(binding [*bayadera-factory* (~factory-fn ~@params)]
     (try ~@body
          (finally (release *bayadera-factory*)))))

;; =================== Dataset =================================================

(defn dataset
  ([data-matrix]
   (dataset *bayadera-factory* data-matrix))
  ([factory data-matrix]
   (if (and (compatible? factory data-matrix)
            (column? data-matrix) (= :ge (matrix-type data-matrix)))
     (->DatasetImpl (p/dataset-engine factory) data-matrix)
     (dragan-says-ex "The engine can not support this data."
                     {:matrix-type (matrix-type data-matrix) :data (info data-matrix)
                      :factory (info factory) :errors
                      (cond-into []
                                 (not (= :ge (matrix-type data-matrix))) "matrix type is not :ge"
                                 (not (compatible? factory data-matrix))
                                 "data is not compatible with factory"
                                 (not (column? data-matrix)) "data matrix is not column-major")}))))

;; ====================== Distribution =========================================

(defn distribution-model
  ([src]
   (distribution-model *bayadera-factory* src nil))
  ([src args]
   (distribution-model *bayadera-factory* src args))
  ([factory src args]
   (p/distribution-model factory src args)))

(defn likelihood-model
  ([src]
   (likelihood-model *bayadera-factory* src nil))
  ([src args]
   (likelihood-model *bayadera-factory* src args))
  ([factory src args]
   (p/likelihood-model factory src args)))

(defn posterior-model
  ([name likelihood prior]
   (if (compatible? (p/model likelihood) (p/model prior))
     (p/posterior-model (p/model prior) name (p/model likelihood))
     (dragan-says-ex (format "Incompatible types of likelihood and prior models."
                             {:likelihood-type (type likelihood) :prior-type (type prior)}))))
  ([likelihood prior]
   (posterior-model (str (gensym "posterior")) likelihood prior)))

(defn likelihood
  ([model]
   (likelihood *bayadera-factory* model))
  ([factory model-provider]
   (if (compatible? factory model-provider)
     (let [model (p/model model-provider)]
       (->LikelihoodImpl factory (p/likelihood-engine factory model) model))
     (dragan-says-ex (format "Model dialect is incompatible with factory."
                             {:type (type model-provider) :factory (type factory)})))))

(defn distribution
  ([model-provider]
   (distribution *bayadera-factory* model-provider))
  ([factory model-provider]
   (if (compatible? factory model-provider)
     (let [model (p/model model-provider)]
       (->DistributionCreator factory (p/distribution-engine factory model)
                              (p/mcmc-factory factory model) model))
     (dragan-says-ex (format "Model dialect is incompatible with factory."
                             {:type (type model-provider) :factory (type factory)}))))
  ([factory-or-name likelihood prior]
   (if (string? factory-or-name)
     (distribution *bayadera-factory* factory-or-name likelihood prior)
     (distribution factory-or-name (str (gensym "posterior")) likelihood prior)))
  ([factory ^String name likelihood prior]
   (let [model (posterior-model name likelihood prior)]
     (if (satisfies? p/ParameterProvider prior)
       (posterior-creator factory model (p/parameters prior))
       (distribution factory model)))))

;; ====================== Measures =============================================

(defn mean [x]
  (native! (p/mean x)))

(defn mode [x]
  (native! (p/mode x)))

(defn median [x]
  (native! (p/median x)))

(defn variance [x]
  (native! (p/variance x)))

(defn sd [x]
  (native! (p/sd x)))

(defn density [d xs]
  (if (compatible? d (p/data xs))
    (p/density (p/engine d) (p/parameters d) (p/data xs))
    (dragan-says-ex (format "Data type is incompatible with the engine."
                            {:type (type d) :xs (info (p/data xs))}))))

(defn log-density [d xs]
  (if (compatible? d (p/data xs))
    (p/log-density (p/engine d) (p/parameters d) (p/data xs))
    (dragan-says-ex (format  "Data type is incompatible with the engine."
                            {:type (type d) :xs (info (p/data xs))}))))

(defn evidence ^double [lik data prior-sample]
  (if (and (compatible? lik data) (compatible? lik (p/data prior-sample)) )
    (p/evidence (p/engine lik) data (p/data prior-sample))
    (dragan-says-ex (format "Data type is incompatible with likelihood engine"
                            {:type (type lik) :prior-sample (info (p/data prior-sample))}))))

;; ================= Estimation ===============================================

(defn sampler
  ([dist]
   (p/sampler dist))
  ([dist options]
   (p/sampler dist options)))

(defn sample!
  ([sampler]
   (p/sample! sampler))
  ([sampler ^long n]
   (p/sample! sampler n)))

(defn sample
  ([sampler]
   (p/sample sampler))
  ([sampler ^long n]
   (p/sample sampler n)))

(defn init!
  ([samp ^long seed]
   (p/init! samp seed))
  ([samp]
   (p/init! samp (srand-int))))

(defn histogram!
  ([estimator ^long n]
   (p/histogram! estimator n)))

(defn histogram
  ([estimator]
   (p/histogram estimator)))

;; ============================================================================
