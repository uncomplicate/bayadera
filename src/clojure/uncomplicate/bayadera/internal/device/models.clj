;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.device.models
  (:require [clojure.java.io :as io]
            [uncomplicate.commons
             [core :refer [Releaseable release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [fmap op]]
            [uncomplicate.neanderthal.internal.api :as na]
            [uncomplicate.neanderthal
             [core :refer [copy]]
             [native :refer [fge]]]
            [uncomplicate.bayadera.internal.protocols :refer :all]
            [uncomplicate.bayadera.internal.device.util :refer [release-deref]]))

;; ==================== Likelihood model ====================================

(deftype DeviceLikelihoodModel [model-dialect name loglik-name model-source]
  Releaseable
  (release [_]
    true)
  na/MemoryContext
  (compatible? [_ o]
    (and (satisfies? DeviceModel o) (= model-dialect (dialect o))))
  DeviceModel
  (dialect [_]
    model-dialect)
  (source [_]
    model-source)
  (sampler-source [_]
    nil)
  LikelihoodModel
  (loglik [_]
    loglik-name)
  ModelProvider
  (model [this]
    this))

(defn device-likelihood-model [source args]
  (let [{:keys [dialect name loglik]
         :or {dialect :c99
              name (str (gensym "likelihood"))
              loglik (format "%s_loglik" name)}} args]
    (->DeviceLikelihoodModel dialect name loglik (if (sequential? source) source [source]))))

;; ==================== Distribution model ====================================

(declare device-posterior-model)

(deftype DeviceDistributionModel [model-dialect post-template name logpdf-name mcmc-logpdf-name
                                  ^long dist-dimension ^long dist-params-size
                                  model-limits model-source sampler-src]
  Releaseable
  (release [_]
    (release model-limits))
  na/MemoryContext
  (compatible? [_ o]
    (and (satisfies? DeviceModel o) (= model-dialect (dialect o))))
  DistributionModel
  (params-size [_]
    dist-params-size)
  (logpdf [_]
    logpdf-name)
  (mcmc-logpdf [_]
    mcmc-logpdf-name)
  (dimension [_]
    dist-dimension)
  (limits [_]
    model-limits)
  PriorModel
  (posterior-model [prior name likelihood]
    (device-posterior-model post-template model-dialect prior name likelihood))
  DeviceModel
  (dialect [_]
    model-dialect)
  (source [_]
    model-source)
  (sampler-source [_]
    sampler-src)
  ModelProvider
  (model [this]
    this))

(defn device-distribution-model
  [post-template source args]
  (let [{:keys [dialect name logpdf mcmc-logpdf dimension params-size
                limits sampler-source]
         :or {dialect :c99
              name (str (gensym "distribution"))
              logpdf (format "%s_logpdf" name)
              mcmc-logpdf logpdf dimension 1 params-size 1}} args]
    (->DeviceDistributionModel dialect post-template name logpdf mcmc-logpdf dimension params-size
                               limits (if (sequential? source) source [source])
                               (if (sequential? sampler-source) sampler-source [sampler-source]))))

(defn device-posterior-model [post-template dialect prior name lik]
  (let [post-name (str (gensym name))
        post-logpdf (format "%s_logpdf" post-name)
        post-mcmc-logpdf (format "%s_mcmc_logpdf" post-name)]
    (->DeviceDistributionModel dialect post-template post-name post-logpdf post-mcmc-logpdf
                               (dimension prior) (params-size prior)
                               (when (limits prior) (copy (limits prior)))
                               (conj (vec (distinct (into (source prior) (source lik))))
                                     (format "%s\n%s"
                                             (format post-template post-logpdf
                                                     (loglik lik) (logpdf prior))
                                             (format post-template post-mcmc-logpdf
                                                     (loglik lik) (logpdf prior))))
                               nil)))


;; ==================== Distribution Models ====================================

(defn uniform-model [post-template src sampler-src]
  (device-distribution-model post-template src
                             {:name "uniform" :params-size 2
                              :limits (fge 2 1 [(- Float/MAX_VALUE) Float/MAX_VALUE])
                              :sampler-source sampler-src}))

(defn gaussian-model [post-template src sampler-src]
  (device-distribution-model post-template src
                             {:name "gaussian" :mcmc-logpdf "gaussian_mcmc_logpdf" :params-size 2
                              :limits (fge 2 1 [(- Float/MAX_VALUE) Float/MAX_VALUE])
                              :sampler-source sampler-src}))

(defn student-t-model [post-template src]
  (device-distribution-model post-template src
                             {:name "student_t" :mcmc-logpdf "student_t_mcmc_logpdf" :params-size 4
                              :limits (fge 2 1 [(- Float/MAX_VALUE) Float/MAX_VALUE])}))

(defn beta-model [post-template src]
  (device-distribution-model post-template src
                             {:name "beta" :mcmc-logpdf "beta_mcmc_logpdf" :params-size 3
                              :limits (fge 2 1 [0.0 1.0])}))

(defn exponential-model [post-template src sampler-src]
  (device-distribution-model post-template src
                             {:name "exponential" :mcmc-logpdf "exponential_mcmc_logpdf" :params-size 2
                              :limits (fge 2 1 [Float/MIN_VALUE Float/MAX_VALUE])
                              :sampler-source sampler-src}))

(defn erlang-model [post-template src sampler-src]
  (device-distribution-model post-template src
                             {:name "erlang" :mcmc-logpdf "erlang_mcmc_logpdf" :params-size 3
                              :limits (fge 2 1 [0 Float/MAX_VALUE])
                              :sampler-source sampler-src}))

(defn gamma-model [post-template src]
  (device-distribution-model post-template src
                             {:name "gamma" :mcmc-logpdf "gamma_mcmc_logpdf" :params-size 2
                              :limits (fge 2 1 [0.0 Float/MAX_VALUE])}))

(defn binomial-model [post-template src]
  (device-distribution-model post-template src
                             {:name "binomial" :mcmc-logpdf "binomial_mcmc_logpdf" :params-size 3
                              :limits (fge 2 1 [0.0 Float/MAX_VALUE])}))

(defn distribution-models [source-lib]
  (let [post-template (deref (source-lib :posterior))]
    {:uniform (delay (uniform-model post-template (deref (source-lib :uniform))
                                    (deref (source-lib :uniform-sampler))))
     :gaussian (delay (gaussian-model post-template (deref (source-lib :gaussian))
                                      (deref (source-lib :gaussian-sampler))))
     :student-t (delay (student-t-model post-template (deref (source-lib :student-t))))
     :beta (delay (beta-model post-template (deref (source-lib :beta))))
     :exponential (delay (exponential-model post-template (deref (source-lib :exponential))
                                            (deref (source-lib :exponential-sampler))))
     :erlang (delay (erlang-model post-template (deref (source-lib :erlang))
                                  (deref (source-lib :erlang-sampler))))
     :gamma (delay (gamma-model post-template (deref (source-lib :gamma))))
     :binomial (delay (binomial-model post-template (deref (source-lib :binomial))))}))

(defn likelihood-models [source-lib]
  {:gaussian (delay (device-likelihood-model (deref (source-lib :gaussian)) {:name "gaussian"}))
   :student-t (delay (device-likelihood-model (deref (source-lib :student-t)) {:name "student_t"}))
   :binomial (delay (device-likelihood-model (deref (source-lib :binomial)) {:name "binomial"}))})

(deftype DeviceLibrary [fact sources dist-models lik-models dist-engines lik-engines samplers mcmc]
  Releaseable
  (release [this]
    (release-deref (vals dist-models))
    (release-deref (vals lik-models))
    (release-deref (vals dist-engines))
    (release-deref (vals lik-engines))
    (release-deref (vals samplers)))
  FactoryProvider
  (factory [_]
    fact)
  ModelFactory
  (distribution-model [this src args]
    (device-distribution-model (get-source this :posterior) (fmap #(get-source this %) src) args))
  (likelihood-model [this src args]
    (device-likelihood-model (fmap #(get-source this %) src) args))
  Library
  (get-source [_ id]
    (if-let [source (sources id)]
      @source
      (if (string? id)
        id
        (dragan-says-ex "There is no such source code." {:id id :available (keys sources)}))))
  (get-distribution-model [_ id]
    (if-let [model (dist-models id)]
      @model
      (dragan-says-ex "There is no such distribution model." {:id id :available (keys dist-models)})))
  (get-likelihood-model [_ id]
    (if-let [model (lik-models id)]
      @model
      (dragan-says-ex "There is no such likelihood model." {:id id :available (keys lik-models)})))
  (get-distribution-engine [_ id]
    (if-let [eng (dist-engines id)]
      eng
      (dragan-says-ex "There is no such distribution engine." {:id id :available (keys dist-engines)})))
  (get-likelihood-engine [_ id]
    (if-let [eng (lik-engines id)]
      eng
      (dragan-says-ex "There is no such likelihood engine." {:id id :available (keys lik-engines)})))
  (get-direct-sampler [_ id]
    (if-let [sampler (samplers id)]
      sampler
      (dragan-says-ex "There is no such direct sampler." {:id id :available (keys samplers)})))
  (get-mcmc-factory [_ id]
    (if-let [mc (mcmc id)]
      mc
      (dragan-says-ex "There is no such mcmc sampler." {:id id :available (keys mcmc)}))))

(defn ^:private slurp-source [format-template id]
  (if-let [resource (io/resource (format format-template (name id)))]
    (slurp resource)
    (dragan-says-ex "Resource does not exist!"
                    {:id id :name (name id) :path (format format-template (name id))})))

(defn source-library
  ([source-template]
   (op (source-library (format source-template "distributions/%s")
                       [:posterior :distribution :uniform :gaussian :student-t
                        :beta :exponential :erlang :gamma :binomial])
       (source-library (format source-template "rng/%s")
                       [:uniform-sampler :gaussian-sampler :exponential-sampler
                        :erlang-sampler])))
  ([format-template names]
   (reduce #(assoc %1 %2 (delay (slurp-source format-template %2))) {} names)))

(defn device-library
  ([source-lib factory dist-models lik-models]
   (let [dist-models (op (distribution-models source-lib) dist-models)
         lik-models (op (likelihood-models source-lib) lik-models)]
     (->DeviceLibrary factory source-lib dist-models lik-models
                      (fmap #(delay (distribution-engine factory (deref %))) dist-models)
                      (fmap #(delay (likelihood-engine factory (deref %))) lik-models)
                      (fmap #(delay (direct-sampler factory (deref %))) dist-models)
                      (fmap #(delay (mcmc-factory factory (deref %))) dist-models))))
  ([source-lib factory]
   (device-library source-lib factory nil nil)))
