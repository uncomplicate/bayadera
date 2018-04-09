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
            [uncomplicate.commons.core :refer [Releaseable release]]
            [uncomplicate.neanderthal.internal.api :as na]
            [uncomplicate.neanderthal
             [core :refer [copy]]
             [native :refer [fge]]]
            [uncomplicate.bayadera.internal.protocols :refer :all]))

(defprotocol DeviceModel
  (dialect [this])
  (source [this])
  (sampler-source [this]))

;; ==================== Likelihood model ====================================

(deftype DeviceLikelihoodModel [model-dialect name loglik-name ^long lik-params-size model-source]
  Releaseable
  (release [_]
    true)
  na/MemoryContext
  (compatible? [_ o]
    (and (satisfies? DeviceModel o) (= model-dialect (dialect o))))
  Model
  (params-size [_]
    lik-params-size)
  DeviceModel
  (dialect [_]
    model-dialect)
  (source [_]
    model-source)
  (sampler-source [_];;TODO remove? integrate into source?
    nil)
  LikelihoodModel
  (loglik [_]
    loglik-name)
  ModelProvider
  (model [this]
    this))

(defn likelihood-model [source & {:keys [dialect name loglik params-size]
                                  :or {dialect :c99
                                       name (str (gensym "likelihood"))
                                       loglik (format "%s_loglik" name)
                                       params-size 1}}]
  (->DeviceLikelihoodModel dialect name loglik params-size (if (sequential? source) source [source])))

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
  Model
  (params-size [_]
    dist-params-size)
  DistributionModel
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

(defn distribution-model
  [post-template source & {:keys [dialect name logpdf mcmc-logpdf dimension params-size
                                  limits sampler-source]
                           :or {dialect :c99
                                name (str (gensym "distribution"))
                                logpdf (format "%s_logpdf" name)
                                mcmc-logpdf logpdf dimension 1 params-size 1}}]
  (->DeviceDistributionModel dialect post-template name logpdf mcmc-logpdf dimension params-size limits
                             (if (sequential? source) source [source])
                             (if (sequential? sampler-source) sampler-source [sampler-source])))

(defn device-posterior-model [post-template dialect prior name lik]
  (let [post-name (str (gensym name))
        post-logpdf (format "%s_logpdf" post-name)
        post-mcmc-logpdf (format "%s_mcmc_logpdf" post-name)
        post-params-size (+ ^long (params-size lik) ^long (params-size prior))]
    (->DeviceDistributionModel dialect post-template post-name post-logpdf post-mcmc-logpdf
                               (dimension prior) post-params-size
                               (when (limits prior) (copy (limits prior)))
                               (conj (vec (distinct (into (source prior) (source lik))))
                                     (format "%s\n%s"
                                             (format post-template post-logpdf
                                                     (loglik lik) (logpdf prior)
                                                     (params-size lik))
                                             (format post-template post-mcmc-logpdf
                                                     (loglik lik) (mcmc-logpdf prior)
                                                     (params-size lik))))
                               nil)))

;; ==================== Distribution Models ====================================

(defn uniform-model [post-template source sampler]
  (distribution-model post-template source
                      :name "uniform"
                      :params-size 2
                      :limits (fge 2 1 [(- Float/MAX_VALUE) Float/MAX_VALUE])
                      :sampler-source sampler))

(defn gaussian-model [post-template source sampler]
  (distribution-model post-template source
                      :name "gaussian" :mcmc-logpdf "gaussian_mcmc_logpdf"
                      :params-size 2
                      :limits (fge 2 1 [(- Float/MAX_VALUE) Float/MAX_VALUE])
                      :sampler-source sampler))

(defn student-t-model [post-template source]
  (distribution-model post-template source
                      :name "student_t" :mcmc-logpdf "student_t_mcmc_logpdf"
                      :params-size 4
                      :limits (fge 2 1 [(- Float/MAX_VALUE) Float/MAX_VALUE])))

(defn beta-model [post-template source]
  (distribution-model post-template source
                      :name "beta" :mcmc-logpdf "beta_mcmc_logpdf" :params-size 3
                      :limits (fge 2 1 [0.0 1.0])))

(defn exponential-model [post-template source sampler]
  (distribution-model post-template source
                      :name "exponential" :mcmc-logpdf "exponential_mcmc_logpdf"
                      :params-size 2
                      :limits (fge 2 1 [Float/MIN_VALUE Float/MAX_VALUE])
                      :sampler-source sampler))

(defn erlang-model [post-template source sampler]
  (distribution-model post-template source
                      :name "erlang" :mcmc-logpdf "erlang_mcmc_logpdf"
                      :params-size 3
                      :limits (fge 2 1 [0 Float/MAX_VALUE])
                      :sampler-source sampler))

(defn gamma-model [post-template source]
  (distribution-model post-template source
                      :name "gamma" :mcmc-logpdf "gamma_mcmc_logpdf"
                      :params-size 2
                      :limits (fge 2 1 [0.0 Float/MAX_VALUE])))

(defn binomial-model [post-template source]
  (distribution-model post-template source
                      :name "binomial" :mcmc-logpdf "binomial_mcmc_logpdf" :params-size 3
                      :limits (fge 2 1 [0.0 Float/MAX_VALUE])))

(defn gaussian-lik-model [source]
  (fn [n] (likelihood-model source :name "gaussian" :params-size n)))

(defn student-t-lik-model [source]
  (fn [n] (likelihood-model source :name "student_t" :params-size n)))

(defn binomial-lik-model [source]
  (likelihood-model source :name "binomial" :params-size 2))
