;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.models
  (:require [clojure.java.io :as io]
            [uncomplicate.commons.core :refer [Releaseable release]]
            [uncomplicate.neanderthal.internal.api :as na]
            [uncomplicate.neanderthal
             [core :refer [copy]]
             [native :refer [fge]]]
            [uncomplicate.bayadera.internal.protocols :refer :all]))

(defprotocol CLModel
  (source [this])
  (sampler-source [this]))

;; ==================== Likelihood model ====================================

(deftype CLLikelihoodModel [name loglik-name ^long lik-params-size model-source]
  Releaseable
  (release [_]
    true)
  na/MemoryContext
  (compatible? [_ o]
    (satisfies? CLModel o))
  Model
  (params-size [_]
    lik-params-size)
  CLModel
  (source [_]
    model-source)
  (sampler-source [_]
    nil)
  LikelihoodModel
  (loglik [_]
    loglik-name))

(defn likelihood-model [source & {:keys [name loglik params-size]
                                  :or {name (str (gensym "likelihood"))
                                       loglik (format "%s_loglik" name)
                                       params-size 1}}]
  (->CLLikelihoodModel name loglik params-size (if (sequential? source) source [source])))

;; ==================== Distribution model ====================================

(declare cl-posterior-model)

(deftype CLDistributionModel [post-template name logpdf-name mcmc-logpdf-name
                              ^long dist-dimension ^long dist-params-size
                              model-limits model-source sampler-kernels]
  Releaseable
  (release [_]
    (release model-limits))
  na/MemoryContext
  (compatible? [_ o]
    (satisfies? CLModel o))
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
    (cl-posterior-model post-template prior name likelihood))
  CLModel
  (source [_]
    model-source)
  (sampler-source [_]
    sampler-kernels)
  ModelProvider
  (model [this]
    this))

(defn distribution-model
  [post-template source & {:keys [name logpdf mcmc-logpdf dimension params-size
                                  limits sampler-source]
                           :or {name (str (gensym "distribution"))
                                logpdf (format "%s_logpdf" name)
                                mcmc-logpdf logpdf dimension 1 params-size 1}}]
  (->CLDistributionModel post-template name logpdf mcmc-logpdf dimension params-size limits
                         (if (sequential? source) source [source])
                         (if (sequential? sampler-source) sampler-source [sampler-source])))

;; ==================== Posterior model ====================================

(deftype CLPosteriorModel [post-template name logpdf-name mcmc-logpdf-name
                           ^long dist-dimension ^long dist-params-size
                           model-limits model-source likelihood-model]
  Releaseable
  (release [_]
    (release model-limits))
  na/MemoryContext
  (compatible? [_ o]
    (satisfies? CLModel o))
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
  LikelihoodModel
  (loglik [_]
    (loglik likelihood-model))
  PriorModel
  (posterior-model [prior name likelihood]
    (cl-posterior-model post-template prior name likelihood))
  CLModel
  (source [_]
    model-source)
  (sampler-source [_]
    nil)
  ModelProvider
  (model [this]
    this))

(defn cl-posterior-model [post-template prior name lik]
  (let [post-name (str (gensym name))
        post-logpdf (format "%s_logpdf" post-name)
        post-mcmc-logpdf (format "%s_mcmc_logpdf" post-name)
        post-params-size (+ ^long (params-size lik) ^long (params-size prior))]
    (->CLPosteriorModel post-template post-name post-logpdf post-mcmc-logpdf
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
                        lik)))


;; ==================== Distribution Models ====================================

(defn uniform-model [post-template source sampler]
  (distribution-model post-template source
                      :name "uniform":mcmc-logpdf "uniform_mcmc_logpdf"
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
