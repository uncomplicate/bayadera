(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl.generic
  (:require [clojure.java.io :as io]
            [uncomplicate.commons.core :refer [Releaseable release]]
            [uncomplicate.neanderthal
             [core :as nc]
             [protocols :as np]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :refer :all]
             [impl :refer [univariate-posterior-creator
                           univariate-posterior-prior-creator]]]))

;; ==================== Likelihood model ====================================

(deftype CLLikelihoodModel [name loglik-name ^long lik-params-size model-source]
  Releaseable
  (release [_]
    true)
  CLModel
  (params-size [_]
    lik-params-size)
  (source [_]
    model-source)
  (sampler-source [_]
    nil)
  LikelihoodModel
  (loglik [_]
    loglik-name))

(defn cl-likelihood-model
  [source & {:keys [name loglik params-size]
             :or {name (str (gensym "likelihood"))
                  loglik (format "%s_loglik" name)
                  params-size 1}}]
  (->CLLikelihoodModel name loglik params-size
                       (if (sequential? source) source [source])))

;; ==================== Posterior model ====================================

(declare ->CLPosteriorModel)

(deftype CLPosteriorModel [name logpdf-name mcmc-logpdf-name
                           ^long dist-dimension ^long dist-params-size
                           lower-limit upper-limit
                           model-source likelihood-model]
  Releaseable
  (release [_]
    (release likelihood-model))
  ModelProvider
  (model [this]
    this)
  DistributionModel
  (logpdf [_]
    logpdf-name)
  (mcmc-logpdf [_]
    mcmc-logpdf-name)
  (dimension [_]
    dist-dimension)
  (lower [_]
    lower-limit)
  (upper [_]
    upper-limit)
  LikelihoodModel
  (loglik [_]
    (loglik likelihood-model))
  CLModel
  (params-size [_]
    dist-params-size)
  (source [_]
    model-source)
  (sampler-source [_]
    nil))

;; ==================== Distribution model ====================================

(deftype CLDistributionModel [name logpdf-name mcmc-logpdf-name
                              ^long dist-dimension ^long dist-params-size
                              lower-limit upper-limit
                              model-source sampler-kernels]
  Releaseable
  (release [_]
    (and
     (release lower-limit)
     (release upper-limit)))
  ModelProvider
  (model [this]
    this)
  DistributionModel
  (logpdf [_]
    logpdf-name)
  (mcmc-logpdf [_]
    mcmc-logpdf-name)
  (dimension [_]
    dist-dimension)
  (lower [_]
    lower-limit)
  (upper [_]
    upper-limit)
  CLModel
  (params-size [_]
    dist-params-size)
  (source [_]
    model-source)
  (sampler-source [_]
    sampler-kernels))

(defn cl-distribution-model
  [source & {:keys [name logpdf mcmc-logpdf dimension params-size
                    lower upper sampler-source]
             :or {name (str (gensym "distribution"))
                  logpdf (format "%s_logpdf" name)
                  mcmc-logpdf logpdf dimension 1 params-size 1}}]
  (->CLDistributionModel name logpdf mcmc-logpdf dimension params-size lower upper
                         (if (sequential? source)
                           source
                           [source])
                         (if (sequential? sampler-source)
                           sampler-source
                           [sampler-source])))

;; ================ Posterior multimethod implementations ======================

(let [post-source (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/posterior.cl"))]

  (defn cl-posterior-model [^String name lik prior]
    (let [post-name (str (gensym name))
          post-logpdf (format "%s_logpdf" post-name)
          post-mcmc-logpdf (format "%s_mcmc_logpdf" post-name)
          post-params-size (+ (long (params-size lik))
                              (long (params-size prior)))]
      (->CLPosteriorModel post-name post-logpdf post-mcmc-logpdf
                          (dimension prior) post-params-size
                          (lower prior) (upper prior)
                          (conj (into [] (dedupe)
                                      (into (source prior) (source lik)))
                                (format "%s\n%s"
                                        (format post-source post-logpdf
                                                (loglik lik) (logpdf prior)
                                                (params-size lik))
                                        (format post-source post-mcmc-logpdf
                                                (loglik lik) (mcmc-logpdf prior)
                                                (params-size lik))))
                          lik))))

(defmethod posterior-model [CLLikelihoodModel CLDistributionModel]
  [name likelihood prior]
  (cl-posterior-model name likelihood prior))

(defmethod posterior-model [CLLikelihoodModel CLPosteriorModel]
  [name likelihood prior]
  (cl-posterior-model name likelihood prior))

(defmethod posterior-model [CLLikelihoodModel Distribution]
  [name likelihood prior]
  (cl-posterior-model name likelihood (model prior)))

(defmethod posterior [1 CLPosteriorModel]
  [factory model]
  (univariate-posterior-creator factory model))

(defmethod posterior [1 CLLikelihoodModel DistributionModel]
  [factory name likelihood prior]
  (univariate-posterior-creator
   factory (cl-posterior-model name likelihood (model prior))))

(defmethod posterior
  [1 CLLikelihoodModel uncomplicate.bayadera.protocols.Distribution]
  [factory name likelihood prior]
  (univariate-posterior-prior-creator
   factory (nc/copy (parameters prior))
   (cl-posterior-model name likelihood (model prior))))

;; ==================== Distribution Models ====================================

(def gaussian-model
  (cl-distribution-model (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/gaussian.h"))
                         :name "gaussian" :params-size 2
                         :lower (sv (- Float/MAX_VALUE)) :upper (sv Float/MAX_VALUE)
                         :sampler-source
                         (slurp (io/resource "uncomplicate/bayadera/opencl/rng/gaussian-sampler.cl"))))
(def uniform-model
  (cl-distribution-model (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/uniform.h"))
                         :name "uniform" :params-size 2
                         :lower (- Float/MAX_VALUE) :upper (sv Float/MAX_VALUE)
                         :sampler-source
                         (slurp (io/resource "uncomplicate/bayadera/opencl/rng/uniform-sampler.cl"))))
(def beta-model
  (cl-distribution-model (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/beta.h"))
                         :name "beta" :mcmc-logpdf "beta_mcmc_logpdf" :params-size 3
                         :lower (sv 0.0) :upper (sv 1.0)))

;; TODO support is from 0 to infinity

(def binomial-model
  (cl-distribution-model (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/binomial.h"))
                         :name "binomial" :mcmc-logpdf "binomial_mcmc_logpdf" :params-size 3
                         :lower (sv 0.0) :upper (sv Float/MAX_VALUE)))

(def binomial-likelihood
  (cl-likelihood-model (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/binomial.h"))
                       :name "binomial" :params-size 2))
