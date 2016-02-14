(ns uncomplicate.bayadera.opencl.generic
  (:require [clojure.java.io :as io]
            [uncomplicate.bayadera.protocols :refer :all]))

(deftype CLLikelihoodModel [name loglik-name ^long lik-params-size model-source]
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

(def ^:private posterior-logpdf
  (slurp (io/resource "uncomplicate/bayadera/opencl/templates/posterior.clt")))

(deftype CLDistributionModel [name logpdf-name mcmc-logpdf-name
                              ^long dist-dimension ^long dist-params-size
                              lower-limit upper-limit
                              model-source sampler-kernels
                              likelihood-model]
  DistributionModel
  (logpdf [_]
    logpdf-name)
  (mcmc-logpdf [_]
    (or mcmc-logpdf-name logpdf-name))
  (dimension [_]
    dist-dimension)
  (lower [_]
    lower-limit)
  (upper [_]
    upper-limit)
  LikelihoodModel
  (loglik [_]
    (if likelihood-model
      (loglik likelihood-model)
      nil))
  CLModel
  (params-size [_]
    dist-params-size)
  (source [_]
    model-source)
  (sampler-source [_]
    sampler-kernels)
  Bayes
  (posterior [prior likelihood]
    (let [likelihood ^CLLikelihoodModel likelihood
          post-name (str (gensym "posterior"))
          post-params-size (+ (long (params-size likelihood)) dist-params-size)]
      (CLDistributionModel. post-name post-name post-name
                            dist-dimension post-params-size lower-limit upper-limit
                            (conj (into [] (dedupe)
                                        (into model-source (source likelihood)))
                                  (format posterior-logpdf post-name
                                          (loglik likelihood) (mcmc-logpdf prior)
                                          (params-size likelihood)))
                            nil likelihood))))

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
                           [sampler-source])
                         nil))

;; TODO kernels (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/kernels.cl"));;TODO move to engine

(def gaussian-model
  (cl-distribution-model (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/gaussian.h"))
                         :name "gaussian" :params-size 2
                         :lower Double/MIN_VALUE :upper Double/MAX_VALUE
                         :sampler-source
                         (slurp (io/resource "uncomplicate/bayadera/opencl/rng/gaussian-sampler.cl"))))
(def uniform-model
  (cl-distribution-model (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/uniform.h"))
                         :name "uniform" :params-size 2
                         :lower Double/MIN_VALUE :upper Double/MAX_VALUE
                         :sampler-source
                         (slurp (io/resource "uncomplicate/bayadera/opencl/rng/uniform-sampler.cl"))))
(def beta-model
  (cl-distribution-model (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/beta.h"))
                         :name "beta" :mcmc-logpdf "beta_mcmc_logpdf" :params-size 3
                         :lower 0.0 :upper 1.0))

  ;; TODO support is from 0 to infinity

(def binomial-model
  (cl-distribution-model (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/binomial.h"))
                         :name "binomial" :mcmc-logpdf "binomial_mcmc_logpdf" :params-size 3
                         :lower 0 :upper Double/MAX_VALUE))

(def binomial-likelihood
  (cl-likelihood-model (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/binomial.h"))
                       :name "binomial" :params-size 2))
