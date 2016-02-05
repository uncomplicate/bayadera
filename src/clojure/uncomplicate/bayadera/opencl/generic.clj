(ns uncomplicate.bayadera.opencl.generic
  (:require [clojure.java.io :as io]
            [uncomplicate.bayadera.protocols :refer [Bayes posterior]]))

(defrecord CLLikelihoodModel [name ^long params-size functions])

(def ^:private posterior-kernels
  (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/posterior.cl")))
(def ^:private posterior-logpdf
  (slurp (io/resource "uncomplicate/bayadera/opencl/templates/posterior.clt")))

(defrecord CLDistributionModel [name ^long dimension ^long params-size
                                lower upper functions kernels]
  Bayes
  (posterior [prior likelihood]
    (let [logpdf (str (gensym "logpdf"))
          params-size (+ (.params-size likelihood) (.params-size prior))]
      (->CLDistributionModel logpdf dimension params-size lower upper
                             (str functions "\n" (.functions likelihood) "\n"
                                  (format posterior-logpdf
                                          logpdf (.name likelihood)
                                          name (.params-size likelihood)))
                             posterior-kernels))))

(let [uniform-src (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/uniform.h"))]

  (def gaussian-model
    (->CLDistributionModel "gaussian_logpdf" 1 2 nil nil
                           (str uniform-src "\n"
                                (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/gaussian.h")))
                           (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/gaussian.cl"))))
  (def uniform-model
    (->CLDistributionModel "uniform_logpdf" 1 2 nil nil
                           uniform-src
                           (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/uniform.cl"))))
  (def beta-model
    (->CLDistributionModel  "beta_logpdf" 1 3 0 1
                            (str uniform-src "\n"
                                 (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/beta.h")))
                            (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/beta.cl"))))

  ;; TODO support is from 0 to infinity
  (def binomial-model
    (->CLDistributionModel "binomial_logpdf" 1 3 nil nil
                           (str uniform-src "\n"
                                (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/binomial.h")))
                           (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/binomial.cl"))))

  (def binomial-likelihood
    (->CLLikelihoodModel "binomial_loglik" 2
                         (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/binomial.h")))))
