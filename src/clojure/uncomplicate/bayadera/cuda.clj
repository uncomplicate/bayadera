;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.cuda
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecuda.core :refer [with-default current-context default-stream]]
            [uncomplicate.bayadera.core :refer [with-bayadera *bayadera-factory*]]
            [uncomplicate.bayadera.internal.device
             [models :as models]
             [nvidia-gtx :as nvidia-gtx]]))

(def uniform-source (slurp (io/resource "uncomplicate/bayadera/internal/cuda/distributions/uniform.cu")))
(def gaussian-source (slurp (io/resource "uncomplicate/bayadera/internal/cuda/distributions/gaussian.cu")))
(def student-t-source (slurp (io/resource "uncomplicate/bayadera/internal/cuda/distributions/student-t.cu")))
(def beta-source (slurp (io/resource "uncomplicate/bayadera/internal/cuda/distributions/beta.cu")))
(def exponential-source (slurp (io/resource "uncomplicate/bayadera/internal/cuda/distributions/exponential.cu")))
(def erlang-source (slurp (io/resource "uncomplicate/bayadera/internal/cuda/distributions/erlang.cu")))
(def gamma-source (slurp (io/resource "uncomplicate/bayadera/internal/cuda/distributions/gamma.cu")))
(def binomial-source (slurp (io/resource "uncomplicate/bayadera/internal/cuda/distributions/binomial.cu")))

(let [post-template (slurp (io/resource "uncomplicate/bayadera/internal/cuda/distributions/posterior.cu"))
      uniform-sampler (slurp (io/resource "uncomplicate/bayadera/internal/cuda/rng/uniform-sampler.cu"))
      gaussian-sampler (slurp (io/resource "uncomplicate/bayadera/internal/cuda/rng/gaussian-sampler.cu"))
      exponential-sampler (slurp (io/resource "uncomplicate/bayadera/internal/cuda/rng/exponential-sampler.cu"))
      erlang-sampler (slurp (io/resource "uncomplicate/bayadera/internal/cuda/rng/erlang-sampler.cu"))]

  (defn cu-distribution-model [source & args]
    (apply models/distribution-model post-template source args))

  (def uniform-model (models/uniform-model post-template uniform-source uniform-sampler))
  (def gaussian-model (models/gaussian-model post-template gaussian-source gaussian-sampler))
  (def student-t-model (models/student-t-model post-template student-t-source))
  (def beta-model (models/beta-model post-template beta-source))
  (def exponential-model (models/exponential-model post-template exponential-source exponential-sampler))
  (def erlang-model (models/erlang-model post-template erlang-source erlang-sampler))
  (def gamma-model (models/gamma-model post-template gamma-source))
  (def binomial-model (models/gamma-model post-template binomial-source))

  (def gaussian-lik-model (models/gaussian-lik-model gaussian-source))
  (def student-t-lik-model (models/student-t-lik-model student-t-source))
  (def binomial-lik-model (models/binomial-lik-model binomial-source))

  (def gtx-bayadera-factory (partial nvidia-gtx/gtx-bayadera-factory
                                     {:uniform uniform-model
                                      :gaussian gaussian-model
                                      :student-t student-t-model
                                      :beta beta-model
                                      :exponential exponential-model
                                      :erlang erlang-model
                                      :gamma gamma-model
                                      :binomial binomial-model}
                                     {:uniform uniform-sampler
                                      :gaussian gaussian-sampler
                                      :exponential exponential-sampler
                                      :erlang erlang-sampler})))

(defmacro with-default-bayadera
  [& body]
  `(with-default
     (with-bayadera gtx-bayadera-factory [(current-context) default-stream] ~@body)))

;; ==================== Function template ======================================

(let [function-source (slurp (io/resource "uncomplicate/bayadera/internal/cuda/distributions/distribution.cu"))]

  (defn fn-source [^String name ^String body]
    (format function-source name body)))
