;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecl.core :refer [with-default *context* *command-queue*]]
            [uncomplicate.bayadera.core :refer [with-bayadera *bayadera-factory*]]
            [uncomplicate.bayadera.internal.device
             [models :as models]
             [amd-gcn :as amd-gcn]]))

(def uniform-source (slurp (io/resource "uncomplicate/bayadera/internal/opencl/distributions/uniform.cl")))
(def gaussian-source (slurp (io/resource "uncomplicate/bayadera/internal/opencl/distributions/gaussian.cl")))
(def student-t-source (slurp (io/resource "uncomplicate/bayadera/internal/opencl/distributions/student-t.cl")))
(def beta-source (slurp (io/resource "uncomplicate/bayadera/internal/opencl/distributions/beta.cl")))
(def exponential-source (slurp (io/resource "uncomplicate/bayadera/internal/opencl/distributions/exponential.cl")))
(def erlang-source (slurp (io/resource "uncomplicate/bayadera/internal/opencl/distributions/erlang.cl")))
(def gamma-source (slurp (io/resource "uncomplicate/bayadera/internal/opencl/distributions/gamma.cl")))
(def binomial-source (slurp (io/resource "uncomplicate/bayadera/internal/opencl/distributions/binomial.cl")))

(let [post-template (slurp (io/resource "uncomplicate/bayadera/internal/opencl/distributions/posterior.cl"))
      uniform-sampler (slurp (io/resource "uncomplicate/bayadera/internal/opencl/rng/uniform-sampler.cl"))
      gaussian-sampler (slurp (io/resource "uncomplicate/bayadera/internal/opencl/rng/gaussian-sampler.cl"))
      exponential-sampler (slurp (io/resource "uncomplicate/bayadera/internal/opencl/rng/exponential-sampler.cl"))
      erlang-sampler (slurp (io/resource "uncomplicate/bayadera/internal/opencl/rng/erlang-sampler.cl"))]

  (defn cl-distribution-model [source & args]
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

  (def gcn-bayadera-factory (partial amd-gcn/gcn-bayadera-factory
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
     (with-bayadera gcn-bayadera-factory [*context* *command-queue*] ~@body)))

;; ==================== Function template ======================================

(let [function-source (slurp (io/resource "uncomplicate/bayadera/internal/opencl/distributions/distribution.cl"))]

  (defn fn-source [^String name ^String body]
    (format function-source name body))
)
