;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.library
  (:require [uncomplicate.commons
             [core :refer [release]]
             [utils :refer [dragan-says-ex cond-into]]]
            [uncomplicate.neanderthal.core :refer [vctr]]
            [uncomplicate.bayadera.distributions
             :refer [uniform-params gaussian-params student-t-params beta-params gamma-params
                     exponential-params erlang-params]]
            [uncomplicate.bayadera.internal
             [protocols :as p]
             [distribution-library :refer [->UniformDistribution ->GaussianDistribution
                                           ->StudentTDistribution ->BetaDistribution
                                           ->GammaDistribution ->ExponentialDistribution
                                           ->ErlangDistribution ->LibraryLikelihood]]]))

(defn ^:private mcmc-factory [library id]
  (deref (p/get-mcmc-factory library id)))

(defn ^:private distribution-engine [library id]
  (deref (p/get-distribution-engine library id)))

(defn ^:private likelihood-engine [library id]
  (deref (p/get-likelihood-engine library id)))

(def ^:dynamic *library*)

(defmacro with-library [library & body]
  `(binding [*library* ~library]
     (try ~@body
          (finally (release *library*)))))

(defn distribution-model
  ([id]
   (p/get-distribution-model *library* id))
  ([library-or-src src-or-args]
   (if (keyword? src-or-args)
     (p/get-distribution-model library-or-src src-or-args)
     (p/distribution-model *library* library-or-src src-or-args)))
  ([library src args]
   (p/distribution-model library src args)))

(defn likelihood-model
  ([id]
   (p/get-likelihood-model *library* id))
  ([library-or-src src-or-args]
   (cond
     (keyword? src-or-args) (p/get-likelihood-model library-or-src src-or-args)
     (string? src-or-args) (p/likelihood-model library-or-src src-or-args nil)
     :default (p/likelihood-model *library* library-or-src src-or-args)))
  ([library src args]
   (p/likelihood-model library src args)))

(defn source
  ([id]
   (p/get-source *library* id))
  ([library id]
   (p/get-source library id)))

(defn likelihood
  ([id]
   (likelihood *library* id))
  ([library id]
   (->LibraryLikelihood (p/factory library) (likelihood-engine library id) (likelihood-model library id))))

;; =================== Distributions ===========================================

(defn uniform
  ([^double a ^double b]
   (uniform *library* a b))
  ([library ^double a ^double b]
   (if-let [params (uniform-params a b)]
     (->UniformDistribution (p/factory library)
                            (p/get-direct-sampler-engine library :uniform)
                            (distribution-engine library :uniform)
                            (vctr (p/factory library) params) a b)
     (dragan-says-ex "Uniform distribution parameters are illegal."
                     {:a a :b b :errors
                      (when-not (< a b) "a is not less than b")}))))

(defn gaussian
  ([^double mu ^double sigma]
   (gaussian *library* mu sigma))
  ([library ^double mu ^double sigma]
   (if-let [params (gaussian-params mu sigma)]
     (->GaussianDistribution (p/factory library)
                             (p/get-direct-sampler-engine library :gaussian)
                             (distribution-engine library :gaussian)
                             (vctr (p/factory library) params) mu sigma)
     (dragan-says-ex "Gaussian distribution parameters are illegal."
                     {:mu mu :sigma sigma :errors (when-not (< 0.0 sigma) "sigma is not positive")}))))

(defn student-t
  ([^double nu ^double mu ^double sigma]
   (student-t *library* nu mu sigma))
  ([^double nu]
   (student-t nu 0.0 1.0))
  ([library ^double nu ^double mu ^double sigma]
   (if-let [params (student-t-params nu mu sigma)]
     (->StudentTDistribution (p/factory library)
                             (p/get-mcmc-factory library :student-t)
                             (distribution-engine library :student-t)
                             (vctr (p/factory library) params) nu mu sigma)
     (dragan-says-ex "Student's t distribution parameters are illegal."
                     {:nu nu :mu mu :sigma sigma :errors
                      (cond-into []
                                 (not (< 0.0 nu) "nu is not positive")
                                 (not (< 0.0 sigma)) "sigma is not positive")})))
  ([factory ^double nu]
   (student-t factory nu 0.0 1.0)))

(defn beta
  ([^double a ^double b]
   (beta *library* a b))
  ([library ^double a ^double b]
   (if-let [params (beta-params a b)]
     (->BetaDistribution (p/factory library)
                         (p/get-mcmc-factory library :beta)
                         (distribution-engine library :beta)
                         (vctr (p/factory library) params) a b)
     (dragan-says-ex "Beta distribution parameters are illegal."
                     {:a a :b b :errors
                      (cond-into []
                                 (not (< 0.0 a) "a is not positive")
                                 (not (< 0.0 b)) "b is not positive")}))))

(defn gamma
  ([^double theta ^double k]
   (beta *library* theta k))
  ([library ^double theta ^double k]
   (if-let [params (gamma-params theta k)]
     (->GammaDistribution (p/factory library)
                          (p/get-mcmc-factory library :gamma)
                          (distribution-engine library :gamma)
                          (vctr (p/factory library) params) theta k)
     (dragan-says-ex "Gamma distribution parameters are illegal."
                     {:theta theta :k k :errors
                      (cond-into []
                                 (not (< 0.0 theta) "theta is not positive")
                                 (not (< 0.0 k)) "k is not positive")}))))

(defn exponential
  ([^double lambda]
   (exponential *library* lambda))
  ([library ^double lambda]
   (if-let [params (exponential-params lambda)]
     (->ExponentialDistribution (p/factory library)
                                (p/get-direct-sampler-engine library :exponential)
                                (distribution-engine library :exponential)
                                (vctr (p/factory library) params) lambda)
     (dragan-says-ex "Exponential distribution parameters are illegal."
                     {:lambda lambda :errors (when-not (< 0.0 lambda) "lambda is not positive")}))))

(defn erlang
  ([^double lambda ^long k]
   (erlang *library* lambda k))
  ([library ^double lambda ^long k]
   (if-let [params (erlang-params lambda k)]
     (->ErlangDistribution (p/factory library)
                           (p/get-direct-sampler-engine library :erlang)
                           (distribution-engine library :erlang)
                           (vctr (p/factory library) params) lambda k)
     (dragan-says-ex "Erlang distribution parameters are illegal."
                     {:lambda lambda :k k :errors
                      (cond-into []
                                 (not (< 0.0 lambda) "lambda is not positive")
                                 (not (< 0.0 k)) "k is not positive")}))))
