;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.host.mkl
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release]]
             [utils :refer [direct-buffer]]]
            [uncomplicate.neanderthal
             [core :refer [dim ge]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.internal.host.mkl :refer [mkl-float]]
            [uncomplicate.bayadera.util :refer [srand-int]]
            [uncomplicate.bayadera.internal.protocols :refer :all])
  (:import uncomplicate.neanderthal.internal.host.MKL))

;; ============================ Direct sampler =================================

(defmacro rng-method [method]
  `(fn [stream# params# res#]
     (let [err# (~method stream# (dim res#) (buffer res#) (buffer params#))]
       (if (= 0 err#)
         res#
         (throw (ex-info "MKL error." {:error-code err#}))))))

(defn new-ars5
  ([seed stream]
   (let [err (MKL/vslNewStreamARS5 seed stream)]
     (if (= 0 err)
       stream
       (ex-info "MKL error." {:error-code err}))))
  ([seed]
   (let-release [stream (direct-buffer Long/BYTES)]
     (new-ars5 seed stream))))

(defn release-stream [stream]
  (let [err (MKL/vslDeleteStream stream)]
    (if (= 0 err)
      (release stream)
      (ex-info "MKL error." {:error-code err}))))

(deftype MKLDirectSamplerEngine [method]
  RandomSamplerEngine
  (sample [_ stream params res]
    (method stream params res)))

(deftype MKLDirectSampler [dist-eng stream sample-count params]
  Releaseable
  (release [_]
    (release-stream stream)
    (release params))
  RandomSampler
  (sample! [this]
    (sample! this sample-count))
  (sample! [this n-or-res]
    (if (integer? n-or-res)
      (let-release [res (ge params 1 n-or-res {:raw true})]
        (sample dist-eng stream params res))
      (sample dist-eng stream params n-or-res))))

;; ======================== Constructor functions ==============================

(def float-gaussian-sampler (->MKLDirectSamplerEngine (rng-method MKL/vsRngGaussian)))
(def double-gaussian-sampler (->MKLDirectSamplerEngine (rng-method MKL/vdRngGaussian)))
(def float-uniform-sampler (->MKLDirectSamplerEngine (rng-method MKL/vsRngUniform)))
(def double-uniform-sampler (->MKLDirectSamplerEngine (rng-method MKL/vdRngUniform)))

(defn create-mkl-direct-sampler [dist-eng seed sample-count params]
  (let-release [stream (new-ars5 seed)]
    (->MKLDirectSampler dist-eng stream sample-count params)))
