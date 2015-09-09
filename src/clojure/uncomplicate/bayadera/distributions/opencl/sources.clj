(ns uncomplicate.bayadera.distributions.opencl.sources
  (:require [clojure.java.io :as io]))

(defn logpdf-source [logpdf-name]
  (format "inline float logpdf(__constant float* params, float x) {return %s(params, x);}" logpdf-name))

(def ^:constant gaussian-source
  (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/gaussian.h")))

(deftype GaussianSource []
  CLSource
  (logpdf-source [_]
    gaussian-source))

(def ^:constant uniform-source
  (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/uniform.h")))

(deftype UniformSource []
  CLSource
  (logpdf-source [_]
    uniform-source))
