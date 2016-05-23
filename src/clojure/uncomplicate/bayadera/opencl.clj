(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecl.core
             :refer [with-default *context* *command-queue*]]
            [uncomplicate.bayadera
             [core :refer [with-bayadera *bayadera-factory*]]]
            [uncomplicate.bayadera.opencl
             [models :refer [->CLLikelihoodModel ->CLPosteriorModel]]
             [amd-gcn :refer [gcn-bayadera-factory]]]))

(defmacro with-default-bayadera
  [& body]
  `(with-default
     (with-bayadera gcn-bayadera-factory [*context* *command-queue*] ~@body)))

;; ==================== Function template ======================================

(let [function-source (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/distribution.cl"))]

  (defn fn-source [^String name ^String body]
    (format function-source name body)))
