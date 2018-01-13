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
