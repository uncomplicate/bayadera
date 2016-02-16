(ns uncomplicate.bayadera.opencl
  (:require [uncomplicate.clojurecl.core
             :refer [with-default *context* *command-queue*]]
            [uncomplicate.bayadera.core :refer [with-bayadera *bayadera-factory*]]
            [uncomplicate.bayadera.opencl.amd-gcn :refer [gcn-engine-factory]]))

(defmacro with-default-bayadera
  [& body]
  `(with-default
     (with-bayadera gcn-engine-factory [*context* *command-queue*] ~@body)))
