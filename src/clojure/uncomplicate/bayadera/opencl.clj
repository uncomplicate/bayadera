(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl
  (:require [uncomplicate.clojurecl
             [core :refer [*context* *command-queue* release]]
             [toolbox :refer [wrap-float wrap-double]]
             [info :refer [queue-context]]]
            [uncomplicate.bayadera.opencl
             [amd-gcn :refer [gcn-dataset-engine gcn-distribution-engine gcn-direct-sampler]]]))

(defmacro with-engine
  ([queue & body]
   `(binding [*double-factory*
              (~factory-fn double-accessor ~@params)]
      (try
        (binding [*single-factory* (~factory-fn float-accessor ~@params)]
          (try ~@body
               (finally (release *single-factory*))))
        (finally (release *double-factory*))))))

(defn gaussian [queue]
  )
