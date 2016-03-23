(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl.utils
  (:require [clojure.java.io :as io]
            [me.raynes.fs :as fsc]))

(defn copy-random123 [include-name tmp-dir-name]
  (io/copy
   (io/input-stream
    (io/resource (format "uncomplicate/bayadera/opencl/rng/include/Random123/%s"
                         include-name)))
   (io/file (format "%s/Random123/%s" tmp-dir-name include-name))))

(defn get-tmp-dir-name []
  (fsc/temp-dir "uncomplicate/"))

(defmacro with-philox [tmp-dir-name & body]
  `(try
     (fsc/mkdirs (format "%s/%s" ~tmp-dir-name "Random123/features/"))
     (doseq [res-name# ["philox.h" "array.h" "features/compilerfeatures.h"
                        "features/openclfeatures.h" "features/sse.h"]]
       (copy-random123 res-name# ~tmp-dir-name))
     (do ~@body)
     (finally
       (fsc/delete-dir ~tmp-dir-name))))
