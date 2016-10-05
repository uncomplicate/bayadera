;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl.util
  (:require [clojure.java.io :as io]
            [me.raynes.fs :as fsc]))

(defn clean-random123 [tmp-dir-name]
  (fsc/delete-dir tmp-dir-name))

(defn copy-random123 [tmp-dir-name include-names]
  (doseq [include-name include-names]
    (io/copy
     (io/input-stream
      (io/resource (format "uncomplicate/bayadera/opencl/rng/include/Random123/%s"
                           include-name)))
     (io/file (format "%s/Random123/%s" tmp-dir-name include-name)))))

(defn copy-philox [tmp-dir-name]
 (try
   (fsc/mkdirs (format "%s/%s" tmp-dir-name "Random123/features/"))
   (copy-random123 tmp-dir-name
                   ["philox.h" "array.h" "features/compilerfeatures.h"
                    "features/openclfeatures.h" "features/sse.h"])
   (catch Exception e
     (clean-random123 tmp-dir-name)
     (throw e))))

(defn get-tmp-dir-name []
  (fsc/temp-dir "uncomplicate/"))

(defmacro with-philox [tmp-dir-name & body]
  `(try
     (copy-philox ~tmp-dir-name)
     (do ~@body)
     (finally
       (clean-random123 ~tmp-dir-name))))
