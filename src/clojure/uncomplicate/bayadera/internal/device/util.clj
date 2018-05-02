;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.device.util
  (:require [uncomplicate.commons
             [core :refer [release]]
             [utils :refer [dragan-says-ex]]])
  (:import [java.nio.file Files Path CopyOption FileVisitOption]
           java.nio.file.attribute.FileAttribute))

(defn delete [path]
  (let [options (make-array FileVisitOption 0)]
    (doseq [path (reverse (iterator-seq (.iterator (Files/walk path options))))]
      (Files/deleteIfExists path))))

(defn copy-philox [^Path path]
  (let [random123-path (.resolve path "Random123")
        attributes (make-array FileAttribute 0)
        options (make-array CopyOption 0)]
    (try
      (Files/createDirectories (.resolve random123-path "features/dummy") attributes)
      (doseq [include-name ["philox.h" "array.h" "features/compilerfeatures.h"
                            "features/openclfeatures.h"]]
        (Files/copy
         (ClassLoader/getSystemResourceAsStream
          (format "uncomplicate/bayadera/internal/include/Random123/%s" include-name))
         (.resolve random123-path ^String include-name)
         ^"[Ljava.nio.file.CopyOption;" options))
      (catch Exception e
        (delete path)
        (throw e)))))

(defn create-tmp-dir []
  (java.nio.file.Files/createTempDirectory "uncomplicate_" (make-array FileAttribute 0)))

(defmacro with-philox [path & body]
  `(try
     (copy-philox ~path)
     (do ~@body)
     (finally
       (delete ~path))))

(defn release-deref [ds]
  (if (sequential? ds)
    (doseq [d ds]
      (when (realized? d) (release @d)))
    (when (realized? ds) (release ds))))
