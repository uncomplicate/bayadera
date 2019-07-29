;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.cuda
  (:require [uncomplicate.clojurecuda.core :refer [with-default current-context default-stream]]
            [uncomplicate.bayadera
             [core :refer [with-bayadera *bayadera-factory*]]
             [library :refer [with-library]]]
            [uncomplicate.bayadera.internal.device
             [models :as models]
             [nvidia-gtx :refer [gtx-bayadera-factory]]]))

(def source-library (models/source-library "uncomplicate/bayadera/internal/device/cuda/%s.cu"))

(def device-library (partial models/device-library source-library))

(defmacro with-default-library [factory & body]
  `(with-library (device-library ~factory)
     ~@body))

(defmacro with-default-bayadera [& body]
  `(with-default
     (with-bayadera gtx-bayadera-factory [(current-context) default-stream]
       (with-default-library *bayadera-factory*
         ~@body))))
