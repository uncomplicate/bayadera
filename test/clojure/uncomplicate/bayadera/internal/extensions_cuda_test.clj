;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.bayadera.internal.extensions-cuda-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.cuda
             :refer [with-engine cuda-float cuda-double *cuda-factory* cuda-handle]]
            [uncomplicate.bayadera.internal.extensions-test :refer [vector-test ge-matrix-test]]))

(with-release [handle (cuda-handle)]
  (with-engine cuda-float handle
    (vector-test *cuda-factory*)
    (ge-matrix-test *cuda-factory*))
  (with-engine cuda-double handle
    (vector-test *cuda-factory*)
    (ge-matrix-test *cuda-factory*)))
