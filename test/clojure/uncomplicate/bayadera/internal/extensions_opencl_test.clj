;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.bayadera.internal.extensions-opencl-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl
             [core :refer [*command-queue* with-default-1]]]
            [uncomplicate.neanderthal.opencl :refer [with-engine opencl-float opencl-double *opencl-factory*]]
            [uncomplicate.bayadera.internal.extensions-test :refer [vector-test ge-matrix-test]]))

(with-default-1
  (with-engine opencl-float *command-queue*
    (vector-test *opencl-factory*)
    (ge-matrix-test *opencl-factory*))
  (with-engine opencl-double *command-queue*
    (vector-test *opencl-factory*)
    (ge-matrix-test *opencl-factory*)))
