;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.util-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.neanderthal
             [core :refer [scal! axpy!]]
             [real :refer [entry asum nrm2]]
             [native :refer [fv fge]]]
            [uncomplicate.bayadera.internal.protocols :refer [->Histogram]]
            [uncomplicate.bayadera.util :refer :all]))

(with-release [wgs 79
               l 1
               u 7
               step (/ (- u l) wgs)
               tlimits (fv l u)
               tpdf (fv [0.02 0.01 0.04 0.05 0.07 0.01 0.2  0.1  0.3  0.1  0.02 0.01 0.01 0.01 0.01
                         0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01
                         0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01
                         0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01
                         0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01])
               tbin-rank (fv (op [ 8 6 7 9 4 3 2 0 10 1 5] (range 11 79)))
               tpdf-asum (asum tpdf)]
  (facts
   "hdi-rank-count tests."
   (hdi-rank-count 0.1 tbin-rank tpdf) => 1
   (hdi-rank-count (/ 0.3 tpdf-asum) tbin-rank tpdf) => 1
   (hdi-rank-count 0.3 tbin-rank tpdf) => 2
   (hdi-rank-count (/ 0.7 tpdf-asum) tbin-rank tpdf) => 4
   (hdi-rank-count (/ 0.86 tpdf-asum) tbin-rank tpdf) => 7
   (hdi-rank-count (/ 0.9 tpdf-asum) tbin-rank tpdf) => 9
   (hdi-rank-count (/ 0.95 tpdf-asum) tbin-rank tpdf) => 14)

  (facts
   "hdi-bins tests."
   (hdi-bins tbin-rank 1) => [8.0 8.0]
   (hdi-bins tbin-rank 2) => [6.0 6.0 8.0 8.0]
   (hdi-bins tbin-rank 5) => [4.0 4.0 6.0 9.0]
   (hdi-bins tbin-rank 6) => [3.0 4.0 6.0 9.0]
   (hdi-bins tbin-rank 12) => [0.0 11.0]
   (hdi-bins tbin-rank 15) => [0.0 14.0]

   (facts
    "hdi-regions tests."
    (nrm2 (axpy! -1 (hdi-regions tlimits tbin-rank 1) (fge 2 1 [1.61 1.68]))) => (roughly 0 0.005)
    (nrm2 (axpy! -1 (hdi-regions tlimits tbin-rank 2) (fge 2 2 [1.46 1.53 1.61 1.68]))) => (roughly 0 0.007)
    (nrm2 (axpy! -1 (hdi-regions tlimits tbin-rank 5) (fge 2 2 [1.30 1.38 1.46 1.76]))) => (roughly 0 0.006)
    (nrm2 (axpy! -1 (hdi-regions tlimits tbin-rank 6) (fge 2 2 [1.23 1.38 1.46 1.76]))) => (roughly 0 0.006)
    (nrm2 (axpy! -1 (hdi-regions tlimits tbin-rank 12) (fge 2 1 [1.00 1.91]))) => (roughly 0 0.002))))
