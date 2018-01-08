;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.math-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [core :refer [sum vctr raw copy! axpy! asum]]
             [native :refer [native-float native-double]]]
            [uncomplicate.bayadera
             [math :as m]
             [vect-math :as vm]]))

(defn test-factorial [factory]
  (facts
   "factorial tests."
   (with-release [x (vctr factory (range 10))
                  y (vctr factory (range 0.1 1 0.1))
                  work1 (raw x)
                  work2 (raw x)]
     (vm/log-factorial! x work1) => (fmap! m/log-factorial (copy! x work2))
     (vm/log-factorial! (copy! x work1)) => (fmap! m/log-factorial (copy! x work2))
     (vm/factorial! x work1) => (fmap! m/factorial (copy! x work2))
     (vm/factorial! (copy! x work1)) => (fmap! m/factorial (copy! x work2)))))

(defn test-beta [factory]
  (facts
   "beta tests."
   (with-release [a (vctr factory (range 0.1 1 0.1))
                  b (vctr factory (reverse (range 0.1 1 0.1)))
                  work1 (raw a)
                  work2 (raw a)
                  work3 (raw a)
                  work4 (raw a)]
     (asum (axpy! -1.0 (vm/log-beta! (copy! a work1) (copy! b work2) work3)
                  (fmap! m/log-beta (copy! a work4) b))) => (roughly 0 0.00001)
     (asum (axpy! -1.0 (vm/log-beta! (copy! a work1) (copy! b work2))
                  (fmap! m/log-beta (copy! a work4) b))) => (roughly 0 0.00001)
     (asum (axpy! -1.0 (vm/beta! (copy! a work1) (copy! b work2) work3)
                  (fmap! m/beta (copy! a work4) b)))  => (roughly 0 0.00001)
     (asum (axpy! -1.0 (vm/beta! (copy! a work1) (copy! b work2))
                  (fmap! m/beta (copy! a work4) b))) => (roughly 0 0.00001))))

(defn test-binco [factory]
  (facts
   "binco tests."
   (with-release [n (vctr factory (range 3 12))
                  k (vctr factory (range 1 10))
                  work1 (raw n)
                  work2 (raw n)
                  work3 (raw n)
                  work4 (raw n)]
     (asum (axpy! -1.0 (vm/log-binco! (copy! n work1) (copy! k work2) work3)
                  (fmap! m/log-binco (copy! n work4) k))) => (roughly 0 0.00001)
     (asum (axpy! -1.0 (vm/log-binco! (copy! n work1) (copy! k work2))
                  (fmap! m/log-binco (copy! n work4) k))) => (roughly 0 0.00001)
     (asum (axpy! -1.0 (vm/binco! (copy! n work1) (copy! k work2) work3)
                  (fmap! m/binco (copy! n work4) k)))  => (roughly 0 0.0001)
     (asum (axpy! -1.0 (vm/binco! (copy! n work1) (copy! k work2))
                  (fmap! m/binco (copy! n work4) k))) => (roughly 0 0.0001))))

(defn test-vect-math [factory]
  (test-factorial factory)
  (test-beta factory)
  (test-binco factory))

(test-vect-math native-float)
(test-vect-math native-double)
