;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.vect-math
  (:require [uncomplicate.commons.core :refer [with-release let-release]]
            [uncomplicate.neanderthal
             [core :refer [axpby! axpy! raw copy copy!]]
             [vect-math :refer [linear-frac linear-frac! lgamma! gamma! exp!]]]))

(defn log-factorial!
  ([x res]
   (lgamma! (linear-frac! 1.0 x 1.0 res)))
  ([x]
   (lgamma! (linear-frac! x 1.0))))

(defn log-factorial [x]
  (let-release [res (linear-frac x 1.0)]
    (lgamma! res)))

(defn factorial!
  ([x res]
   (gamma! (linear-frac! 1.0 x 1.0 res)))
  ([x]
   (gamma! (linear-frac! x 1.0))))

(defn factorial
  [x]
  (let-release [res (linear-frac x 1.0)]
    (gamma! res)))

(defn log-beta!
  ([a b res]
   (lgamma! (axpy! a (copy! b res)))
   (axpby! 1.0 (axpy! (lgamma! a) (lgamma! b)) -1.0 res))
  ([a b]
   (with-release [work (copy b)]
     (axpy! -1.0 (lgamma! (axpy! a work)) (axpy! (lgamma! b) (lgamma! a))))))

(defn log-beta [a b]
  (let-release [res (raw a)]
    (with-release [a-copy (copy a)
                   b-copy (copy b)]
      (log-beta! a-copy b-copy res))))

(defn beta!
  ([a b res]
   (exp! (log-beta! a b res)))
  ([a b]
   (exp! (log-beta! a b))))

(defn beta [a b]
  (exp! (log-beta a b)))

(defn log-binco!
  ([n k res]
   (log-factorial! (axpby! 1.0 n -1.0 (copy! k res)))
   (axpby! 1.0 (log-factorial! n) -1.0 (axpy! (log-factorial! k) res)))
  ([n k]
   (with-release [work (copy k)]
     (axpy! -1.0 (axpy! (log-factorial! k) (log-factorial! (axpby! 1.0 n -1.0 work)))
            (log-factorial! n)))))

(defn log-binco [a b]
  (let-release [res (raw a)]
    (with-release [a-copy (copy a)
                   b-copy (copy b)]
      (log-binco! a-copy b-copy res))))

(defn binco!
  ([a b res]
   (exp! (log-binco! a b res)))
  ([a b]
   (exp! (log-binco! a b))))

(defn binco [a b]
  (exp! (log-binco a b)))
