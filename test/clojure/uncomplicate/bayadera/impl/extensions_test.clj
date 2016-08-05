(ns uncomplicate.bayadera.impl.extensions-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :refer [create-vector create-ge-matrix sum]]
            [uncomplicate.neanderthal.impl.cblas :refer [cblas-single cblas-double]]
            [uncomplicate.bayadera.impl.extensions :refer :all]
            [uncomplicate.bayadera.core :refer [mean variance sd]]))

(defn vector-test [factory]
  (with-release [x (create-vector factory (range 10))
                 x0 (create-vector factory 0)]
    (facts
     "Vector as a Location and Spread"
     (Double/isNaN (mean x0)) => true
     (variance x0) => 0.0
     (sd x0) => 0.0
     (mean x) => 4.5
     (variance x) => (roughly 8.25 0.0001)
     (sd x) => (roughly 2.87223 0.0001))))

(vector-test cblas-single)
(vector-test cblas-double)

(defn ge-matrix-test [factory]
  (with-release [a (create-ge-matrix factory 3 5 (range 15))
                 a20 (create-ge-matrix factory 2 0)]
    (facts
     "GE Matrix as a Location and Spread"
     (every? #(Double/isNaN %) (mean a20)) => true
     (variance a20) => (create-vector factory [0 0])
     (sd a20) => (create-vector factory [0 0])
     (mean a) => (create-vector factory [6 7 8])
     (sum (variance a)) =>  (roughly 54)
     (sum (sd a)) => (roughly 12.7278))))

(ge-matrix-test cblas-single)
(ge-matrix-test cblas-double)

(defn sequence-test [constructor]
  (let [x (apply constructor (range 10))
        x0 (apply constructor [])]
    (facts
     "Sequence/collection as a Location and Spread"
     (Double/isNaN (mean x0)) => true
     (variance x0) => 0.0
     (sd x0) => 0.0
     (mean x) => 4.5
     (variance x) => (roughly 8.25 0.0001)
     (sd x) => (roughly 2.87223 0.0001))))

(sequence-test list)
(sequence-test vector)

(defn nested-sequence-test [constructor]
  (with-release [a (apply constructor [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]])
                 a02 (apply constructor [[] []])]
    (facts
     "Nested sequence/vector as a Location and Spread"
     (Double/isNaN (mean a02)) => true
     (variance a02) => 0.0
     (sd a02) => 0.0
     (mean a) =>  [6.0 7.0 8.0]
     (apply + (variance a)) => (roughly 54)
     (apply + (sd a)) => (roughly 12.7278))))

(nested-sequence-test list)
(nested-sequence-test vector)
