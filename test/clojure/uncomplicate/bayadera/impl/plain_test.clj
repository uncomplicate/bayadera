(ns uncomplicate.bayadera.impl.plain-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :refer [create-vector create-ge-matrix sum]]
            [uncomplicate.neanderthal.impl.cblas :refer [cblas-single cblas-double]]
            [uncomplicate.bayadera.impl.plain :refer :all]
            [uncomplicate.bayadera.core :refer [mean variance sd]]))

(defn vector-location-test [factory]
  (with-release [x (create-vector factory (range 10))
                 x0 (create-vector factory 0)]
     (facts
      "Vector as a Location"
      (Double/isNaN (mean x0)) => true
      (variance x0) => 0.0
      (sd x0) => 0.0
      (mean x) => 4.5
      (variance x) => (roughly 8.24999 0.0001)
      (sd x) => (roughly 2.87223 0.0001))))

(defn ge-matrix-location-test [factory]
  (with-release [a (create-ge-matrix factory 3 5 (range 15))
                 a20 (create-ge-matrix factory 2 0)]
     (facts
      "GE Matrix as a Location"
      (every? #(Double/isNaN %) (mean a20)) => true
      (variance a20) => (create-vector factory [0 0])
      (sd a20) => (create-vector factory [0 0])
      (mean a) => (create-vector factory [6 7 8])
      (sum (variance a)) =>  (roughly 54)
      (sum (sd a)) => (roughly 12.7278))))

(vector-location-test cblas-single)
(vector-location-test cblas-double)

(ge-matrix-location-test cblas-single)
(ge-matrix-location-test cblas-double)
