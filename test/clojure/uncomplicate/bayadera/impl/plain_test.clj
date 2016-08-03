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
      (variance x) => (roughly 9.16666 0.01)
      (sd x) => (roughly 3.02765 0.0001))))

(defn ge-matrix-location-test [factory]
  (with-release [a (create-ge-matrix factory 3 5 (range 15))
                 a02 (create-ge-matrix factory 0 2)]
     (facts
      "GE Matrix as a Location"
      (every? #(Double/isNaN %) (mean a02)) => true
      (variance a02) => (create-vector factory [0 0])
      (sd a02) => (create-vector factory [0 0])
      (mean a) => (create-vector factory [1 4 7 10 13])
      (sum (variance a)) => (roughly (sum (create-vector factory [1 1 1 1 1])))
      (sum (sd a)) => (roughly (sum (create-vector factory [1 1 1 1 1]))))))

(vector-location-test cblas-single)
(vector-location-test cblas-double)

(ge-matrix-location-test cblas-single)
(ge-matrix-location-test cblas-double)
