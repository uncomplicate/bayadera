(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.distributions-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.bayadera.distributions :refer :all]))

(facts
 "Binomial Distribution + see examples/fpsap"
 (binomial-params 7 0.3) => [7 0.3]
 (binomial-params -1 7) => nil
 (binomial-check-nk 9 0) => true
 (binomial-check-nk 9 10) => false
 (binomial-mode 4 0.8) => 4)

(facts
 "Beta Distribution"
 (beta-pdf 2.3 3.3 0.14) => (roughly 1.079)
 (beta-cdf 2.3 3.3 0.18) => (roughly 0.122)

 (beta-cdf 4.343 6.5454 0.0) => 0.0
 (beta-cdf 4.232 1.232 1.0) => 1.0)

(facts
 "Gamma Distribution"
 (gamma-pdf 3.3 3 5.77) => (roughly (erlang-pdf (/ 1 3.3) 3 5.77))
 (gamma-cdf 4.57 5 13.33) => (erlang-cdf (/ 1 4.57) 5 13.33))

(facts
 "Student's t distribution"
 (t-pdf 2.45 2.34) => (roughly 0.047657)
 (- (t-cdf 2.45 3.12) (t-cdf 2.45 2.34)) => (roughly 0.02602))
