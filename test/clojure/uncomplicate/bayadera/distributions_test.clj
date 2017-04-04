;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.distributions-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.neanderthal
             [core :refer [sum]]
             [native :refer [fv]]]
            [uncomplicate.bayadera.distributions :refer :all]))

(facts
 "Binomial Distribution + see examples/fapsp"
 (binomial-params 7 0.3) => [7 0.3]
 (binomial-params -1 7) => nil
 (binomial-check 9 0.4 0) => true
 (binomial-check 9 0.34 10) => false
 (binomial-mode 4 0.8) => 4
 (binomial-median 4 0.8) => 3)

(facts
 "Binomial (Bernoulli) Likelihood"
 (binomial-lik-params -1 7) => nil)

(facts
 "Geometric Distribution + see examples/fapsp"
 (geometric-check 0.3 3) => true
 (geometric-check 0.2 0) => false
 (geometric-params 0.4) => [0.4]
 (geometric-params 1.2) => nil
 (geometric-median 0.24) => 4
 (geometric-mode 0.33) => 1)

(facts
 "Pascal (Negative Binomial) Distribution + see examples/fapsp"
 (pascal-check 3 0.3) => true
 (pascal-check 0 0) => false
 (pascal-params 3 0.4) => [3 0.4]
 (pascal-params 3 1.2) => nil
 (pascal-mode 5 0.24) => 17)

(facts
 "Hypergeometric Distribution + see examples/fapsp"
 (hypergeometric-check 10 4 3 3) => true
 (hypergeometric-check 10 14 3 3) => false
 (hypergeometric-params 10 4 3) => [10 4 3]
 (hypergeometric-params 3 10 2) => nil
 (hypergeometric-mode 10 4 3) => 4)

(facts
 "Poisson Distribution + see examples/fapsp"
 (poisson-check 10 4) => true
 (poisson-check 0 14) => false
 (poisson-params 10) => [10.0]
 (poisson-params -1.0) => nil
 (poisson-mode 10.3) => 10)

(facts
 "Exponential Distribution + see examples/fapsp"
 (exponential-check 0.3 5) => true
 (exponential-check 0 4) => false
 (exponential-check 4 0) => false
 (exponential-params 0.2) => [0.2 -1.6094379124341003]
 (exponential-params -0.2) => nil
 (exponential-mode 0.4) => 0.0
 (exponential-median 0.4) => 1.732867951399863)

(facts
 "Erlang Distribution + see examples/fapsp"
 (erlang-check 0.3 5) => true
 (erlang-check 0 4) => false
 (erlang-check 4 0) => false
 (erlang-params 0.2 7) => [0.2 7 -17.845316599048804]
 (erlang-params 2 0.7) => nil
 (erlang-mode 0.4 2) => 2.5
 (Double/isNaN (erlang-mode 0.4 0)) => true
 (erlang-median 2.0 7) => 3.6732673267326734)

(facts
 "Uniform Distribution + see examples/fapsp"
 (uniform-check 0.3 5) => true
 (uniform-check 4 0) => false
 (uniform-params 0.2 7) => [0.2 7.0]
 (uniform-params 4 2) => nil
 (uniform-mean 0.4 2) => 1.2
 (uniform-mode 0.4 2) => 1.2
 (uniform-median 0.4 2) => 1.2)

(facts
 "Gaussian Distribution + see examples/fapsp"
 (gaussian-check 0.3 5) => true
 (gaussian-check 4 0) => false
 (gaussian-params 0.2 7) => [0.2 7.0]
 (gaussian-params 0.2 -2) => nil
 (gaussian-mean 0.4) => 0.4
 (gaussian-mode 0.4) => 0.4
 (gaussian-median 0.4) => 0.4)

(facts
 "Student's t Distribution"
 (t-check 0.3) => true
 (t-check 0) => false
 (t-check 2.0 3.0 0.5) => true
 (t-check 2.0 3.0 -0.5) => false
 (t-params 0.2) => [0.2 0.0 1.0 -1.6221247803726204]
 (t-params 0.2 -2 -2.0) => nil
 (t-mean 2.3 0.4) => 0.4
 (t-mode 2.3 0.4) => 0.4
 (t-median 2.3 0.4) => 0.4
 (t-variance 2.3 3.1) => 73.67666666666672
 (t-pdf 2.45 2.34) => (roughly 0.047657)
 (- (t-cdf 2.45 3.12) (t-cdf 2.45 2.34)) => (roughly 0.02602))

(facts
 "Beta Distribution"
 (beta-check 3 4) => true
 (beta-check -2 0) => false
 (beta-params 2.3 3.3) => [2.3 3.3 2.978625424679165]
 (beta-params 2.3 -3.3) => nil
 (beta-pdf 2.3 3.3 0.14) => (roughly 1.079)
 (beta-cdf 2.3 3.3 0.18) => (roughly 0.122)
 (beta-cdf 4.343 6.5454 0.0) => 0.0
 (beta-cdf 4.232 1.232 1.0) => 1.0)

(facts
 "Gamma Distribution"
 (gamma-check 3 4) => true
 (gamma-check -2 0) => false
 (gamma-params 2.3 3.3) => [2.3 3.3 -3.7356986835805768]
 (gamma-params 2.3 -3.3) => nil
 (gamma-pdf 3.3 3 5.77) => (roughly (erlang-pdf (/ 1 3.3) 3 5.77))
 (gamma-cdf 4.57 5 13.33) => (erlang-cdf (/ 1 4.57) 5 13.33))

(facts
 "Multinomial Distribution + see examples/fapsp"
 (multinomial-check (fv 0.2 0.3 0.5)) => true
 (multinomial-check (fv 0.2 0.3 0.6)) => false
 (multinomial-check (fv 0.2 0.3 3)) => false
 (multinomial-check (fv 0.2 0.3 0.5) (fv 1 2 3)) => true
 (multinomial-check (fv 0.2 0.3 5) (fv 1 2 3)) => false
 (multinomial-check (fv 0.2 0.3 0.5) (fv 1 2 -3)) => false
 (multinomial-mean (fv 0.2 0.3 0.5) 10) => (fv 2 3 5)
 (sum (multinomial-variance (fv 0.2 0.3 0.5) 10)) => (roughly (sum (fv 1.6 2.1 2.5))))
