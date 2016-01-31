(ns uncomplicate.bayadera.examples.fapsp.ch04-distributions-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core :refer [with-release]]
            [uncomplicate.neanderthal
             [math :refer [exp sqrt]]
             [native :refer [dv]]]
            [uncomplicate.bayadera.distributions :refer :all]))

;; Examples from the book by Oliver Ibe - Fundamentals of Applied Probability
;; and Random Processes, Chapter 4.

;; ================= Binomial Distribution =====================
(facts
 "Problem 4.1"
 (binomial-cdf 4 1/6 1) => (roughly 0.86806))

(facts
 "Problem 4.4"
 (- 1 (binomial-cdf 4 30/100 2)) => (roughly 0.0837)
 (binomial-cdf 4 30/100 0) => (roughly 0.2401))

(facts
 "Problem 4.5"
 (- 1 (binomial-cdf 6 1/3 3)) => (roughly 73/729))

(facts
 "Problem 4.6"
 (- 1 (binomial-cdf 100 0.001 2)) => (roughly 0.00015038))

(facts
 "Problem 4.7"
 (- 1 (binomial-cdf 4 10/100 3)) => (roughly 0.0001)
 (binomial-pmf 4 10/100 3) => (roughly 0.0036))

(facts
 "Problem 4.8"
 (binomial-cdf 8 0.10 1) => (roughly 0.8131)
 (binomial-pmf 8 0.10 1) => (roughly 0.3826)
 (binomial-mean 8 0.10) => 0.8)

(facts
 "Problem 4.9"
 (binomial-mean 4 25/100) => 1.0
 (binomial-variance 4 25/100) => 0.75)

(facts
 "Problem 4.11"
 (- 1(binomial-cdf 8 0.1 1)) => (roughly 0.1869))

(facts
 "Problem 4.12"
 (- 1 (binomial-cdf 12 0.7 9)) => (roughly 0.2528))

(facts
 "Problem 4.13"
 (- 1 (binomial-cdf 4 0.1 1)) => (roughly 0.0523)
 (- 1 (binomial-pmf 20 0.1 0)) => (roughly 0.8784))

(facts
 "Problem 4.15"
 (binomial-pmf 10 40/100 2) => (roughly 0.1209))

(facts
 "Problem 4.16"
 (binomial-pmf 5 0.4 2) => (roughly 0.3456)
 (- 1 (binomial-cdf 5 0.4 2)) => (roughly 0.31744)
 (binomial-mean 5 0.4) => 2.0)

(facts
 "Problem 4.17"
 (binomial-pmf 10 2/8 4) => (roughly 0.146))

;; ================== Geometric Distribution =================

(facts
 "Problem 4.20"
 (geometric-pmf 1/6 4) => (roughly 0.0964))

(facts
 "Problem 4.21"
 (* (geometric-normalizer 6 1/6)
    (double (reduce (fn ^double [^double r ^double k]
                      (+ r (* k (geometric-pmf 1/6 k))))
                    0.0 (range 1 7))))
 => (roughly 2.9788))

(facts
 "Problem 4.23"
 (geometric-pmf 20/100 10) => (roughly 0.02684)
 (- 1 (geometric-cdf 20/100 9)) => (roughly 0.1342))

(facts
 "Problem 4.24"
 (let [p (/ (- 2200 2000) (- 2200 800))]
   p => 1/7
   (geometric-mean p) => (roughly 7)))

;; ==================== Pascal Distribution ====================

(facts
 "Problem 4.25"
 (pascal-pmf 1 0.05 3) =>  0.045125
 (pascal-pmf 2 0.05 5) => (roughly 0.00857)
 (binomial-pmf 8 0.05 0) => (roughly 0.6634))

(facts
 "Problem 4.26"
 (- 1 (pascal-cdf 3 20/100 5)) => (roughly 0.00672))

(facts
 "Problem 4.27"
 (pascal-pmf 3 20/100 10) => (roughly 0.0604)
 (- 1 (pascal-cdf 3 20/100 9)) => (roughly 0.08564))

(facts
 "Problem 4.29"
 (let [p (* 0.75 0.5)]
   (geometric-pmf p 3) => (roughly 0.1465)
   (pascal-pmf 2 p 5) => (roughly 0.1373)
   (geometric-pmf p 3) => (roughly 0.1465)
   (/ (pascal-pmf 2 p 5) (- 1 (pascal-cdf 2 p 2))) => (roughly 0.1598)))

(facts
 "Problem 4.30"
 (pascal-pmf 3 0.3 8) => (roughly 0.0953))

(facts
 "Problem 4.31"
 (pascal-pmf 3 0.6 6) => (roughly 0.13824)
 (binomial-pmf 12 0.6 8) => (roughly 0.2128))

;; ==================== Hyper-geometric Distribution ================

(facts
 "Problem 4.32"
 (hypergeometric-pmf 10 4 5 2) => (roughly 0.4762))

(facts
 "Problem 4.33"
 (hypergeometric-pmf 100 2 20 2) => (roughly 0.03838)
 (hypergeometric-pmf 100 2 20 0) => (roughly 0.63838))

(facts
 "Problem 4.34"
 (- 1 (hypergeometric-cdf 12 6 8 3)) => (roughly 0.7273)
 (- (hypergeometric-cdf 12 6 8 6) (hypergeometric-cdf 12 6 8 3)) => (roughly 0.7273))

(facts
 "Problem 4.35"
 (hypergeometric-pmf 30 12 15 8) => (roughly 0.10155)
 (hypergeometric-mean 30 18 15) => 9.0)

(facts
 "Problem 4.36"
 (hypergeometric-pmf 22 10 4 2) => (roughly 0.4060))

;; ==================== Poisson Distribution ================

(facts
 "Problem 4.37"
 (- 1 (poisson-pmf 50/60 0)) => (roughly 0.5654))

(facts
 "Problem 4.38"
 (poisson-pmf 7 0) => (roughly 0.0009 0.00002)
 (poisson-cdf 7 3) => (roughly 0.0818))

(facts
 "Problem 4.39"
 (poisson-cdf 10 3) => (roughly 0.01034)
 (- 1 (poisson-cdf 10 1)) => (roughly 0.9995))

(facts
 "Problem 4.40"
 (- 1 (poisson-cdf 4 3)) => (roughly 0.5665))

(facts
 "Problem 4.41"
 (poisson-pmf 4 0) => (roughly 0.0183)
 (- 1 (poisson-cdf 4 2)) => (roughly 0.7619))

(facts
 "Problem 4.42"
 (poisson-pmf 3 7) => (roughly 0.0216)
 (poisson-cdf 3 3) => (roughly 0.6472)
 (poisson-pmf 3 0) => (roughly 0.0498))

;; ==================== Exponential Distribution ================

(facts
 "Problem 4.43"
 (exponential-mean 4) => 0.25
 (exponential-cdf 4 1) => (roughly 0.9817))

(facts
 "Problem 4.44"
 (exponential-mean 0.25) => 4.0
 (exponential-variance 0.25) => 16.0
 (- 1 (exponential-cdf 0.25 2)) => (roughly 0.6065)
 (exponential-cdf 0.25 (- 4 2)) => (roughly 0.3935))

(facts
 "Problem 4.45"
 (exponential-mean 2) => 0.5
 (exponential-variance 2) => 0.25
 (- 1 (exponential-cdf 2 1)) => (roughly 0.1353))

(facts
 "Problem 4.46"
 (- 1 (exponential-cdf 0.1 15)) => (roughly 0.2231))

(facts
 "Problem 4.47"
 (- 1 (exponential-cdf 0.2 (- 10 8))) => (roughly 0.6703)
 (exponential-mean 0.2) => 5.0)

(facts
 "Problem 4.48"
 (- 1 (exponential-cdf 1/30 120)) => (roughly 0.0183))

(facts
 "Problem 4.49"
 (exponential-cdf 1/3 2) => (roughly 0.4866)
 (- 1 (exponential-cdf 1/3 4)) => (roughly 0.2636)
 (- 1 (exponential-cdf 1/3 4)) => (roughly 0.2636)
 (exponential-mean 1/3) => 3.0)

(facts
 "Problem 4.50"
 (- 1 (exponential-cdf 1/4 2)) => (roughly 0.6065)
 (- 1 (exponential-cdf 1/4 5)) => (roughly 0.2865))

(facts
 "Problem 4.51"
 (exponential-mean 0.02) => 50.0
 (let [t<40 (exponential-cdf 0.02 40)
       pf (fn ^double [^double t]
            (/ (exponential-cdf 0.02 t) t<40))])
 (- (exponential-cdf 0.02 60) (exponential-cdf 0.02 40)) => (roughly 0.1481))

;; ==================== Erlang Distribution ================

(facts
 "Example 4.19"
 (- 1 (erlang-cdf 1/10 (- 6 2) 20)) => (roughly 0.8571))

(facts
 "Problem 4.54"
 (erlang-mean 2 3) => 1.5
 (erlang-cdf 2 3 6)  => (roughly 0.9997))

(facts
 "Problem 4.55"
 (erlang-mean 5 5) => 1.0
 (- 1 (erlang-cdf 5 5 1)) => (roughly 0.4405))

;; ==================== Uniform Distribution ================

(facts
 "Problem 4.56"
 (uniform-mean 10 30) => 20.0
 (uniform-variance 10 30) => (roughly 33.33))

(facts
 "Problem 4.57"
 (- (uniform-cdf 0 10 (uniform-mean 0 10))
    (uniform-cdf 0 10 (sqrt (uniform-variance 0 10)))) => (roughly 0.2113))

(facts
 "Problem 4.58"
 (uniform-mean 3 15) => 9.0
 (uniform-variance 3 15) => 12.0
 (- (uniform-cdf 3 15 10) (uniform-cdf 3 15 5)) => (roughly 5/12)
 (uniform-cdf 3 15 6) => 0.25)

(facts
 "Problem 4.59"
 (+ (- (uniform-cdf 0 30 30) (uniform-cdf 0 30 25))
    (- (uniform-cdf 0 30 15) (uniform-cdf 0 30 10))) => (roughly 1/3)
 (+ (- (uniform-cdf 0 30 20) (uniform-cdf 0 30 15))
    (- (uniform-cdf 0 30 5) (uniform-cdf 0 30 0))) => (roughly 1/3))

(facts
 "Problem 4.60"
 (uniform-mean 2 6) => 4.0
 (uniform-cdf 2 6 1) => 0.0
 (- (uniform-cdf 2 6 5) (uniform-cdf 2 6 3)) => 0.5)

;; ==================== Gaussian (Normal) Distribution ================

(facts
 "Problem 4.61"
 (* 200 (- (gaussian-cdf 140 10 145) (gaussian-cdf 140 10 110))) => (roughly 138.04)
 (* 200 (gaussian-cdf 140 10 120)) => (roughly 4.55)
 (* 200 (- 1 (gaussian-cdf 140 10 170))) => (roughly 0.27))

(facts
 "Problem 4.62"
 (- 1 (gaussian-cdf 70 10 50)) => (roughly 0.9772)
 (gaussian-cdf 70 10 60) => (roughly 0.1587)
 (- (gaussian-cdf 70 10 90) (gaussian-cdf 70 10 60)) => (roughly 0.8186))

(facts
 "Problem 4.63"
 (- (binomial-cdf 12 0.5 8) (binomial-cdf 12 0.5 3)) => (roughly 0.8540))

(facts
 "Problem 4.66"
 (- (gaussian-cdf 40 (sqrt 16) 48) (gaussian-cdf 40 (sqrt 16) 30))
 => (roughly 0.9710))

;; ==================== Multinomial Distribution ================

(facts
 "Example 5.15"
 (multinomial-pmf (dv 1/6 1/6 4/6) (dv 2 1 4)) => (roughly 0.0960))

(facts
 "Example 5.16"
 (multinomial-pmf (dv 50/100 20/100 30/100) (dv 6 2 2)) => (roughly 0.0709))

(facts
 "Problem 5.23"
 (multinomial-pmf (dv 16/40 24/40) (dv 9 11)) => (roughly 0.1597))

(facts
 "Problem 5.24"
 (multinomial-pmf (dv 10/40 16/40 14/40) (dv 5 9 6)) => (roughly 0.0365))

(facts
 "Problem 5.25"
 (multinomial-pmf (dv 20/100 50/100 20/100 10/100) (dv 6 4 1 1))
 => (roughly 0.0022176)
 (multinomial-pmf (dv 20/100 50/100 20/100 10/100) (dv 6 4 2 0))
 => (roughly 0.0022176)
 (multinomial-pmf (dv 20/100 50/100 30/100) (dv 4 3 5)) => (roughly 0.01347)
 (binomial-pmf 12 10/100 4) => (roughly 0.0213)
 (multinomial-pmf (dv 10/100 90/100) (dv 4 8)) => (binomial-pmf 12 10/100 4)
 (multinomial-pmf (dv 10/100 90/100) (dv 0 12)) => (roughly 0.2824))

(facts
 "Problem 5.26"
 (multinomial-pmf (dv 50/100 35/100 10/100 5/100) (dv 30 5 3 2))
 => (roughly 0.0000261217)
 (multinomial-pmf (dv 50/100 35/100 15/100) (dv 30 4 6)) => (roughly 0.00002833)
 (multinomial-pmf (dv 95/100 5/100) (dv 40 0)) => (roughly 0.1285)
 (multinomial-pmf (dv 85/100 10/100 5/100) (dv 40 0 0)) => (roughly 0.001502)
 (multinomial-pmf (dv 85/100 15/100) (dv 40 0))
 => (multinomial-pmf (dv 85/100 10/100 5/100) (dv 40 0 0))
 (float (multinomial-pmf (dv 85/100 15/100) (dv 40 0)))
 => (float (binomial-pmf 40 15/100 0)))

(facts
 "Problem 5.27"
 (multinomial-pmf (dv 8/20 7/20 5/20) (dv 4 4 2)) => (roughly 0.0756)
 (multinomial-pmf (dv 8/20 12/20) (dv 5 5)) => (roughly 0.2006)
 (float (multinomial-pmf (dv 8/20 12/20) (dv 5 5)))
 => (float (binomial-pmf 10 8/20 5)))
