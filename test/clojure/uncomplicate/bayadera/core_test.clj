(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.core-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [math :refer [log exp sqrt]]
             [core :refer [dim nrm2 copy dot scal! transfer row ge col axpy! mrows ncols]]
             [real :refer [sum entry]]
             [native :refer [fv]]]
            [uncomplicate.bayadera
             [core :refer :all]
             [distributions :refer [beta-pdf binomial-lik-params]]
             [mcmc :refer [mix!]]
             [math :refer [log-beta]]]
            [uncomplicate.bayadera.internal
             [protocols :as p]]))

(defmacro roughly100 [exp]
  `(let [v# (double ~exp)]
     (roughly v# (/ v# 100.0)) ))

(defn test-uniform [factory]
  (let [a 100.0
        b 133.4]
    (facts
     "Core functions for uniform distribution."
     (with-release [dist (uniform factory a b)
                    uniform-sampler (sampler dist)
                    cl-sample (dataset factory (sample uniform-sampler))
                    cl-pdf (pdf dist cl-sample)]
       (dim (mean cl-sample)) => 1
       (dim (sd cl-sample)) => 1
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (sd cl-sample) 0) => (roughly100 (sd dist))
       (/ (sum cl-pdf) (dim cl-pdf)) => (roughly (/ 1.0 (- b a)))))))

(defn test-gaussian [factory]
  (let [mu 200.0
        sigma 10.0]
    (facts
     "Core functions for gaussian distribution."
     (with-release [dist (gaussian factory mu sigma)
                    gaussian-sampler (sampler dist)
                    cl-sample (dataset factory (sample gaussian-sampler))]
       (mean dist) => mu
       (sd dist) => sigma
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (variance cl-sample) 0)  => (roughly 100.0 2.0)
       (entry (sd cl-sample) 0) => (roughly100 sigma)))))

(defn test-student-t [factory]
  (let [nu 30
        mu 0
        sigma 1]
    (facts
     "Core functions for Student's distribution."
     (with-release [dist (student-t factory nu mu sigma)
                    t-sampler (sampler dist)
                    cl-sample (dataset factory (sample t-sampler))
                    host-sample (transfer (p/data cl-sample))]
       (entry (mean t-sampler) 0) => (roughly (mean dist) 0.01)
       (entry (mean cl-sample) 0) => (roughly (mean dist) 0.01)
       (entry (sd t-sampler) 0) => (roughly100 (sd dist))))))

(defn test-beta [factory]
  (let [a 2.0
        b 5.0
        beta-pdf (fn ^double [^double x] (beta-pdf a b x))]
    (facts
     "Core functions for beta distribution."
     (with-release [dist (beta factory a b)
                    beta-sampler (sampler dist)
                    cl-sample (dataset factory (sample beta-sampler))
                    cl-pdf (pdf dist cl-sample)
                    host-sample-data (transfer (p/data cl-sample))]
       (entry (mean beta-sampler) 0) => (roughly100 (mean dist))
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (sd cl-sample) 0) => (roughly100 (sd dist))
       (sum cl-pdf) => (roughly (sum (fmap! beta-pdf (row host-sample-data 0))))))))

(defn test-gamma [factory]
  (let [theta 2.0
        k 5.0]
    (facts
     "Core functions for gamma distribution."
     (with-release [dist (gamma factory theta k)
                    gamma-sampler (sampler dist)
                    cl-sample (dataset factory (sample gamma-sampler))
                    cl-pdf (pdf dist cl-sample)
                    host-sample-data (transfer (p/data cl-sample))]
       (entry (mean gamma-sampler) 0) => (roughly100 (mean dist))
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (sd cl-sample) 0) => (roughly100 (sd dist))))))


(defn test-exponential [factory]
  (let [lambda 3.5]
    (facts
     "Core functions for exponential distribution."
     (with-release [dist (exponential factory lambda)
                    exponential-sampler (sampler dist)
                    cl-sample (dataset factory (sample exponential-sampler))]

       (mean dist) => (/ 1.0 lambda)
       (entry (mean cl-sample) 0) => (roughly (mean dist) 0.05)
       (entry (variance cl-sample) 0)  => (roughly (variance dist) 0.05)
       (entry (sd cl-sample) 0) => (roughly (sd dist) 0.03)))))

(defn test-erlang [factory]
  (let [lambda 3.5
        k 9]
    (facts
     "Core functions for erlang distribution."
     (with-release [dist (erlang factory lambda k)
                    erlang-sampler (sampler dist)
                    cl-sample (dataset factory (sample erlang-sampler))]

       (mean dist) => (/ k lambda)
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (variance cl-sample) 0)  => (roughly 0.75 0.04)
       (entry (sd cl-sample) 0) => (roughly 0.866 0.04)))))

(defn test-beta-bernouli [factory binomial-lik-model]
  (let [a 8.5 b 3.5
        z 6 N 9
        a1 (+ z a) b1 (+ (- N z) b)]
    (with-release [prior-dist (beta factory a b)
                   prior-sampler (sampler prior-dist)
                   prior-sample (dataset factory (sample prior-sampler))
                   post (posterior factory "post" binomial-lik-model prior-dist)
                   post-dist (post (fv (binomial-lik-params N z)))
                   post-sampler (doto (sampler post-dist) (mix!))
                   post-sample (dataset factory (sample post-sampler))
                   post-pdf (pdf post-dist post-sample)
                   real-post (beta factory a1 b1)
                   real-sampler (sampler real-post)
                   real-sample (dataset factory (sample real-sampler))
                   real-pdf (pdf real-post real-sample)]

      (let [prior-evidence (evidence post-dist prior-sample)]
        (facts
         "Core functions for beta-bernoulli distribution."
         (entry (mean prior-sample) 0) => (roughly (mean prior-dist))
         prior-evidence => (roughly (exp (- (log-beta a1 b1) (log-beta a b))))
         (/ (sum (scal! (/ prior-evidence) post-pdf)) (dim post-pdf))
         =>   (roughly100 (/ (sum real-pdf) (dim real-pdf)))
         (entry (mean post-sample) 0) =>  (roughly100 (mean real-post))
         (entry (sd post-sample) 0) =>  (roughly100 (sd real-post)))))))

(defn test-dataset [bayadera-factory]
  (let [data-size (* 31 (long (Math/pow 2 16)))]
    (with-release [data-matrix (ge bayadera-factory 22 data-size (repeatedly (* 22 data-size) rand))
                   data-set (dataset bayadera-factory data-matrix)]
      (facts
       "Test histogram"
       (/ (sum (col (:pdf (histogram data-set)) 4)) (mrows (:pdf (histogram data-set)))) => (roughly 1.0))
      (facts
       "Test variance"
       (sum (axpy! -1 (variance (p/data data-set)) (variance data-set))) => (roughly 0 0.003))
      (facts
       "Test mean"
       (sum (axpy! -1 (mean (p/data data-set)) (mean data-set))) => (roughly 0 0.003)))))

(defn test-all [factory binomial-lik-model]
  (test-uniform factory)
  (test-gaussian factory)
  (test-exponential factory)
  (test-erlang factory)
  (test-student-t factory)
  (test-beta factory)
  (test-gamma factory)
  (test-beta-bernouli factory binomial-lik-model)
  (test-dataset factory))
