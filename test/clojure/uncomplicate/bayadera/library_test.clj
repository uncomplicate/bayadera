(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.library-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [math :refer [log exp sqrt]]
             [core :refer [dim nrm2 copy dot scal! transfer row vctr]]
             [real :refer [sum entry]]]
            [uncomplicate.bayadera
             [core :refer [mean sd variance sampler dataset sample density distribution
                           evidence histogram]]
             [distributions :refer [beta-pdf binomial-lik-params]]
             [library :refer :all]
             [mcmc :refer [mix!]]
             [math :refer [log-beta]]]
            [uncomplicate.bayadera.internal
             [protocols :as p]]))

(defmacro roughly100 [exp]
  `(let [v# (double ~exp)]
     (roughly v# (/ v# 100.0)) ))

(defn test-uniform [library]
  (let [a 100.0
        b 133.4]
    (facts
     "Core functions for uniform distribution."
     (with-release [dist (uniform library a b)
                    uniform-sampler (sampler dist)
                    cl-sample (dataset (p/factory library) (sample uniform-sampler))
                    cl-pdf (density dist cl-sample)]
       (dim (mean cl-sample)) => 1
       (dim (sd cl-sample)) => 1
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (sd cl-sample) 0) => (roughly100 (sd dist))
       (/ (sum cl-pdf) (dim cl-pdf)) => (roughly (/ 1.0 (- b a)))))))

(defn test-gaussian [library]
  (let [mu 200.0
        sigma 10.0]
    (facts
     "Core functions for gaussian distribution."
     (with-release [dist (gaussian library mu sigma)
                    gaussian-sampler (sampler dist)
                    cl-sample (dataset (p/factory library) (sample gaussian-sampler))]
       (mean dist) => mu
       (sd dist) => sigma
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (variance cl-sample) 0)  => (roughly 100.0 2.0)
       (entry (sd cl-sample) 0) => (roughly100 sigma)))))

(defn test-student-t [library]
  (let [nu 30
        mu 0
        sigma 1]
    (facts
     "Core functions for Student's distribution."
     (with-release [dist (student-t library nu mu sigma)
                    t-sampler (sampler dist)
                    cl-sample (dataset (p/factory library) (sample t-sampler))
                    host-sample (transfer (p/data cl-sample))]
       (entry (mean t-sampler) 0) => (roughly (mean dist) 0.01)
       (entry (mean cl-sample) 0) => (roughly (mean dist) 0.01)
       (entry (sd t-sampler) 0) => (roughly100 (sd dist))))))

(defn test-beta [library]
  (let [a 2.0
        b 5.0
        beta-pdf (fn ^double [^double x] (beta-pdf a b x))]
    (facts
     "Core functions for beta distribution."
     (with-release [dist (beta library a b)
                    beta-sampler (sampler dist)
                    cl-sample (dataset (p/factory library) (sample beta-sampler))
                    cl-pdf (density dist cl-sample)
                    host-sample-data (transfer (p/data cl-sample))]
       (entry (mean beta-sampler) 0) => (roughly100 (mean dist))
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (sd cl-sample) 0) => (roughly100 (sd dist))
       (sum cl-pdf) => (roughly (sum (fmap! beta-pdf (row host-sample-data 0))))))))

(defn test-gamma [library]
  (let [theta 2.0
        k 5.0]
    (facts
     "Core functions for gamma distribution."
     (with-release [dist (gamma library theta k)
                    gamma-sampler (sampler dist)
                    cl-sample (dataset (p/factory library) (sample gamma-sampler))
                    cl-pdf (density dist cl-sample)
                    host-sample-data (transfer (p/data cl-sample))]
       (entry (mean gamma-sampler) 0) => (roughly100 (mean dist))
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (sd cl-sample) 0) => (roughly100 (sd dist))))))


(defn test-exponential [library]
  (let [lambda 3.5]
    (facts
     "Core functions for exponential distribution."
     (with-release [dist (exponential library lambda)
                    exponential-sampler (sampler dist)
                    cl-sample (dataset (p/factory library) (sample exponential-sampler))]
       (mean dist) => (/ 1.0 lambda)
       (entry (mean cl-sample) 0) => (roughly (mean dist) 0.05)
       (entry (variance cl-sample) 0)  => (roughly (variance dist) 0.05)
       (entry (sd cl-sample) 0) => (roughly (sd dist) 0.03)))))

(defn test-erlang [library]
  (let [lambda 3.5
        k 9]
    (facts
     "Core functions for erlang distribution."
     (with-release [dist (erlang library lambda k)
                    erlang-sampler (sampler dist)
                    cl-sample (dataset (p/factory library) (sample erlang-sampler))]
       (mean dist) => (/ k lambda)
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (variance cl-sample) 0)  => (roughly 0.75 0.04)
       (entry (sd cl-sample) 0) => (roughly 0.866 0.04)))))

(defn test-beta-bernouli [library]
  (let [a 8.5 b 3.5
        z 6 N 9
        a1 (+ z a) b1 (+ (- N z) b)]
    (let [factory (p/factory library)]
      (with-release [prior-dist (beta library a b)
                     prior-sampler (sampler prior-dist)
                     prior-sample (dataset factory (sample prior-sampler))
                     binomial-lik (likelihood library :binomial)
                     coin-data (vctr factory (binomial-lik-params N z))
                     post (distribution factory binomial-lik prior-dist)
                     post-dist (post coin-data)
                     post-sampler (doto (sampler post-dist) (mix!))
                     post-sample (dataset (p/factory library) (sample post-sampler))
                     post-pdf (density post-dist post-sample)
                     real-post (beta library a1 b1)
                     real-sampler (sampler real-post)
                     real-sample (dataset (p/factory library) (sample real-sampler))
                     real-pdf (density real-post real-sample)]

        (let [prior-evidence (evidence binomial-lik coin-data prior-sample)]
          (facts
           "Core functions for beta-bernoulli distribution."
           (entry (mean prior-sample) 0) => (roughly (mean prior-dist))
           prior-evidence => (roughly (exp (- (log-beta a1 b1) (log-beta a b))))
           (/ (sum (scal! (/ prior-evidence) post-pdf)) (dim post-pdf))
           =>   (roughly100 (/ (sum real-pdf) (dim real-pdf)))
           (entry (mean post-sample) 0) =>  (roughly100 (mean real-post))
           (entry (sd post-sample) 0) =>  (roughly100 (sd real-post))))))))

(defn test-all [library]
  (test-uniform library)
  (test-gaussian library)
  (test-exponential library)
  (test-erlang library)
  (test-student-t library)
  (test-beta library)
  (test-gamma library)
  (test-beta-bernouli library))
