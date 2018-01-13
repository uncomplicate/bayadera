(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl.core-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [math :refer [log exp sqrt]]
             [core :refer [dim nrm2 copy dot scal! transfer row]]
             [real :refer [sum entry]]
             [native :refer [fv]]]
            [uncomplicate.bayadera
             [core :refer :all]
             [distributions :refer [beta-pdf binomial-lik-params]]
             [mcmc :refer [mix!]]
             [math :refer [log-beta]]]
            [uncomplicate.bayadera.internal.protocols :as p]
            [uncomplicate.bayadera.opencl :refer [with-default-bayadera]]
            [uncomplicate.bayadera.internal.device.models :refer [distributions likelihoods]]))

(defmacro roughly100 [exp]
  `(let [v# (double ~exp)]
     (roughly v# (/ v# 100.0)) ))

(with-default-bayadera

  (let [a 100.0
        b 133.4]
    (facts
     "Core functions for uniform distribution."
     (with-release [dist (uniform a b)
                    uniform-sampler (sampler dist)
                    cl-sample (dataset (sample uniform-sampler))
                    cl-pdf (pdf dist cl-sample)]

       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (sd cl-sample) 0) => (roughly100 (sd dist))
       (/ (sum cl-pdf) (dim cl-pdf)) => (roughly (/ 1.0 (- b a))))))

  (let [mu 200.0
        sigma 10.0]
    (facts
     "Core functions for gaussian distribution."
     (with-release [dist (gaussian mu sigma)
                    gaussian-sampler (sampler dist)
                    cl-sample (dataset (sample gaussian-sampler))]

       (mean dist) => mu
       (sd dist) => sigma
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (variance cl-sample) 0)  => (roughly100 (variance dist))
       (entry (sd cl-sample) 0) => (roughly100 sigma))))

  (let [nu 30
        mu 0
        sigma 1]
    (facts
     "Core functions for Student's distribution."
     (with-release [dist (student-t nu mu sigma)
                    t-sampler (sampler dist)]
       (with-release [cl-sample (dataset (sample t-sampler))
                      host-sample (transfer (p/data cl-sample))]

         (entry (mean t-sampler) 0) => (roughly (mean dist) 0.01)
         (entry (mean cl-sample) 0) => (roughly (mean dist) 0.01)
         (entry (sd t-sampler) 0) => (roughly100 (sd dist))))))

  (let [a 2.0
        b 5.0
        beta-pdf (fn ^double [^double x] (beta-pdf a b x))]
    (facts
     "Core functions for beta distribution."
     (with-release [dist (beta a b)
                    beta-sampler (sampler dist)
                    cl-sample (dataset (sample beta-sampler))
                    cl-pdf (pdf dist cl-sample)
                    host-sample-data (transfer (p/data cl-sample))]
       (entry (mean beta-sampler) 0) => (roughly100 (mean dist))
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (sd cl-sample) 0) => (roughly100 (sd dist))
       (sum cl-pdf) => (roughly (sum (fmap! beta-pdf (row host-sample-data 0)))))))

  (let [theta 2.0
        k 5.0]
    (facts
     "Core functions for gamma distribution."
     (with-release [dist (gamma theta k)
                    gamma-sampler (sampler dist)
                    cl-sample (dataset (sample gamma-sampler))
                    cl-pdf (pdf dist cl-sample)
                    host-sample-data (transfer (p/data cl-sample))]
       (entry (mean gamma-sampler) 0) => (roughly100 (mean dist))
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (sd cl-sample) 0) => (roughly100 (sd dist)))))

  (let [lambda 3.5]
    (facts
     "Core functions for exponential distribution."
     (with-release [dist (exponential lambda)
                    exponential-sampler (sampler dist)
                    cl-sample (dataset (sample exponential-sampler))]

       (mean dist) => (/ 1.0 lambda)
       (entry (mean cl-sample) 0) => (roughly (mean dist) 0.05)
       (entry (variance cl-sample) 0)  => (roughly (variance dist) 0.05)
       (entry (sd cl-sample) 0) => (roughly (sd dist) 0.03))))

  (let [lambda 3.5
        k 9]
    (facts
     "Core functions for erlang distribution."
     (with-release [dist (erlang lambda k)
                    erlang-sampler (sampler dist)
                    cl-sample (dataset (sample erlang-sampler))]

       (mean dist) => (/ k lambda)
       (entry (mean cl-sample) 0) => (roughly100 (mean dist))
       (entry (variance cl-sample) 0)  => (roughly100 (variance dist))
       (entry (sd cl-sample) 0) => (roughly100 (sd dist))))))

(with-default-bayadera
  (let [a 8.5 b 3.5
        z 6 N 9
        a1 (+ z a) b1 (+ (- N z) b)]
    (with-release [prior-dist (beta a b)
                   prior-sampler (sampler prior-dist)
                   prior-sample (dataset (sample prior-sampler))
                   post (posterior "post" (:binomial likelihoods) prior-dist)
                   post-dist (post (fv (binomial-lik-params N z)))
                   post-sampler (doto (sampler post-dist) (mix!))
                   post-sample (dataset (sample post-sampler))
                   post-pdf (pdf post-dist post-sample)
                   real-post (beta a1 b1)
                   real-sampler (sampler real-post)
                   real-sample (dataset (sample real-sampler))
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
