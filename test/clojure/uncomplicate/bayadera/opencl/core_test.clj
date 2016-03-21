(ns uncomplicate.bayadera.opencl.core-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [math :refer [log exp]]
             [core :refer [dim sum nrm2 copy dot scal! transfer!]]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [core :refer :all]
             [distributions :refer [beta-pdf]]
             [impl :refer :all]
             [math :refer [log-beta]]
             [visual :refer :all]]
            [uncomplicate.bayadera.opencl :refer [with-default-bayadera]]
            [uncomplicate.bayadera.opencl.generic
             :refer [binomial-likelihood beta-model]]))

(with-default-bayadera

  (facts
   "Core functions for gaussian distribution."
   (let [sample-count (* 256 44 94)
         mu 200.0
         sigma 10.0]
     (with-release [dist (gaussian mu sigma)
                    gaussian-sampler (sampler dist)
                    cl-sample (dataset (sample gaussian-sampler sample-count))]

       (mean dist) => mu
       (sd dist) => sigma
       (mean cl-sample) => (roughly mu)
       (sd cl-sample) => (roughly sigma (/ sigma 100))
       (mean-variance cl-sample) => (sv (mean cl-sample) (variance cl-sample))
       (mean-sd cl-sample) => (sv (mean cl-sample) (sd cl-sample)))))

  (facts
   "Core functions for uniform distribution."
   (let [sample-count (* 256 44 94)
         a 100.0
         b 133.4]
     (with-release [dist (uniform a b)
                    uniform-sampler (sampler dist)
                    cl-sample (dataset (sample uniform-sampler sample-count))
                    cl-pdf (pdf dist cl-sample)]

       (mean cl-sample) => (roughly (mean dist))
       (sd cl-sample) => (roughly (sd dist) (/ (sd dist) 100.0))
       (mean-variance cl-sample) => (sv (mean cl-sample) (variance cl-sample))
       (mean-sd cl-sample) => (sv (mean cl-sample) (sd cl-sample))
       (/ (sum cl-pdf) (dim cl-pdf)) => (roughly (/ 1.0 (- b a))))))

  (let [sample-count (* 256 44 94)
        a 2.0
        b 5.0
        beta-pdf (fn ^double [^double x] (beta-pdf a b x))]
    (facts
     "Core functions for beta distribution."
     (with-release [dist (beta a b)
                    beta-sampler (time (sampler dist))
                    cl-sample (dataset (sample beta-sampler sample-count))
                    cl-pdf (pdf dist cl-sample)
                    host (transfer! (p/data cl-sample) (sv sample-count))]
       (mean cl-sample) => (roughly (mean dist) (/ (mean dist) 100.0))
       (sd cl-sample) => (roughly (sd dist) (/ (sd dist) 100.0))
       (mean-variance cl-sample) => (sv (mean cl-sample) (variance cl-sample))
       (mean-sd cl-sample) => (sv (mean cl-sample) (sd cl-sample))
       (sum cl-pdf) => (roughly (sum (fmap! beta-pdf host)))))))

(with-default-bayadera
  (let [sample-count (* 256 44 94)
        a 2.0 b 5.0
        z 3.0 N 5.0
        a1 (+ z a) b1 (+ (- N z) b)]
    (with-release [prior-dist (beta a b)
                   prior-sample (dataset (sample (sampler prior-dist) sample-count))
                   post (posterior "post" binomial-likelihood prior-dist)
                   post-dist (post (binomial-lik-params N z))
                   post-sampler (time (sampler post-dist))
                   post-sample (dataset (sample post-sampler sample-count))
                   post-pdf (pdf post-dist post-sample)
                   real-post (beta a1 b1)
                   real-sampler (sampler real-post)
                   real-sample (dataset (sample real-sampler sample-count))
                   real-pdf (pdf real-post real-sample)]
      (let [prior-evidence (evidence post-dist prior-sample)]
        (facts
         "Core functions for beta-bernoulli distribution."
         prior-evidence => (roughly (exp (- (log-beta a1 b1) (log-beta a b)))
                                    (/ prior-evidence 100.0))
         (sum (scal! (/ prior-evidence) post-pdf))
         => (roughly (sum real-pdf) (/ (sum real-pdf) 100.0))
         (mean post-sample) => (roughly (mean real-post) (/ (mean real-post) 100.0))
         (sd post-sample) => (roughly (sd real-post) (/ (sd real-post) 100.0)))))))
