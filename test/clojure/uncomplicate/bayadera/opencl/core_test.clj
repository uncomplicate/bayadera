(ns uncomplicate.bayadera.opencl.core-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [math :refer [log exp]]
             [core :refer [dim sum nrm2 fmap! copy dot scal! transfer!]]
             [real :refer [entry]]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [core :refer :all]
             [distributions :refer [beta-pdf]]
             [impl :refer :all]
             [math :refer [log-beta]]
             [visual :refer :all]]
            [uncomplicate.bayadera.opencl
             [generic :refer [binomial-likelihood beta-model]]
             [amd-gcn :refer [gcn-engine-factory]]]))

(with-release [dev (first (sort-by-cl-version (devices (first (platforms)))))
               ctx (context [dev])
               cqueue (command-queue ctx dev)]
  (facts
   "Core functions for gaussian distribution."
   (let [sample-count (* 256 44 94)
         mu 200.0
         sigma 10.0]
     (with-release [engine-factory (gcn-engine-factory ctx cqueue)
                    dist (gaussian engine-factory mu sigma)
                    gaussian-sampler (sampler dist)
                    cl-sample (dataset engine-factory (sample gaussian-sampler sample-count))]

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
     (with-release [engine-factory (gcn-engine-factory ctx cqueue)
                    dist (uniform engine-factory a b)
                    uniform-sampler (sampler dist)
                    cl-sample (dataset engine-factory (sample uniform-sampler sample-count))
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
     (with-release [engine-factory (gcn-engine-factory ctx cqueue)
                    dist (beta engine-factory a b)
                    beta-sampler (time (sampler dist))
                    cl-sample (dataset engine-factory (sample beta-sampler sample-count))
                    cl-pdf (pdf dist cl-sample)
                    host (transfer! (p/data cl-sample) (sv sample-count))]
       (mean cl-sample) => (roughly (mean dist) (/ (mean dist) 100.0))
       (sd cl-sample) => (roughly (sd dist) (/ (sd dist) 100.0))
       (mean-variance cl-sample) => (sv (mean cl-sample) (variance cl-sample))
       (mean-sd cl-sample) => (sv (mean cl-sample) (sd cl-sample))
       (sum cl-pdf) => (roughly (sum (time (fmap! beta-pdf host))))))))

(with-release [dev (first (sort-by-cl-version (devices (first (platforms)))))
               ctx (context [dev])
               cqueue (command-queue ctx dev)]

  (let [sample-count (* 256 44 94)
        a 2.0
        b 5.0
        z 3.0
        N 5.0
        a1 (+ z a)
        b1 (+ (- N z) b)
        posterior-model (posterior binomial-likelihood beta-model)]
    (with-release [engine-factory (gcn-engine-factory ctx cqueue)
                   prior-dist (beta engine-factory a b)
                   prior-sampler (sampler prior-dist)
                   prior-sample (dataset engine-factory (sample prior-sampler sample-count))
                   post (distribution engine-factory posterior-model)
                   post-dist (post (sv N z) (sv a b (log-beta a b)))
                   post-sampler (time (sampler post-dist))
                   post-sample (dataset engine-factory (sample post-sampler sample-count))
                   post-pdf (pdf post-dist post-sample)
                   host-sample (transfer! (p/data post-sample) (sv sample-count))
                   host-pdf (transfer! post-pdf (sv sample-count))
                   real-post (beta engine-factory a1 b1)
                   real-sampler (sampler real-post)
                   real-sample (dataset engine-factory (sample real-sampler sample-count))
                   real-pdf (pdf real-post real-sample)]
      (let [prior-evidence (evidence post-dist prior-sample)]
        (facts
         "Core functions for beta-bernoulli distribution."

         prior-evidence => (roughly (exp (- (log-beta a1 b1) (log-beta a b))))
         (/ (entry host-pdf 0) prior-evidence) => (roughly (beta-pdf a1 b1 (entry host-sample 0)))
         (sum (scal! (/ 1.0 prior-evidence) post-pdf)) => (roughly (sum real-pdf))
         (mean post-sample) => (roughly (mean real-post) (/ (mean real-post) 100.0))
         (sd post-sample) => (roughly (sd real-post) (/ (sd real-post) 100.0)))))))
