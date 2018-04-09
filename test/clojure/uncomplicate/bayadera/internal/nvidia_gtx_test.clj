(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.nvidia-gtx-test
  (:require [midje.sweet :refer :all]
            [clojure.java.io :as io]
            [clojure.string :refer [split-lines]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.fluokitten.core :refer [fmap! op]]
            [uncomplicate.clojurecuda
             [core :refer :all :as clojurecuda :exclude [parameters]]
             [info :refer [ctx-device max-block-dim-x driver-version multiprocessor-count]]]
            [uncomplicate.neanderthal
             [core :refer [vctr ge native native! subvector sum entry imax imin row raw copy axpy! nrm2]]
             [vect-math :refer [linear-frac!]]
             [cuda :refer [cuda-float]]
             [block :refer :all]]
            [uncomplicate.bayadera
             [distributions :refer [gaussian-pdf gaussian-log-pdf binomial-lik-params beta-params]]
             [cuda :refer :all :as cuda :exclude [gtx-bayadera-factory]]]
            [uncomplicate.bayadera.internal.protocols :refer :all]
            [uncomplicate.bayadera.internal.device.nvidia-gtx :refer :all]
            [uncomplicate.bayadera.core-test :refer [test-all]])
  (:import uncomplicate.bayadera.internal.device.nvidia_gtx.GTXStretch))

(with-default
  (let [dev (ctx-device)
        wgs 256
        cudart-version (driver-version)]
    (with-release [neanderthal-factory (cuda-float (current-context) default-stream)]

      (facts
       "Nvidia GTX direct sampler for Uniform distribution"
       (with-release [normal-sampler (gtx-direct-sampler (current-context) default-stream
                                                         uniform-model wgs cudart-version)
                      params (vctr neanderthal-factory [-99.9 200.1])
                      smpl (sample normal-sampler 123 params 10000)
                      native-sample (native smpl)]
         (let [sample-1d (row native-sample 0)]
           (seq (subvector sample-1d 0 4))
           => (list 108.93402099609375 139.30517578125 -82.35221862792969 47.42252731323242)
           (seq (subvector sample-1d 9996 4))
           => (list 17.43636131286621 151.42117309570312 42.78262710571289 107.87583923339844)
           (entry sample-1d (imax sample-1d)) => 200.0757293701172
           (entry sample-1d (imin sample-1d)) => -99.86572265625
           (mean sample-1d) => 51.33118125)))

      (facts
       "Nvidia GTX direct sampler for Gaussian distribution"
       (with-release [gaussian-sampler (gtx-direct-sampler (current-context) default-stream
                                                           gaussian-model wgs cudart-version)
                      params (vctr neanderthal-factory [100 200.1])
                      smpl (sample gaussian-sampler 123 params 10000)
                      native-sample (native smpl)]
         (let [sample-1d (row native-sample 0)]
           (seq (subvector sample-1d 0 4))
           => (list -27.02083969116211 55.2710075378418 185.74417114257812 322.7049255371094)
           (seq (subvector sample-1d 9996 4))
           => (list 175.25135803222656 7.720771312713623 126.18173217773438 -69.4984359741211)
           (entry sample-1d (imax sample-1d)) => 868.6444702148438
           (entry sample-1d (imin sample-1d)) => -610.0802612304688
           (mean sample-1d) => 95.13473125)))

      (facts
       "Nvidia GTX direct sampler for Erlang distribution"
       (with-release [erlang-sampler (gtx-direct-sampler (current-context) default-stream
                                                         erlang-model wgs cudart-version)
                      params (vctr neanderthal-factory [2 3])
                      smpl (sample erlang-sampler 123 params 10000)
                      native-sample (native smpl)]
         (let [sample-1d (row native-sample 0)]
           (seq (subvector sample-1d 0 4))
           => (list 1.571028470993042 1.4484456777572632 0.798355758190155 1.1712464094161987)
           (seq (subvector sample-1d 9996 4))
           => (list 0.7548800110816956 2.2858035564422607 1.19755220413208 1.3439300060272217)
           (entry sample-1d (imax sample-1d)) => 7.1595940589904785
           (entry sample-1d (imin sample-1d)) => 0.04825383424758911
           (mean sample-1d) => 1.5008443359375)))

      (facts
         "OpenCL GCN direct sampler for Exponential distribution"
         (with-release [exponential-sampler (gtx-direct-sampler (current-context) default-stream
                                                                exponential-model wgs cudart-version)
                        params (vctr neanderthal-factory [4])
                        smpl (sample exponential-sampler 123 params 10000)
                        native-sample (native smpl)]
           (let [sample-1d (row native-sample 0)]
             (seq (subvector sample-1d 0 4))
             => (list 0.29777514934539795 0.3990693986415863 0.015068254433572292 0.1688637137413025)
             (seq (subvector sample-1d 9996 4))
             => (list 0.12403398752212524 0.45463457703590393 0.16137929260730743 0.29489004611968994)
             (entry sample-1d (imax sample-1d)) => 2.3556251525878906
             (entry sample-1d (imin sample-1d)) => 2.8555818062159233E-5
             (mean sample-1d) => 0.2526310546875))))))

(with-default
  (let [dev (ctx-device)
        wgs 256]
    (with-release [neanderthal-factory (cuda-float (current-context) default-stream)]

      (facts
       "Nvidia GTX distribution engine with Gaussian distribution"
       (with-release [gaussian-engine (gtx-distribution-engine (current-context) default-stream
                                                               gaussian-model wgs)
                      params (vctr neanderthal-factory [0.0 1.0])
                      x (ge neanderthal-factory 1 200 (range -10 10 0.1))
                      native-x-pdf (native! (density gaussian-engine params x))
                      native-x-log-pdf (native! (log-density gaussian-engine params x))
                      native-x0 (native x)
                      native-x1 (copy native-x0)]

         (nrm2 (axpy! -1 (row (fmap! (fn ^double [^double x] (gaussian-pdf 0.0 1.0 x)) native-x0) 0)
                      native-x-pdf)) => (roughly 0.0 0.00001)

         (nrm2 (axpy! -1 (row (fmap! (fn ^double [^double x] (gaussian-log-pdf 0.0 1.0 x)) native-x1) 0)
                      native-x-log-pdf)) => (roughly 0.0 0.0001)))

      (facts
       "Nvidia GTX posterior engine with Beta-Binomial model."
       (let [n 50
             z 15
             a 3
             b 2]
         (with-release [post-engine (gtx-distribution-engine
                                     (current-context) default-stream
                                     (posterior-model beta-model "beta_binomial" binomial-lik-model)
                                     wgs)
                        beta-engine (gtx-distribution-engine (current-context) default-stream
                                                             beta-model wgs)
                        binomial-lik-engine (gtx-likelihood-engine (current-context) default-stream
                                                                   binomial-lik-model wgs)
                        lik-params (vctr neanderthal-factory (binomial-lik-params n z))
                        params (vctr neanderthal-factory (op (binomial-lik-params n z) (beta-params a b)))
                        beta-params (vctr neanderthal-factory (beta-params (+ a z) (+ b (- n z))))
                        x (ge neanderthal-factory 1 200 (range 0.001 1 0.001))
                        x-pdf (density post-engine params x)
                        x-log-pdf (log-density post-engine params x)
                        x-beta-pdf (density beta-engine beta-params x)
                        x-beta-log-pdf (log-density beta-engine beta-params x)]

           (nrm2 (linear-frac! (axpy! -1 x-log-pdf x-beta-log-pdf) -32.61044)) => (roughly 0.0 0.0001)
           (evidence binomial-lik-engine lik-params x) => 1.6357453252754427E-15))))))

(with-default
  (let [dev (ctx-device)
        wgs 256
        cudart-version (driver-version)
        walker-count (* 44 wgs)
        means-count (long (blocks-count wgs (/ walker-count 2)))
        acc-count (long (blocks-count wgs means-count))
        seed 123
        a 2.0]
    (with-release [neanderthal-factory (cuda-float (current-context) default-stream)]

      (facts
         "Nvidia GTX stretch with Uniform model."
         (with-release [params (vctr neanderthal-factory [-1 2])
                        limits (ge neanderthal-factory 2 1 [-1 2])
                        uniform-sampler (mcmc-sampler (gtx-stretch-factory
                                                       (current-context) default-stream
                                                       neanderthal-factory nil uniform-model
                                                       wgs cudart-version)
                                                      walker-count params)]
           (let [stretch-move-bare-kernel (.stretch-move-bare-kernel ^GTXStretch uniform-sampler)
                 stretch-move-kernel (.stretch-move-kernel ^GTXStretch uniform-sampler)
                 sum-means-kernel (.sum-means-kernel ^GTXStretch uniform-sampler)
                 cu-params (.cu-params ^GTXStretch uniform-sampler)
                 cu-xs (.cu-xs ^GTXStretch uniform-sampler)
                 cu-s0 (.cu-s0 ^GTXStretch uniform-sampler)
                 cu-logfn-s0 (.cu-logfn-s0 ^GTXStretch uniform-sampler)
                 cu-s1 (.cu-s1 ^GTXStretch uniform-sampler)
                 cu-logfn-s1 (.cu-logfn-s1 ^GTXStretch uniform-sampler)
                 cu-accept (.cu-accept ^GTXStretch uniform-sampler)
                 cu-means-acc (create-data-source neanderthal-factory means-count)
                 means-acc-array (float-array 10)
                 accept-array (int-array 10)
                 acc (ge neanderthal-factory 1 acc-count)
                 hstream (.hstream ^GTXStretch uniform-sampler)]

             (init-position! uniform-sampler seed limits)
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [1.0883402824401855 1.3920516967773438 -0.8245221972465515 0.47322529554367065]
             (init! uniform-sampler seed) => uniform-sampler
             (move-bare! uniform-sampler) => uniform-sampler
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [0.7279692888259888 1.81407630443573 0.040318019688129425 0.4697103202342987]
             (move-bare! uniform-sampler)
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [1.040357232093811 1.4457943439483643 0.3761849105358124 1.5768483877182007]

             (init-position! uniform-sampler seed limits)
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [1.0883402824401855 1.3920516967773438 -0.8245221972465515 0.47322529554367065]

             (launch! stretch-move-bare-kernel (grid-1d (/ walker-count 2) wgs) hstream
                      (clojurecuda/parameters (/ walker-count 2) (int seed) (int 3333)
                                              cu-params cu-s1 cu-s0 cu-logfn-s0 (float a) (float 1.0) (int 0)))
             (launch! stretch-move-bare-kernel (grid-1d (/ walker-count 2) wgs) hstream
                      (clojurecuda/parameters (/ walker-count 2) (inc (int seed)) (int 4444)
                                              cu-params cu-s0 cu-s1 cu-logfn-s1 (float a) (float 1.0) (int 0)))
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [0.7279692888259888 1.81407630443573 0.040318019688129425 0.4697103202342987]

             (launch! stretch-move-bare-kernel (grid-1d (/ walker-count 2) wgs) hstream
                      (clojurecuda/parameters (/ walker-count 2) (int seed) (int 3333) cu-params
                                              cu-s1 cu-s0 cu-logfn-s0 (float a) (float 1.0) (int 1)))
             (launch! stretch-move-bare-kernel (grid-1d (/ walker-count 2) wgs) hstream
                      (clojurecuda/parameters (/ walker-count 2) (inc (int seed)) (int 4444) cu-params
                                              cu-s0 cu-s1 cu-logfn-s1 (float a) (float 1.0) (int 1)))
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [1.040357232093811 1.4457943439483643 0.3761849105358124 1.5768483877182007]

             (memset! cu-accept (int 0) hstream)
             (memset! cu-means-acc (float 0) hstream)
             (launch! stretch-move-kernel (grid-1d (/ walker-count 2) wgs) hstream
                      (clojurecuda/parameters (/ walker-count 2) (int seed) (int 1111)
                                              cu-params cu-s1 cu-s0 cu-logfn-s0 cu-accept cu-means-acc
                                              (float a) (int 0)))
             (launch! stretch-move-kernel (grid-1d (/ walker-count 2) wgs) hstream
                      (clojurecuda/parameters (/ walker-count 2) (inc (int seed)) (int 2222)
                                              cu-params cu-s0 cu-s1 cu-logfn-s1 cu-accept cu-means-acc
                                              (float a) (int 0)))
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [1.011080265045166 1.615005373954773 0.3426262140274048 1.4122663736343384]
             (memcpy-host! cu-means-acc means-acc-array)
             (seq means-acc-array) => (map float [269.26575 286.3589 288.09372 240.0009 265.76953
                                                  274.17465 257.67914 302.7213 244.6228 277.85284])
             (memcpy-host! cu-accept accept-array)
             (seq accept-array) => [423 422 424 428 414 439 428 409 429 409]

             (launch! sum-means-kernel (grid-2d 1 means-count 1 wgs) hstream
                      (clojurecuda/parameters 1 means-count (buffer acc) cu-means-acc))
             (sum acc) => (float 5822.918))))

      (facts
       "Nvidia GTX stretch with Gaussian model."
       (with-release [params (vctr neanderthal-factory [3 1.0])
                      limits (ge neanderthal-factory 2 1 [-7 7])
                      gaussian-sampler (mcmc-sampler (gtx-stretch-factory
                                                      (current-context) default-stream
                                                      neanderthal-factory nil gaussian-model
                                                      wgs cudart-version)
                                                     walker-count params)]
         (init-position! gaussian-sampler seed limits)
         (take 4 (native (row (sample gaussian-sampler) 0)))
         => [2.7455878257751465 4.16290807723999 -6.181103706359863 -0.12494856119155884]
         (init! gaussian-sampler seed) => gaussian-sampler
         (move-bare! gaussian-sampler) => gaussian-sampler
         (take 4 (native (row (sample gaussian-sampler) 0)))
         => [2.7455878257751465 4.16290807723999 -2.1451826095581055 -0.14135171473026276]
         (move-bare! gaussian-sampler)
         (take 4 (native (row (sample gaussian-sampler) 0)))
         => [3.9621310234069824 2.9586496353149414 -0.5778038501739502 5.025292873382568]))

      (facts
       "Nvidia GTX stretch burn-in with Gaussian model."
       (with-release [params (vctr neanderthal-factory [3 1.0])
                      limits (ge neanderthal-factory 2 1 [-7 7])
                      gaussian-sampler (mcmc-sampler (gtx-stretch-factory
                                                      (current-context) default-stream
                                                      neanderthal-factory nil gaussian-model
                                                      wgs cudart-version)
                                                     walker-count params)]
         (init-position! gaussian-sampler seed limits)
         (init! gaussian-sampler (inc seed))
         (burn-in! gaussian-sampler 100 1.5)
         (first (native! (mean (sample gaussian-sampler)))) => 2.955857992172241
         (take-nth 1500 (native! (row (sample gaussian-sampler) 0)))
         => [3.222633123397827 2.4723587036132812 3.094208240509033 3.0372467041015625
             5.75404167175293 2.7033119201660156 2.5398690700531006 3.4444446563720703]
         (first (native! (sd (sample gaussian-sampler)))) => 0.9953223466873169)))))

(with-default
  (let [dev (ctx-device)
        wgs 256
        cudart-version (driver-version)
        walker-count (* 2 44 wgs)
        seed 123
        a 8.0]
    (with-release [bayadera-factory (cuda/gtx-bayadera-factory (current-context) default-stream 20 wgs)]
      (let [mcmc-engine-factory (mcmc-factory bayadera-factory gaussian-model)]
        (with-release [params (vctr bayadera-factory [200 1])
                       limits (ge bayadera-factory 2 1 [180.0 220.0])
                       dummy-sample-matrix (ge bayadera-factory 1 (* 100 walker-count) (cycle [201 199 138]))]
          (let [engine (mcmc-sampler mcmc-engine-factory walker-count params)]
            (facts
             "Test MCMC stretch engine."
             (init-position! engine 123 limits)
             (init! engine 1243)
             (burn-in! engine 5120 a)
             (acc-rate! engine a) => 0.48193359375
             (entry (:tau (:autocorrelation (run-sampler! engine 670 a))) 0) => (roughly 9.9158 0.001))))))))

(with-default
  (let [dev (ctx-device)
        wgs 256
        cudart-version (driver-version)
        walker-count (* 2 44 wgs)
        seed 123
        a 8.0]
    (with-release [neanderthal-factory (cuda-float (current-context) default-stream)
                   engine (gtx-acor-engine (current-context) default-stream wgs)
                   data-matrix-67
                   (ge neanderthal-factory 1 67
                       (map (comp float read-string)
                            (split-lines (slurp (io/resource "uncomplicate/bayadera/internal/acor-data-67")))))
                   data-matrix-367
                   (ge neanderthal-factory 1 367
                       (map (comp float read-string)
                            (split-lines (slurp (io/resource "uncomplicate/bayadera/internal/acor-data-367")))))
                   data-matrix-112640
                   (ge neanderthal-factory 1 112640
                       (map (comp float read-string)
                            (split-lines (slurp (io/resource "uncomplicate/bayadera/internal/acor-data-112640")))))]
      (let []
       (facts
         "Test MCMC acor."
         (first (:tau (acor engine data-matrix-67))) => 11.826833724975586
         (first (:tau (acor engine data-matrix-367))) => 17.302560806274414
         (let [autocorrelation (acor engine data-matrix-112640)]
           (entry  (:tau autocorrelation) 0) => (roughly 20.41 0.001)
           (entry (:sigma autocorrelation) 0) => (roughly 0.009 0.001)))))))

(with-default
  (with-release [factory (cuda/gtx-bayadera-factory (current-context) default-stream)]
    (test-all factory binomial-lik-model)))
