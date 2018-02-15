(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.nvidia-gtx-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.fluokitten.core :refer [fmap! op]]
            [uncomplicate.clojurecuda
             [core :refer :all :exclude [parameters]]
             [info :refer [ctx-device max-block-dim-x driver-version]]]
            [uncomplicate.neanderthal
             [core :refer [vctr ge native native! subvector sum entry imax imin row raw copy axpy! nrm2]]
             [vect-math :refer [linear-frac!]]
             [cuda :refer [cuda-float]]]
            [uncomplicate.bayadera
             [distributions :refer [gaussian-pdf gaussian-log-pdf binomial-lik-params beta-params]]
             [cuda :refer :all :exclude [gtx-bayadera-factory]]]
            [uncomplicate.bayadera.internal
             [protocols :refer :all]
             [nvidia-gtx :refer :all]]
            [uncomplicate.bayadera.core-test :refer :all]
            [uncomplicate.bayadera.internal.device-test :refer :all]))

(with-default
  (let [dev (ctx-device)
        wgs (max-block-dim-x dev)
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
        wgs (max-block-dim-x dev)
        cudart-version (driver-version)]
    (with-release [neanderthal-factory (cuda-float (current-context) default-stream)]

      (facts
       "Nvidia GTX distribution engine with Gaussian distribution"
       (with-release [gaussian-engine (gtx-distribution-engine (current-context) default-stream
                                                               gaussian-model wgs cudart-version)
                      params (vctr neanderthal-factory [0.0 1.0])
                      x (ge neanderthal-factory 1 200 (range -10 10 0.1))
                      native-x-pdf (native! (pdf gaussian-engine params x))
                      native-x-log-pdf (native! (log-pdf gaussian-engine params x))
                      native-x0 (native x)
                      native-x1 (copy native-x0)]

         (nrm2 (axpy! -1 (row (fmap! (fn ^double [^double x] (gaussian-pdf 0.0 1.0 x)) native-x0) 0)
                      native-x-pdf)) => (roughly 0.0 0.00001)

         (nrm2 (axpy! -1 (row (fmap! (fn ^double [^double x] (gaussian-log-pdf 0.0 1.0 x)) native-x1) 0)
                      native-x-log-pdf)) => (roughly 0.0 0.0001)
         (Double/isNaN (evidence gaussian-engine params x)) => truthy))

      (facts
       "Nvidia GTX posterior engine with Beta-Binomial model."
       (let [n 50
             z 15
             a 3
             b 2]
         (with-release [post-engine (gtx-posterior-engine
                                     (current-context) default-stream
                                     (posterior-model beta-model "beta_binomial" binomial-lik-model)
                                     wgs cudart-version)
                        beta-engine (gtx-distribution-engine (current-context) default-stream
                                                             beta-model wgs cudart-version)
                        params (vctr neanderthal-factory (op (binomial-lik-params n z) (beta-params a b)))
                        beta-params (vctr neanderthal-factory (beta-params (+ a z) (+ b (- n z))))
                        x (ge neanderthal-factory 1 200 (range 0.001 1 0.001))
                        x-pdf (pdf post-engine params x)
                        x-log-pdf (log-pdf post-engine params x)
                        x-beta-pdf (pdf beta-engine beta-params x)
                        x-beta-log-pdf (log-pdf beta-engine beta-params x)]

           (nrm2 (linear-frac! (axpy! -1 x-log-pdf x-beta-log-pdf) -32.61044)) => (roughly 0.0 0.0001)
           (evidence post-engine params x) => 1.6357453252754427E-15))))))


#_(with-default

  (with-release [factory (gtx-bayadera-factory (current-context) default-stream)]
    (test-uniform factory)
    ;;(test-gaussian factory)
    ;;(test-erlang factory)
    ;;(test-exponential factory)
;;    (test-student-t factory)
;;    (test-gamma factory)
    #_(test-all factory)
    (test-dataset factory)
    #_(test-mcmc factory gaussian-model)))
