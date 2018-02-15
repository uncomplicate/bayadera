(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.amd-gcn-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.fluokitten.core :refer [fmap! op]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [queue-device max-compute-units max-work-group-size]]]
            [uncomplicate.neanderthal
             [core :refer [vctr ge native native! subvector sum entry imax imin row raw copy axpy! nrm2]]
             [vect-math :refer [linear-frac!]]
             [opencl :refer [opencl-float]]]
            [uncomplicate.bayadera
             [distributions :refer [gaussian-pdf gaussian-log-pdf binomial-lik-params beta-params]]
             [opencl :refer :all :as ocl :exclude [gcn-bayadera-factory]]]
            [uncomplicate.bayadera.internal
             [protocols :refer :all]
             [amd-gcn :refer :all]
             [util :refer [create-tmp-dir with-philox]]]
            [uncomplicate.bayadera.core-test :refer [test-all]]
            [uncomplicate.bayadera.internal.device-test :refer :all]))

(with-default
  (let [dev (queue-device *command-queue*)
        wgs (max-work-group-size dev)
        tmp-dir-name (create-tmp-dir)]
    (with-philox tmp-dir-name
      (with-release [neanderthal-factory (opencl-float *context* *command-queue*)]

        (facts
         "OpenCL GCN direct sampler for Uniform distribution"
         (with-release [uniform-sampler (gcn-direct-sampler *context* *command-queue* tmp-dir-name
                                                           uniform-model wgs)
                        params (vctr neanderthal-factory [-99.9 200.1])
                        smpl (sample uniform-sampler 123 params 10000)
                        native-sample (native smpl)]
           (let [sample-1d (row native-sample 0)]
             (seq (subvector sample-1d 0 4))
             => (list 108.93401336669922 139.30517578125 -82.35221862792969 47.422523498535156)
             (seq (subvector sample-1d 9996 4))
             => (list 17.436363220214844 151.42117309570312 42.782630920410156 107.8758316040039)
             (entry sample-1d (imax sample-1d)) => 200.07574462890625
             (entry sample-1d (imin sample-1d)) => -99.86572265625
             (mean sample-1d) => 51.33118125)))

        (facts
         "OpenCL GCN direct sampler for Gaussian distribution"
         (with-release [gaussian-sampler (gcn-direct-sampler *context* *command-queue* tmp-dir-name
                                                             gaussian-model wgs)
                        params (vctr neanderthal-factory [100 200.1])
                        smpl (sample gaussian-sampler 123 params 10000)
                        native-sample (native smpl)]
           (let [sample-1d (row native-sample 0)]
             (seq (subvector sample-1d 0 4))
             => (list -27.02086639404297 55.27095031738281 185.74420166015625 322.7049560546875)
             (seq (subvector sample-1d 9996 4))
             => (list 175.25137329101562 7.7207489013671875 126.18175506591797 -69.49845886230469)
             (entry sample-1d (imax sample-1d)) => 868.6443481445312
             (entry sample-1d (imin sample-1d)) => -610.0803833007812
             (mean sample-1d) => 95.134725)))

        (facts
         "OpenCL GCN direct sampler for Erlang distribution"
         (with-release [erlang-sampler (gcn-direct-sampler *context* *command-queue* tmp-dir-name
                                                           erlang-model wgs)
                        params (vctr neanderthal-factory [2 3])
                        smpl (sample erlang-sampler 123 params 10000)
                        native-sample (native smpl)]
           (let [sample-1d (row native-sample 0)]
             (seq (subvector sample-1d 0 4))
             => (list 1.571028709411621 1.4484457969665527 0.7983559370040894 1.1712465286254883)
             (seq (subvector sample-1d 9996 4))
             => (list 0.7548800706863403 2.28580379486084 1.1975524425506592 1.3439302444458008)
             (entry sample-1d (imax sample-1d)) => 7.159594535827637
             (entry sample-1d (imin sample-1d)) => 0.0482538677752018
             (mean sample-1d) => 1.50084453125)
           ))

        (facts
         "OpenCL GCN direct sampler for Exponential distribution"
         (with-release [exponential-sampler (gcn-direct-sampler *context* *command-queue* tmp-dir-name
                                                           exponential-model wgs)
                        params (vctr neanderthal-factory [4])
                        smpl (sample exponential-sampler 123 params 10000)
                        native-sample (native smpl)]
           (let [sample-1d (row native-sample 0)]
             (seq (subvector sample-1d 0 4))
             => (list 0.29777517914772034 0.39906948804855347 0.015068267472088337 0.1688637137413025)
             (seq (subvector sample-1d 9996 4))
             => (list 0.12403401732444763 0.4546346068382263 0.16137930750846863 0.29489007592201233)
             (entry sample-1d (imax sample-1d)) => 2.3556253910064697
             (entry sample-1d (imin sample-1d)) => 2.8567157642100938E-5
             (mean sample-1d) => 0.252631103515625)))))))

(with-default
  (let [dev (queue-device *command-queue*)
        wgs (max-work-group-size dev)
        tmp-dir-name (create-tmp-dir)]
    (with-philox tmp-dir-name
      (with-release [neanderthal-factory (opencl-float *context* *command-queue*)]

        (facts
         "OpenCL GCN distribution engine with Gaussian distribution"
         (with-release [gaussian-engine (gcn-distribution-engine *context* *command-queue* tmp-dir-name
                                                                 gaussian-model wgs)
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
         "OpenCL GCN posterior engine with Beta-Binomial model."
         (let [n 50
               z 15
               a 3
               b 2]
           (with-release [post-engine (gcn-posterior-engine
                                       *context* *command-queue* tmp-dir-name
                                       (posterior-model beta-model "beta_binomial" binomial-lik-model)
                                       wgs)
                          beta-engine (gcn-distribution-engine *context* *command-queue* tmp-dir-name
                                                               beta-model wgs)
                          params (vctr neanderthal-factory (op (binomial-lik-params n z) (beta-params a b)))
                          beta-params (vctr neanderthal-factory (beta-params (+ a z) (+ b (- n z))))
                          x (ge neanderthal-factory 1 200 (range 0.001 1 0.001))
                          x-pdf (pdf post-engine params x)
                          x-log-pdf (log-pdf post-engine params x)
                          x-beta-pdf (pdf beta-engine beta-params x)
                          x-beta-log-pdf (log-pdf beta-engine beta-params x)]

             (nrm2 (linear-frac! (axpy! -1 x-log-pdf x-beta-log-pdf) -32.61044)) => (roughly 0.0 0.0001)
             (evidence post-engine params x) => 1.6357374080751345E-15)))))))

#_(with-default

  (with-release [factory (ocl/gcn-bayadera-factory *context* *command-queue*)]
    (test-all factory)
    (test-dataset factory)
    (test-mcmc factory gaussian-model)))
