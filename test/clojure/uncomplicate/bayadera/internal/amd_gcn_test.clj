(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.amd-gcn-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release wrap-int wrap-float]]
            [uncomplicate.fluokitten.core :refer [fmap! op]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [queue-device max-compute-units max-work-group-size]]
             [toolbox :refer [count-work-groups]]]
            [uncomplicate.neanderthal
             [core :refer [vctr ge native native! subvector sum entry imax imin row raw copy axpy! nrm2]]
             [vect-math :refer [linear-frac!]]
             [opencl :refer [opencl-float]]
             [block :refer :all]]
            [uncomplicate.bayadera
             [distributions :refer [gaussian-pdf gaussian-log-pdf binomial-lik-params beta-params]]
             [opencl :refer :all :as ocl :exclude [gcn-bayadera-factory]]]
            [uncomplicate.bayadera.internal
             [protocols :refer :all]
             [amd-gcn :refer :all]
             [util :refer [create-tmp-dir with-philox]]]
            [uncomplicate.bayadera.core-test :refer [test-all]]
            [uncomplicate.bayadera.internal.device-test :refer :all])
  (:import [uncomplicate.bayadera.internal.amd_gcn GCNStretch]))

(with-default
  (let [dev (queue-device *command-queue*)
        wgs 256
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
             (mean sample-1d) => 1.50084453125)))

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
        wgs 256
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

(with-default
  (let [dev (queue-device *command-queue*)
        wgs 256
        tmp-dir-name (create-tmp-dir)
        walker-count (* (max-compute-units dev) wgs)
        means-count (long (count-work-groups wgs (/ walker-count 2)))
        acc-count (long (count-work-groups wgs means-count))
        seed 123
        a 2.0]
    (with-philox tmp-dir-name
      (with-release [neanderthal-factory (opencl-float *context* *command-queue*)]
        (facts
         "OpenCL GCN stretch with Uniform model."
         (with-release [params (vctr neanderthal-factory [-1 2])
                        limits (ge neanderthal-factory 2 1 [-1 2])
                        uniform-sampler (mcmc-sampler (gcn-stretch-factory
                                                       *context* *command-queue* tmp-dir-name
                                                       neanderthal-factory uniform-model wgs)
                                                      walker-count params)]
           (let [stretch-move-bare-kernel (.stretch-move-odd-bare-kernel ^GCNStretch uniform-sampler)
                 stretch-move-kernel (.stretch-move-odd-kernel ^GCNStretch uniform-sampler)
                 sum-means-kernel (.sum-means-kernel ^GCNStretch uniform-sampler)
                 cl-params (.cl-params ^GCNStretch uniform-sampler)
                 cl-xs (.cl-xs ^GCNStretch uniform-sampler)
                 cl-s0 (.cl-s0 ^GCNStretch uniform-sampler)
                 cl-logfn-s0 (.cl-logfn-s0 ^GCNStretch uniform-sampler)
                 cl-s1 (.cl-s1 ^GCNStretch uniform-sampler)
                 cl-logfn-s1 (.cl-logfn-s1 ^GCNStretch uniform-sampler)
                 cl-accept (.cl-accept ^GCNStretch uniform-sampler)
                 cl-means-acc (create-data-source neanderthal-factory means-count)
                 means-acc-array (float-array 10)
                 accept-array (int-array 10)
                 acc (ge neanderthal-factory 1 acc-count)
                 cqueue (.cqueue ^GCNStretch uniform-sampler)]

             (init-position! uniform-sampler seed limits)
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [1.0883402824401855 1.3920516967773438 -0.8245221972465515 0.47322529554367065]
             (init! uniform-sampler seed) => uniform-sampler
             (move-bare! uniform-sampler) => uniform-sampler
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [0.7279692888259888 1.8140764236450195 0.04031801223754883 0.4697103500366211]
             (move-bare! uniform-sampler)
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [1.0403573513031006 1.4457943439483643 0.37618488073349 1.5768483877182007]

             (init-position! uniform-sampler seed limits)
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [1.0883402824401855 1.3920516967773438 -0.8245221972465515 0.47322529554367065]

             (set-args! stretch-move-bare-kernel 0 (wrap-int seed) (wrap-int 3333)
                        cl-params cl-s1 cl-s0 cl-logfn-s0 (wrap-float a) (wrap-float 1.0) (wrap-int 0))
             (enq-nd! cqueue stretch-move-bare-kernel (work-size-1d (/ walker-count 2)))
             (set-args! stretch-move-bare-kernel 0 (wrap-int (inc seed)) (wrap-int 4444)
                        cl-params cl-s0 cl-s1 cl-logfn-s1 (wrap-float a) (wrap-float 1.0) (wrap-int 0))
             (enq-nd! cqueue stretch-move-bare-kernel (work-size-1d (/ walker-count 2)))
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [0.7279692888259888 1.8140764236450195 0.04031801223754883 0.4697103500366211]

             (set-args! stretch-move-bare-kernel 0 (wrap-int seed) (wrap-int 3333)
                        cl-params cl-s1 cl-s0 cl-logfn-s0 (wrap-float a) (wrap-float 1.0) (wrap-int 1))
             (enq-nd! cqueue stretch-move-bare-kernel (work-size-1d (/ walker-count 2)))
             (set-args! stretch-move-bare-kernel 0 (wrap-int (inc seed)) (wrap-int 4444)
                        cl-params cl-s0 cl-s1 cl-logfn-s1 (wrap-float a) (wrap-float 1.0) (wrap-int 1))
             (enq-nd! cqueue stretch-move-bare-kernel (work-size-1d (/ walker-count 2)))
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [1.0403573513031006 1.4457943439483643 0.37618488073349 1.5768483877182007]

             (enq-fill! cqueue cl-accept (int-array 1))
             (enq-fill! cqueue cl-means-acc (float-array 1))
             (set-args! stretch-move-kernel 0 (wrap-int seed) (wrap-int 1111)
                        cl-params cl-s1 cl-s0 cl-logfn-s0 cl-accept cl-means-acc (wrap-float a)
                        (wrap-int 0))
             (enq-nd! cqueue stretch-move-kernel (work-size-1d (/ walker-count 2)))
             (set-args! stretch-move-kernel 0 (wrap-int (inc seed)) (wrap-int 2222)
                        cl-params cl-s0 cl-s1 cl-logfn-s1 cl-accept cl-means-acc (wrap-float a)
                        (wrap-int 0))
             (enq-nd! cqueue stretch-move-kernel (work-size-1d (/ walker-count 2)))
             (take 4 (native (row (sample uniform-sampler) 0)))
             => [1.0110803842544556 1.615005373954773 0.3426262140274048 1.4122662544250488]
             (enq-read! cqueue cl-means-acc means-acc-array)
             (seq means-acc-array) => (map float [269.26575 286.35892 288.0937 240.0009 265.76953
                                                  274.17465 257.67914 302.7213 244.6228 277.85284])
             (enq-read! cqueue cl-accept accept-array)
             (seq accept-array) => [423 422 424 428 414 439 428 409 429 409]

             (set-args! sum-means-kernel 0 (buffer acc) cl-means-acc)
             (enq-nd! cqueue sum-means-kernel (work-size-2d 1 means-count))
             (sum (native acc)) => (float 5822.9175)

             )))

        (facts
         "OpenCL GCN stretch with Gaussian model."
         (with-release [params (vctr neanderthal-factory [3 1.0])
                        limits (ge neanderthal-factory 2 1 [-7 7])
                        gaussian-sampler (mcmc-sampler (gcn-stretch-factory
                                                        *context* *command-queue* tmp-dir-name
                                                        neanderthal-factory gaussian-model wgs)
                                                       walker-count params)]
           (init-position! gaussian-sampler seed limits)
           (take 4 (native (row (sample gaussian-sampler) 0)))
           => [2.7455875873565674 4.162908554077148 -6.181103706359863 -0.12494850158691406]
           (init! gaussian-sampler seed) => gaussian-sampler
           (move-bare! gaussian-sampler) => gaussian-sampler
           (take 4 (native (row (sample gaussian-sampler) 0)))
           => [2.7455875873565674 4.162908554077148 -2.1451826095581055 -0.14135169982910156]
           (move-bare! gaussian-sampler)
           (take 4 (native (row (sample gaussian-sampler) 0)))
           => [3.962130546569824 2.9586503505706787 -0.5778036117553711 5.02529239654541]))

        (with-release [neanderthal-factory (opencl-float *context* *command-queue*)]
          (facts
           "OpenCL GCN stretch burn-in with Gaussian model."
           (with-release [params (vctr neanderthal-factory [3 1.0])
                          limits (ge neanderthal-factory 2 1 [-7 7])
                          gaussian-sampler (mcmc-sampler (gcn-stretch-factory
                                                          *context* *command-queue* tmp-dir-name
                                                          neanderthal-factory gaussian-model wgs)
                                                         walker-count params)]
             (init-position! gaussian-sampler seed limits)
             (init! gaussian-sampler (inc seed))
             (burn-in! gaussian-sampler 100 1.5)
             (first (native! (mean (sample gaussian-sampler)))) => 2.9559195041656494
             (take-nth 1500 (native! (row (sample gaussian-sampler) 0)))
             => [3.222665309906006 2.4721696376800537 3.094174385070801 3.0374388694763184
                 5.753951072692871 2.7033936977386475 2.5397140979766846 3.4444146156311035]
             (first (native! (sd (sample gaussian-sampler)))) => 0.9953131079673767)))))))

(with-default
  (let [dev (queue-device *command-queue*)
        wgs 256
        tmp-dir-name (create-tmp-dir)
        walker-count (* 2 44 wgs)
        seed 123
        a 8.0]
    (with-philox tmp-dir-name
      (with-release [bayadera-factory (ocl/gcn-bayadera-factory *context* *command-queue*)]
        (let [mcmc-engine-factory (mcmc-factory bayadera-factory gaussian-model)]
          (with-release [params (vctr bayadera-factory [200 1])
                         limits (ge bayadera-factory 2 1 [180.0 220.0])]
            (let [engine (mcmc-sampler mcmc-engine-factory walker-count params)]
              (facts
               "Test MCMC stretch engine."
               (init-position! engine 123 limits)
               (init! engine 1243)
               (burn-in! engine 5120 a)
               (acc-rate! engine a) => 0.4808682528409091
               (map println (:autocorrelation (run-sampler! engine 67 a))) => :a
               (entry (:tau (:autocorrelation (run-sampler! engine 67 a))) 0) => 5.757689952850342
               ))))

        ))))

#_(with-default
  (with-release [factory (ocl/gcn-bayadera-factory *context* *command-queue*)]
    (test-all factory)
    (test-dataset factory)
    (test-mcmc factory gaussian-model)))
