(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.amd-gcn-test
  (:require [midje.sweet :refer :all]
            [clojure.java.io :as io]
            [clojure.string :refer [split-lines]]
            [uncomplicate.commons
             [core :refer [with-release wrap-int wrap-float]]
             [utils :refer [count-groups]]]
            [uncomplicate.fluokitten.core :refer [fmap! op]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [queue-device max-compute-units max-work-group-size]]]
            [uncomplicate.neanderthal
             [core :refer [vctr ge native native! subvector sum entry imax imin row raw
                           copy axpy! nrm2 transfer! col]]
             [vect-math :refer [linear-frac!]]
             [opencl :refer [opencl-float]]
             [block :refer :all]]
            [uncomplicate.neanderthal.internal.device.random123 :refer [temp-dir]]
            [uncomplicate.bayadera
             [distributions :refer [gaussian-pdf gaussian-log-pdf binomial-lik-params beta-params]]
             [opencl :refer :all]]
            [uncomplicate.bayadera.internal.protocols :refer :all]
            [uncomplicate.bayadera.internal.device
             [models :as models]
             [amd-gcn :refer :all]]
            [uncomplicate.bayadera.core-test :refer [test-dataset]]
            [uncomplicate.bayadera.library-test :refer [test-all]])
  (:import uncomplicate.bayadera.internal.device.amd_gcn.GCNStretch))

(with-default
  (let [dev (queue-device *command-queue*)
        wgs 256]
    (with-release [neanderthal-factory (opencl-float *context* *command-queue*)
                   distributions (models/distribution-models source-library)
                   likelihoods (models/likelihood-models source-library)]

      (facts

       "OpenCL GCN direct sampler for Uniform distribution"
       (with-release [uniform-model (deref (distributions :uniform))
                      uniform-sampler (gcn-direct-sampler-engine *context* *command-queue*
                                                                 temp-dir uniform-model wgs)
                      params (vctr neanderthal-factory [-99.9 200.1])
                      smpl (sample uniform-sampler 123 params (ge neanderthal-factory 1 10000))
                      native-sample (native smpl)]
         (let [sample-1d (row native-sample 0)]
           (seq (subvector sample-1d 0 4))
           => [108.93402099609375 139.30517578125 -82.35221862792969 47.42252731323242]
           (seq (subvector sample-1d 9996 4))
           => [17.43636131286621 151.42117309570312 42.78262710571289 107.87583923339844]
           (entry sample-1d (imax sample-1d)) => 200.0757293701172
           (entry sample-1d (imin sample-1d)) => -99.86572265625
           (mean sample-1d) => 51.331178125)))

      (facts
       "OpenCL GCN direct sampler for Gaussian distribution"
       (with-release [gaussian-model (deref (distributions :gaussian))
                      gaussian-sampler (gcn-direct-sampler-engine *context* *command-queue*
                                                                  temp-dir gaussian-model wgs)
                      params (vctr neanderthal-factory [100 200.1])
                      smpl (sample gaussian-sampler 123 params (ge neanderthal-factory 1 10000))
                      native-sample (native smpl)]
         (let [sample-1d (row native-sample 0)]
           (seq (subvector sample-1d 0 4))
           => [-27.020862579345703 55.27095031738281 185.7442169189453 322.7049560546875]
           (seq (subvector sample-1d 9996 4))
           => [175.25137329101562 7.720747470855713 126.18175506591797 -69.49845886230469]
           (entry sample-1d (imax sample-1d)) => 868.6443481445312
           (entry sample-1d (imin sample-1d)) => -610.0803833007812
           (mean sample-1d) => 95.1347296875)))

      (facts
       "OpenCL GCN direct sampler for Erlang distribution"
       (with-release [erlang-model (deref (distributions :erlang))
                      erlang-sampler (gcn-direct-sampler-engine *context* *command-queue*
                                                                temp-dir erlang-model wgs)
                      params (vctr neanderthal-factory [2 3])
                      smpl (sample erlang-sampler 123 params (ge neanderthal-factory 1 10000))
                      native-sample (native smpl)]
         (let [sample-1d (row native-sample 0)]
           (seq (subvector sample-1d 0 4))
           => [1.571028709411621 1.4484457969665527 0.7983559370040894 1.1712465286254883]
           (seq (subvector sample-1d 9996 4))
           => [0.7548800706863403 2.28580379486084 1.1975524425506592 1.3439302444458008]
           (entry sample-1d (imax sample-1d)) => 7.159594535827637
           (entry sample-1d (imin sample-1d)) => 0.0482538677752018
           (mean sample-1d) => 1.50084443359375)))

      (facts
       "OpenCL GCN direct sampler for Exponential distribution"
       (with-release [exponential-model (deref (distributions :exponential))
                      exponential-sampler (gcn-direct-sampler-engine *context* *command-queue*
                                                                     temp-dir exponential-model wgs)
                      params (vctr neanderthal-factory [4])
                      smpl (sample exponential-sampler 123 params (ge neanderthal-factory 1 10000))
                      native-sample (native smpl)]
         (let [sample-1d (row native-sample 0)]
           (seq (subvector sample-1d 0 4))
           => [0.29777517914772034 0.39906948804855347 0.015068267472088337 0.1688637137413025]
           (seq (subvector sample-1d 9996 4))
           => [0.12403401732444763 0.4546346068382263 0.16137930750846863 0.29489007592201233]
           (entry sample-1d (imax sample-1d)) => 2.3556253910064697
           (entry sample-1d (imin sample-1d)) => 2.8567157642100938E-5
           (mean sample-1d) => 0.2526310943603516))))))

(with-default
  (let [dev (queue-device *command-queue*)
        wgs 256]
    (with-release [neanderthal-factory (opencl-float *context* *command-queue*)
                   distributions (models/distribution-models source-library)
                   likelihoods (models/likelihood-models source-library)]
      (facts
       "OpenCL GCN distribution engine with Gaussian distribution"
       (with-release [gaussian-model (deref (distributions :gaussian))
                      gaussian-engine (gcn-distribution-engine *context* *command-queue*
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
       "OpenCL GCN posterior engine with Beta-Binomial model."
       (let [n 50
             z 15
             a 3
             b 2]
         (with-release [beta-model (deref (distributions :beta))
                        binomial-lik-model (deref (likelihoods :binomial))
                        post-engine (gcn-distribution-engine
                                     *context* *command-queue*
                                     (posterior-model beta-model "beta_binomial" binomial-lik-model)
                                     wgs)
                        beta-engine
                        (gcn-distribution-engine *context* *command-queue* beta-model wgs)
                        binomial-lik-engine;;TODO prettify
                        (gcn-likelihood-engine *context* *command-queue* binomial-lik-model wgs)
                        lik-params (vctr neanderthal-factory (binomial-lik-params n z))
                        params (vctr neanderthal-factory (op (binomial-lik-params n z) (beta-params a b)))
                        beta-params (vctr neanderthal-factory (beta-params (+ a z) (+ b (- n z))))
                        x (ge neanderthal-factory 1 200 (range 0.001 1 0.001))
                        x-pdf (density post-engine params x)
                        x-log-pdf (log-density post-engine params x)
                        x-beta-pdf (density beta-engine beta-params x)
                        x-beta-log-pdf (log-density beta-engine beta-params x)]

           (nrm2 (linear-frac! (axpy! -1 x-log-pdf x-beta-log-pdf) -32.61044)) => (roughly 0.0 0.0001)
           (evidence binomial-lik-engine lik-params x) => 1.6357375222552396E-15))))))

(with-default
  (let [dev (queue-device *command-queue*)
        wgs 256
        walker-count (* 44 wgs)
        means-count (long (count-groups wgs (/ walker-count 2)))
        acc-count (long (count-groups wgs means-count))
        seed 123
        a 2.0]
    (with-release [neanderthal-factory (opencl-float *context* *command-queue*)
                   distributions (models/distribution-models source-library)]
      (facts
       "OpenCL GCN stretch with Uniform model."
       (with-release [uniform-model (deref (distributions :uniform))
                      params (vctr neanderthal-factory [-1 2])
                      res (ge neanderthal-factory 1 walker-count {:raw true})
                      limits (ge neanderthal-factory 2 1 [-1 2])
                      uniform-sampler (create-sampler
                                       (gcn-stretch-factory
                                        *context* *command-queue* temp-dir
                                        neanderthal-factory nil uniform-model wgs)
                                       seed walker-count params)]
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

           (init! uniform-sampler seed) => uniform-sampler
           (init-position! uniform-sampler seed limits)
           (take 4 (native (row (sample! uniform-sampler) 0)))
           => [0.7279692888259888 1.81407630443573 0.040318019688129425 0.4697103202342987]

           (take 4 (native (row (sample! uniform-sampler) 0)))
           => [1.040357232093811 1.4457943439483643 0.3761849105358124 1.5768483877182007]

           (take 4 (native (row (sample! uniform-sampler) 0)))
           => [1.047235131263733 1.1567966938018799 0.6802869439125061 1.6528078317642212]

           (take 4 (native (row (sample! uniform-sampler) 0)))
           => [0.9332534670829773 1.9495460987091064 0.5958949327468872 1.6429908275604248]

           (init! uniform-sampler seed) => uniform-sampler
           (init-position! uniform-sampler seed limits)

           (set-args! stretch-move-bare-kernel 0 (wrap-int seed) (wrap-int 3333) (wrap-int 0) (wrap-int 2)
                      cl-params cl-s1 cl-s0 cl-logfn-s0 (wrap-float a) (wrap-float 1.0) (wrap-int 0))
           (enq-kernel! cqueue stretch-move-bare-kernel (work-size-1d (/ walker-count 2)))
           (set-args! stretch-move-bare-kernel 0 (wrap-int (inc seed)) (wrap-int 4444) (wrap-int 0) (wrap-int 2)
                      cl-params cl-s0 cl-s1 cl-logfn-s1 (wrap-float a) (wrap-float 1.0) (wrap-int 0))
           (enq-kernel! cqueue stretch-move-bare-kernel (work-size-1d (/ walker-count 2)))
           (enq-copy! cqueue cl-xs (buffer res))
           (take 4 (native (row res 0)))
           => [0.7279692888259888 1.81407630443573 0.040318019688129425 0.4697103202342987]

           (set-args! stretch-move-bare-kernel 0 (wrap-int seed) (wrap-int 3333) (wrap-int 0) (wrap-int 2)
                      cl-params cl-s1 cl-s0 cl-logfn-s0 (wrap-float a) (wrap-float 1.0) (wrap-int 1))
           (enq-kernel! cqueue stretch-move-bare-kernel (work-size-1d (/ walker-count 2)))
           (set-args! stretch-move-bare-kernel 0 (wrap-int (inc seed)) (wrap-int 4444) (wrap-int 0) (wrap-int 2)
                      cl-params cl-s0 cl-s1 cl-logfn-s1 (wrap-float a) (wrap-float 1.0) (wrap-int 1))
           (enq-kernel! cqueue stretch-move-bare-kernel (work-size-1d (/ walker-count 2)))
           (enq-copy! cqueue cl-xs (buffer res))
           (take 4 (native (row res 0)))
           => [1.040357232093811 1.4457943439483643 0.3761849105358124 1.5768483877182007]

           (enq-fill! cqueue cl-accept (int-array 1))
           (enq-fill! cqueue cl-means-acc (float-array 1))
           (set-args! stretch-move-kernel 0 (wrap-int seed) (wrap-int 1111) (wrap-int 0) (wrap-int 2)
                      cl-params cl-s1 cl-s0 cl-logfn-s0 cl-accept cl-means-acc (wrap-float a)
                      (wrap-int 0))
           (enq-kernel! cqueue stretch-move-kernel (work-size-1d (/ walker-count 2)))
           (set-args! stretch-move-kernel 0 (wrap-int (inc seed)) (wrap-int 2222) (wrap-int 0) (wrap-int 2)
                      cl-params cl-s0 cl-s1 cl-logfn-s1 cl-accept cl-means-acc (wrap-float a)
                      (wrap-int 0))
           (enq-kernel! cqueue stretch-move-kernel (work-size-1d (/ walker-count 2)))
           (enq-copy! cqueue cl-xs (buffer res))
           (take 4 (native (row res 0)))
           => [1.011080265045166 1.615005373954773 0.3426262140274048 1.4122663736343384]
           (enq-read! cqueue cl-means-acc means-acc-array)
           (seq means-acc-array) => (map float [269.26575 286.3589 288.09372 240.0009 265.76953
                                                274.17465 257.67914 302.7213 244.6228 277.85284])
           (enq-read! cqueue cl-accept accept-array)
           (seq accept-array) => [423 422 424 428 414 439 428 409 429 409]

           (set-args! sum-means-kernel 0 (buffer acc) cl-means-acc)
           (enq-kernel! cqueue sum-means-kernel (work-size-2d 1 means-count))
           (sum acc) => (float 5822.918))))

      (facts
       "OpenCL GCN stretch with Gaussian model."
       (with-release [gaussian-model (deref (distributions :gaussian))
                      params (vctr neanderthal-factory [3 1.0])
                      limits (ge neanderthal-factory 2 1 [-7 7])
                      gaussian-sampler (create-sampler (gcn-stretch-factory
                                                        *context* *command-queue* temp-dir
                                                        neanderthal-factory nil gaussian-model wgs)
                                                       seed walker-count params)]
         (init! gaussian-sampler seed) => gaussian-sampler
         (init-position! gaussian-sampler seed limits)
         (take 4 (native (row (sample! gaussian-sampler) 0)))
         => [2.7455878257751465 4.16290807723999 -2.1451826095581055 -0.14135171473026276]
         (take 4 (native (row (sample! gaussian-sampler) 0)))
         => [3.9621310234069824 2.9586496353149414 -0.5778038501739502 5.025292873382568]
         (take 4 (native (row (sample! gaussian-sampler) 0)))
         => [3.8151602745056152 2.415064573287964 0.7977100610733032 5.292686939239502]))

      (with-release [neanderthal-factory (opencl-float *context* *command-queue*)]
        (facts
         "OpenCL GCN stretch burn-in with Gaussian model."
         (with-release [gaussian-model (deref (distributions :gaussian))
                        params (vctr neanderthal-factory [3 1.0])
                        limits (ge neanderthal-factory 2 1 [-7 7])
                        gaussian-sampler (create-sampler (gcn-stretch-factory
                                                          *context* *command-queue* temp-dir
                                                          neanderthal-factory nil gaussian-model wgs)
                                                         seed walker-count params)]
           (init! gaussian-sampler (inc seed))
           (init-position! gaussian-sampler seed limits)
           (burn-in! gaussian-sampler 100 1.5)
           (first (native! (mean (sample! gaussian-sampler)))) => 2.9549477100372314
           (take-nth 1500 (native! (row (sample! gaussian-sampler) 0)))
           => [3.3298428058624268 2.3125054836273193 3.583237409591675 3.3418703079223633
               4.829986572265625 2.704503059387207 2.064406156539917 3.443021535873413]
           (first (native! (sd (sample! gaussian-sampler)))) => 0.995916485786438))))))

(with-default
  (let [dev (queue-device *command-queue*)
        wgs 256
        walker-count (* 2 64 wgs)
        seed 123
        a 8.0]
    (with-release [distributions (models/distribution-models source-library)
                   gaussian-model (deref (distributions :gaussian))
                   bayadera-factory (gcn-bayadera-factory *context* *command-queue* 64 wgs)]
      (let [mcmc-engine-factory (mcmc-factory bayadera-factory gaussian-model)]
        (with-release [params (vctr bayadera-factory [200 1])
                       limits (ge bayadera-factory 2 1 [180.0 220.0])
                       dummy-sample-matrix (ge bayadera-factory 1 (* 100 walker-count) (cycle [201 199 138]))]
          (let [engine (create-sampler mcmc-engine-factory seed walker-count params)]
            (facts
             "Test MCMC stretch engine."
             (init! engine 1243)
             (init-position! engine 123 limits)
             (burn-in! engine 5120 a)
             (acc-rate! engine a) => 0.48345947265625
             (entry (:tau (:autocorrelation (run-sampler! engine 670 a))) 0) => 8.159395217895508)))))))

(with-default
  (let [dev (queue-device *command-queue*)
        wgs 256
        seed 123
        a 8.0]
    (with-release [dev-queue (command-queue *context* (queue-device *command-queue*)
                                            :queue-on-device-default :queue-on-device
                                            :out-of-order-exec-mode)
                   neanderthal-factory (opencl-float *context* *command-queue*)
                   engine (gcn-acor-engine *context* *command-queue* wgs)
                   data-matrix-67
                   (let [data (map (comp float read-string)
                                   (split-lines (slurp (io/resource "uncomplicate/bayadera/internal/acor-data-67"))))
                         matrix (ge neanderthal-factory 2 67)]
                     (transfer! data (row matrix 0))
                     (transfer! (map (partial * 2) data) (row matrix 1))
                     matrix)
                   data-matrix-367
                   (let [data (map (comp float read-string)
                                   (split-lines (slurp (io/resource "uncomplicate/bayadera/internal/acor-data-367"))))
                         matrix (ge neanderthal-factory 2 367)]
                     (transfer! data (row matrix 0))
                     (transfer! (map (partial + 1) data) (row matrix 1))
                     matrix)
                   data-matrix-112640
                   (let [data (map (comp float read-string)
                                   (split-lines (slurp (io/resource "uncomplicate/bayadera/internal/acor-data-112640"))))
                         matrix (ge neanderthal-factory 2 112640)]
                     (transfer! data (row matrix 0))
                     (transfer! (map (partial * 2) data) (row matrix 1))
                     matrix)]
      (facts
       "Test MCMC acor."
       (seq (:tau (acor engine data-matrix-67))) => [10.66264820098877 10.66264820098877]
       (seq (:tau (acor engine data-matrix-367))) => [17.302553176879883 17.302553176879883]
       (let [autocorrelation (acor engine data-matrix-112640)]
         (let [autocorrelation (acor engine data-matrix-112640)]
           (second (:tau autocorrelation)) => (roughly 20.41 0.001)
           (entry (:sigma autocorrelation) 0) => (roughly 0.009 0.001)))))))

(with-default
  (with-release [factory (gcn-bayadera-factory *context* *command-queue*)
                 library (device-library factory)]
    (test-all library)))
