(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.nvidia-gtx-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecuda
             [core :refer :all :exclude [parameters]]
             [info :refer [ctx-device max-block-dim-x driver-version]]]
            [uncomplicate.neanderthal
             [core :refer [vctr native subvector row entry imax imin]]
             [cuda :refer [cuda-float]]]
            [uncomplicate.bayadera.cuda :refer :all :exclude [gtx-bayadera-factory]]
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
                      cu-params (vctr neanderthal-factory [-99.9 200.1])
                      smpl (sample normal-sampler 123 cu-params 10000)
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
       "OpenCL GCN direct sampler for Gaussian distribution"
       (with-release [gaussian-sampler (gtx-direct-sampler (current-context) default-stream
                                                           gaussian-model wgs cudart-version)
                      cu-params (vctr neanderthal-factory [100 200.1])
                      smpl (sample gaussian-sampler 123 cu-params 10000)
                      native-sample (native smpl)]
         (let [sample-1d (row native-sample 0)]
           (seq (subvector sample-1d 0 4))
           => (list -27.02083969116211 55.2710075378418 185.74417114257812 322.7049255371094)
           (seq (subvector sample-1d 9996 4))
           => (list 175.25135803222656 7.720771312713623 126.18173217773438 -69.4984359741211)
           (entry sample-1d (imax sample-1d)) => 868.6444702148438
           (entry sample-1d (imin sample-1d)) => -610.0802612304688
           (mean sample-1d) => 95.13473125)
         ))
      )))

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
