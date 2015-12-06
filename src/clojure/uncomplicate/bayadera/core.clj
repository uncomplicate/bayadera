(ns uncomplicate.bayadera.core
  (:require [uncomplicate.clojurecl.toolbox :refer [wrap-int wrap-float]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [raw dim alter! create transfer! vect?]]
             [native :refer [sv]]
             [real :refer [entry]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [impl :refer :all]
             [special :refer [lnbeta]]]))

(defn dataset [factory src]
  (->UnivariateDataSet (p/dataset-engine factory)
                       (cond (number? src)
                             (create (np/factory factory) src)
                             (vect? src) src)))

(defn gaussian [factory ^double mu ^double sigma]
  (let [params (transfer! [mu sigma] (create (np/factory factory) 2))]
    (->GaussianDistribution factory
                            (p/gaussian-engine factory)
                            (->DirectSampler (np/factory factory) (p/gaussian-sampler factory)
                                             (rand-int Integer/MAX_VALUE) params)
                            params
                            mu sigma)))

(defn uniform [factory ^double a ^double b]
  (->UnivariateDistribution factory
                            (p/uniform-engine factory)
                            (p/uniform-sampler factory)
                            (->UniformMeasures a b)
                            (transfer! [a b] (create (np/factory factory) 2))))

(defn build-model [factory model]
  (let [engine (p/custom-engine factory model)
        sampler (p/mcmc-sampler factory model)]
    (fn [params]
      (->UnivariateDistribution factory
                                engine sampler nil
                                (transfer! params (create (np/factory factory) (dim params)))))))

(defn beta [factory ^double a ^double b]
  (let [params (transfer! [a b] (create (np/factory factory) 2))]
    (->UnivariateDistribution factory
                              (p/beta-engine factory)
                              (p/mcmc-engine (p/beta-sampler factory) (* 44 256 32) (sv a b (lnbeta a b)) 0 1);;TODO params!!!!!
                              (->BetaMeasures a b)
                              params)))

(defn mean-variance [x]
  (p/mean-variance (p/measures x)))

(defn mean-sd [x]
  (alter! (p/mean-variance (p/measures x)) 1 sqrt))

(defn mean [x]
  (p/mean (p/measures x)))

(defn variance [x]
  (p/variance (p/measures x)))

(defn sd [x]
  (p/sd (p/measures x)))

(defn parameters [dist]
  (p/parameters dist))

(defn sample! [dist result]
  (p/sample! (p/sampler dist) (rand-int Integer/MAX_VALUE) (p/parameters dist) (p/data result)))

;; TODO each built-in distribution should have a pre-configured mcmc sampler. Start with beta,
;; but this code should be built into the Distribution/Sampler protocols
(defn mcmc-sampler [dist]
  (let [mcmc-sampler (p/sampler dist)]
    (p/set-position! mcmc-sampler (wrap-int (rand-int Integer/MAX_VALUE)))
    (p/init! mcmc-sampler (wrap-int (rand-int Integer/MAX_VALUE)))
    (p/burn-in! mcmc-sampler 1024 (wrap-float 8.0))
    (p/init! mcmc-sampler (wrap-int (rand-int Integer/MAX_VALUE)))
    (p/run-sampler! mcmc-sampler 256 (wrap-float 8.0))
    mcmc-sampler))

(defn sampler [dist]
  (p/sampler dist))

(defn mcmc-sample [sampler n]
  (p/sample! sampler n))

#_(defn sample! [sampler result]
  (p/sample! sampler (p/data result)))

(defn sample [dist n]
  (let [result (dataset (p/factory dist) n)]
    (sample! dist result)
    result))

;;TODO later
#_(defn pdf! [dist xs result]
  (p/pdf! (p/engine dist) xs result))

#_(defn pdf [dist xs]
  (let [result (raw xs)]
    (pdf! dist xs result)
    result))
