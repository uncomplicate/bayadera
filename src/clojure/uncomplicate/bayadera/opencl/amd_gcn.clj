(ns uncomplicate.bayadera.opencl.amd-gcn
  (:require [clojure.java.io :as io]
            [me.raynes.fs :as fsc]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.clojurecl.toolbox
             :refer [enq-reduce enq-read-double count-work-groups
                     wrap-int wrap-float]]
            [uncomplicate.neanderthal
             [core :refer [dim]]
             [real :refer [sum]]
             [protocols :as np]
             [block :refer [buffer]]
             [native :refer [sv]]
             [opencl :refer [gcn-single]]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.opencl
             [utils :refer [with-philox get-tmp-dir-name]]
             [amd-gcn-stretch :refer [gcn-stretch-1d-factory]]])
  (:import [uncomplicate.bayadera.protocols CLDistributionModel CLLikelihoodModel]))

(let [posterior-kernels (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/posterior.cl"))
      posterior-logpdf (slurp (io/resource "uncomplicate/bayadera/opencl/templates/posterior.clt"))]
  (defn posterior [^CLLikelihoodModel likelihood
                   ^CLDistributionModel prior]
    (let [logpdf (str (gensym "logpdf"))
          params-size (+ (.params-size likelihood) (.params-size prior))]
      (->CLDistributionModel
       logpdf
       (.dimension prior)
       params-size
       (.lower prior)
       (.upper prior)
       (str (.functions prior) "\n" (.functions likelihood) "\n"
            (format posterior-logpdf
                    logpdf (.name likelihood)
                    (.name prior) (.params-size likelihood)))
       posterior-kernels))))

(deftype GCNDirectSampler [cqueue prog]
  Releaseable
  (release [_]
    (release prog))
  RandomSampler
  (sample! [this seed cl-params res]
    (with-release [sample-kernel (kernel prog "sample")]
      (set-args! sample-kernel 0 (buffer cl-params) (wrap-int seed) (buffer res))
      (enq-nd! cqueue sample-kernel (work-size-1d (dim res)))
      this)))

(deftype GCNDistributionEngine [cqueue prog]
  Releaseable
  (release [_]
    (release prog))
  DistributionEngine
  (logpdf! [this cl-params x res]
    (with-release [logpdf-kernel (kernel prog "logpdf")]
      (set-args! logpdf-kernel 0 (buffer cl-params) (buffer x) (buffer res))
      (enq-nd! cqueue logpdf-kernel (work-size-1d (dim x)))
      this))
  (pdf! [this cl-params x res]
    (with-release [pdf-kernel (kernel prog "pdf")]
      (set-args! pdf-kernel 0 (buffer cl-params) (buffer x) (buffer res))
      (enq-nd! cqueue pdf-kernel (work-size-1d (dim x)))
      this)))

(deftype GCNDataSetEngine [ctx cqueue prog ^long WGS]
  Releaseable
 (release [_]
    (release prog))
  Spread
  (mean-variance [this data-vect]
    (let [m (/ (sum data-vect) (dim data-vect))
          acc-size (* Double/BYTES (count-work-groups WGS (dim data-vect)))]
      (with-release [variance-kernel (kernel prog "variance_reduce")
                     sum-reduction-kernel (kernel prog "sum_reduction")
                     cl-acc (cl-buffer ctx acc-size :read-write)]
        (set-args! variance-kernel 0 cl-acc (buffer data-vect) (wrap-float m))
        (set-arg! sum-reduction-kernel 0 cl-acc)
        (enq-reduce cqueue variance-kernel sum-reduction-kernel WGS (dim data-vect))
        (sv m (/ (enq-read-double cqueue cl-acc) (dec (dim data-vect))))))))

(let [src [(slurp (io/resource "uncomplicate/clojurecl/kernels/reduction.cl"))
           (slurp (io/resource "uncomplicate/bayadera/opencl/dataset/amd-gcn.cl"))]]
  (defn gcn-dataset-engine
    ([ctx cqueue ^long WGS]
     (let [prog (build-program! (program-with-source ctx src)
                                (format "-cl-std=CL2.0 -DWGS=%s" WGS)
                                nil)]
       (->GCNDataSetEngine ctx cqueue prog WGS)))
    ([ctx queue]
     (gcn-dataset-engine ctx queue 256))))

(defn gcn-distribution-engine
  ([ctx cqueue tmp-dir-name ^CLDistributionModel model WGS]
   (let [prog (build-program!
               (program-with-source
                ctx [(.functions model) (.kernels model)])
               (format "-cl-std=CL2.0 -DLOGPDF=%s -DPARAMS_SIZE=%d -DWGS=%s -I%s"
                       (.name model) (.params-size model) WGS tmp-dir-name)
               nil)]
     (->GCNDistributionEngine cqueue prog)))
  ([ctx queue model]
   (gcn-distribution-engine ctx queue model 256)))

(defn gcn-direct-sampler
  ([ctx cqueue tmp-dir-name ^CLDistributionModel model WGS]
   (->GCNDirectSampler
    cqueue
    (build-program!
     (program-with-source ctx [(.functions model) (.kernels model)])
     (format "-cl-std=CL2.0 -DWGS=%s -I%s/" WGS tmp-dir-name)
     nil)))
  ([ctx cqueue model]
   (gcn-direct-sampler ctx cqueue model 256)))

;; =========================== Distribution creators ===========================

(defrecord GCNEngineFactory [ctx cqueue tmp-dir-name ^long WGS
                             dataset-eng neanderthal-factory
                             gaussial-model gaussian-eng gaussian-samp
                             uniform-model uniform-eng uniform-samp
                             beta-model beta-eng beta-samp
                             binomial-model binomial-eng binomial-samp]
  Releaseable
  (release [_]
    (and (release dataset-eng)
         (release neanderthal-factory)
         (release gaussian-eng)
         (release gaussian-samp)
         (release uniform-eng)
         (release uniform-samp)
         (release beta-eng)
         (release beta-samp)
         (release binomial-eng)
         (release binomial-samp)))
  DistributionEngineFactory
  (gaussian-engine [_]
    gaussian-eng)
  (uniform-engine [_]
    uniform-eng)
  (binomial-engine [_]
    binomial-eng)
  (beta-engine [_]
    beta-eng)
  (custom-engine [_ model]
    (with-philox tmp-dir-name
      (gcn-distribution-engine ctx cqueue tmp-dir-name model WGS)))
  SamplerFactory
  (gaussian-sampler [_]
    gaussian-samp)
  (uniform-sampler [_]
    uniform-samp)
  (binomial-sampler [_]
    binomial-samp)
  (beta-sampler [_]
    beta-samp)
  (mcmc-factory [_ model]
    (with-philox tmp-dir-name
      (gcn-stretch-1d-factory ctx cqueue tmp-dir-name model WGS)))
   DataSetFactory
  (dataset-engine [_]
    dataset-eng)
  np/FactoryProvider
  (factory [_]
    neanderthal-factory))

(let [gaussian-model
      (->CLDistributionModel "gaussian_logpdf" 1 2 nil nil
                             (str (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/uniform.h"))
                                  "\n"
                                  (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/gaussian.h")))
                             (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/gaussian.cl")))
      uniform-model
      (->CLDistributionModel "uniform_logpdf" 1 2 nil nil
                             (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/uniform.h"))
                             (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/uniform.cl")))
      beta-model
      (->CLDistributionModel  "beta_logpdf" 1 3 nil nil
                             (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/beta.h"))
                             (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/beta.cl")))
      binomial-model
      (->CLDistributionModel "binomial_logpdf" 1 3 nil nil
                             (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/binomial.h"))
                             (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/binomial.cl")))]

  (defn gcn-engine-factory
    ([ctx cqueue ^long WGS]
     (let [tmp-dir-name (get-tmp-dir-name)]
       (with-philox tmp-dir-name
         (->GCNEngineFactory
          ctx cqueue tmp-dir-name
          WGS
          (gcn-dataset-engine ctx cqueue WGS)
          (gcn-single ctx cqueue)
          gaussian-model
          (gcn-distribution-engine ctx cqueue tmp-dir-name gaussian-model WGS)
          (gcn-direct-sampler ctx cqueue tmp-dir-name gaussian-model WGS)
          uniform-model
          (gcn-distribution-engine ctx cqueue tmp-dir-name  uniform-model WGS)
          (gcn-direct-sampler ctx cqueue tmp-dir-name uniform-model WGS)
          beta-model
          (gcn-distribution-engine ctx cqueue tmp-dir-name beta-model WGS)
          (gcn-stretch-1d-factory ctx cqueue tmp-dir-name beta-model WGS)
          binomial-model
          (gcn-distribution-engine ctx cqueue tmp-dir-name binomial-model WGS)
          (gcn-stretch-1d-factory ctx cqueue tmp-dir-name binomial-model WGS)))))
    ([ctx cqueue]
     (gcn-engine-factory ctx cqueue 256))))
