(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl.amd-gcn
  (:require [clojure.java.io :as io]
            [uncomplicate.commons.core
             :refer [Releaseable release with-release let-release
                     wrap-int wrap-float]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [max-compute-units max-work-group-size queue-device]]]
            [uncomplicate.clojurecl.toolbox
             :refer [enq-reduce enq-read-double count-work-groups]]
            [uncomplicate.neanderthal
             [core :refer [create-raw ncols mrows scal! transfer copy raw ecount]]
             [protocols :as np]
             [block :refer [buffer]]
             [opencl :refer [opencl-single]]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.opencl
             [utils :refer [with-philox get-tmp-dir-name]]
             [models :refer :all]
             [amd-gcn-stretch :refer [gcn-stretch-factory]]]))

(deftype GCNDirectSampler [cqueue prog ^long DIM]
  Releaseable
  (release [_]
    (release prog))
  RandomSampler
  (sample! [this seed cl-params n]
    (let-release [res (create-raw (np/factory cl-params) DIM n)]
      (with-release [sample-kernel (kernel prog "sample")]
        (set-args! sample-kernel 0 (buffer cl-params) (wrap-int seed) (buffer res))
        (enq-nd! cqueue sample-kernel (work-size-1d n))
        res))))

(deftype GCNDistributionEngine [ctx cqueue prog ^long WGS dist-model]
  Releaseable
  (release [_]
    (release prog))
  ModelProvider
  (model [_]
    dist-model)
  DistributionEngine
  (log-pdf [this cl-params x]
    (let-release [res (create-raw (np/factory cl-params) (ncols x))]
      (with-release [logpdf-kernel (kernel prog "logpdf")]
        (set-args! logpdf-kernel 0 (buffer cl-params) (buffer x) (buffer res))
        (enq-nd! cqueue logpdf-kernel (work-size-1d (ncols x)))
        res)))
  (pdf [this cl-params x]
    (let-release [res (create-raw (np/factory cl-params) (ncols x))]
      (with-release [pdf-kernel (kernel prog "pdf")]
        (set-args! pdf-kernel 0 (buffer cl-params) (buffer x) (buffer res))
        (enq-nd! cqueue pdf-kernel (work-size-1d (ncols x)))
        res)))
  (evidence [this cl-params x]
    (if (satisfies? LikelihoodModel dist-model)
      (let [n (ncols x)
            acc-size (* Double/BYTES (count-work-groups WGS n))]
        (with-release [evidence-kernel (kernel prog "evidence_reduce")
                       sum-reduction-kernel (kernel prog "sum_reduction")
                       cl-acc (cl-buffer ctx acc-size :read-write)]
          (set-args! evidence-kernel 0 cl-acc (buffer cl-params) (buffer x))
          (set-arg! sum-reduction-kernel 0 cl-acc)
          (enq-reduce cqueue evidence-kernel sum-reduction-kernel n WGS)
          (/ (enq-read-double cqueue cl-acc) n)))
      1.0)))

(deftype GCNDataSetEngine [ctx cqueue prog ^long WGS]
  Releaseable
  (release [_]
    (release prog))
  DatasetEngine
  (means [this data-matrix]
    (let [m (mrows data-matrix)
          n (ncols data-matrix)
          wgsn (min n WGS)
          wgsm (/ WGS wgsn)
          acc-size (* Float/BYTES (max 1 (* m (count-work-groups wgsn n))))]
      (let-release [res (create-raw (np/factory data-matrix) m)]
        (with-release [cl-acc (cl-buffer ctx acc-size :read-write)
                       sum-reduction-kernel (kernel prog "sum_reduction_horizontal")
                       mean-kernel (kernel prog "mean_reduce")]
          (set-arg! sum-reduction-kernel 0 cl-acc)
          (set-args! mean-kernel cl-acc (buffer data-matrix))
          (enq-reduce cqueue mean-kernel sum-reduction-kernel m n wgsm wgsn)
          (enq-copy! cqueue cl-acc (buffer res))
          (scal! (/ 1.0 n) (transfer res))))))
  (variances [this data-matrix]
    (let [m (mrows data-matrix)
          n (ncols data-matrix)
          wgsn (min n WGS)
          wgsm (/ WGS wgsn)
          acc-size (* Float/BYTES (max 1 (* m (count-work-groups wgsn n))))]
      (with-release [cl-acc (cl-buffer ctx acc-size :read-write)
                     res-vec (create-raw (np/factory data-matrix) m)
                     sum-reduction-kernel (kernel prog "sum_reduction_horizontal")
                     mean-kernel (kernel prog "mean_reduce")
                     variance-kernel (kernel prog "variance_reduce")]
        (set-arg! sum-reduction-kernel 0 cl-acc)
        (set-args! mean-kernel cl-acc (buffer data-matrix))
        (enq-reduce cqueue mean-kernel sum-reduction-kernel m n wgsm wgsn)
        (enq-copy! cqueue cl-acc (buffer res-vec))
        (scal! (/ 1.0 n) res-vec)
        (set-args! variance-kernel 0 cl-acc (buffer data-matrix) (buffer res-vec))
        (enq-reduce cqueue variance-kernel sum-reduction-kernel m n wgsm wgsn)
        (enq-copy! cqueue cl-acc (buffer res-vec))
        (scal! (/ 1.0 n) (transfer res-vec)))))
  EstimateEngine
  (histogram [this data-matrix]
    (let [m (mrows data-matrix)
          n (ncols data-matrix)
          wgsn (min n WGS)
          wgsm (/ WGS wgsn)
          acc-size (* 2 Float/BYTES (max 1 (* m (count-work-groups wgsn n))))]
      (with-release [cl-min-max (cl-buffer ctx acc-size :read-write)
                     uint-res (cl-buffer ctx (* Integer/BYTES WGS m) :read-write)
                     result (create-raw (np/factory data-matrix) WGS m)
                     limits (create-raw (np/factory data-matrix) 2 m)
                     bin-ranks (create-raw (np/factory data-matrix) WGS m)
                     min-max-reduction-kernel (kernel prog "min_max_reduction")
                     min-max-kernel (kernel prog "min_max_reduce")
                     histogram-kernel (kernel prog "histogram")
                     uint-to-real-kernel (kernel prog "uint_to_real")
                     local-sort-kernel (kernel prog "bitonic_local")]
        (set-arg! min-max-reduction-kernel 0 cl-min-max)
        (set-args! min-max-kernel cl-min-max (buffer data-matrix))
        (enq-reduce cqueue min-max-kernel min-max-reduction-kernel m n wgsm wgsn)
        (enq-copy! cqueue cl-min-max (buffer limits))
        (enq-fill! cqueue uint-res (wrap-int 0))
        (set-args! histogram-kernel
                   (buffer limits) (buffer data-matrix)
                   (wrap-int (ecount data-matrix)) uint-res)
        (enq-nd! cqueue histogram-kernel
                 (work-size-2d (mrows data-matrix) n wgsm wgsn))
        (set-args! uint-to-real-kernel
                   (wrap-float (/ WGS n)) (buffer limits)
                   uint-res (buffer result))
        (enq-nd! cqueue uint-to-real-kernel (work-size-2d WGS m))
        (set-args! local-sort-kernel (buffer result) (buffer bin-ranks))
        (enq-nd! cqueue local-sort-kernel (work-size-1d (* m WGS)))
        (->Estimate (transfer limits) (transfer result) (transfer bin-ranks))))))

(let [dataset-src [(slurp (io/resource "uncomplicate/clojurecl/kernels/reduction.cl"))
                   (slurp (io/resource "uncomplicate/bayadera/opencl/dataset/amd-gcn.cl"))
                   (slurp (io/resource "uncomplicate/bayadera/opencl/engines/estimate.cl"))]]

  (defn gcn-dataset-engine
    ([ctx cqueue ^long WGS]
     (let [prog (build-program! (program-with-source ctx dataset-src)
                                (format "-cl-std=CL2.0 -DREAL=float -DREAL2=float2 -DACCUMULATOR=float -DWGS=%d" WGS)
                                nil)]
       (->GCNDataSetEngine ctx cqueue prog WGS)))
    ([ctx queue]
     (gcn-dataset-engine ctx queue 256))))

(let [kernels-src (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/dist-kernels.cl"))]

  (defn gcn-distribution-engine
    ([ctx cqueue tmp-dir-name model WGS]
     (let [prog (build-program!
                 (program-with-source ctx (conj (source model) kernels-src))
                 (format "-cl-std=CL2.0 -DREAL=float -DACCUMULATOR=float -DLOGPDF=%s -DPARAMS_SIZE=%d -DDIM=%d -DWGS=%d -I%s"
                         (logpdf model) (params-size model) (dimension model)
                         WGS tmp-dir-name)
                 nil)]
       (->GCNDistributionEngine ctx cqueue prog WGS model)))))

(let [kernels-src (format "%s\n%s\n%s"
                          (slurp (io/resource "uncomplicate/clojurecl/kernels/reduction.cl"))
                          (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/lik-kernels.cl"))
                          (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/dist-kernels.cl")))]

  (defn gcn-posterior-engine
    ([ctx cqueue tmp-dir-name model WGS]
     (let [prog (build-program!
                 (program-with-source ctx (conj (source model) kernels-src))
                 (format "-cl-std=CL2.0 -DREAL=float -DLOGPDF=%s -DLOGLIK=%s -DPARAMS_SIZE=%d -DDIM=%d -DWGS=%d -I%s"
                         (logpdf model) (loglik model)
                         (params-size model) (dimension model) WGS tmp-dir-name)
                 nil)]
       (->GCNDistributionEngine ctx cqueue prog WGS model)))))

(defn gcn-direct-sampler
  ([ctx cqueue tmp-dir-name model WGS]
   (->GCNDirectSampler
    cqueue
    (build-program!
     (program-with-source ctx (into (source model) (sampler-source model)))
     (format "-cl-std=CL2.0 -DREAL=float -DWGS=%d -I%s/" WGS tmp-dir-name)
     nil)
    (dimension model))))

;; =========================== Distribution creators ===========================

(defrecord GCNBayaderaFactory [ctx cqueue tmp-dir-name
                               ^long compute-units ^long WGS
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
  (distribution-engine [_ model]
    (with-philox tmp-dir-name
      (gcn-distribution-engine ctx cqueue tmp-dir-name model WGS)))
  (posterior-engine [_ model]
    (with-philox tmp-dir-name
      (gcn-posterior-engine ctx cqueue tmp-dir-name model WGS)))
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
      (gcn-stretch-factory ctx cqueue tmp-dir-name
                           neanderthal-factory model WGS)))
  (processing-elements [_]
    (* compute-units WGS))
  DataSetFactory
  (dataset-engine [_]
    dataset-eng)
  np/FactoryProvider
  (factory [_]
    neanderthal-factory))

(defn gcn-bayadera-factory
  ([ctx cqueue ^long compute-units ^long WGS]
   (let [tmp-dir-name (get-tmp-dir-name)
         neanderthal-factory (opencl-single ctx cqueue)]
     (with-philox tmp-dir-name
       (->GCNBayaderaFactory
        ctx cqueue tmp-dir-name
        compute-units WGS
        (gcn-dataset-engine ctx cqueue WGS)
        neanderthal-factory
        gaussian-model
        (gcn-distribution-engine ctx cqueue tmp-dir-name gaussian-model WGS)
        (gcn-direct-sampler ctx cqueue tmp-dir-name gaussian-model WGS)
        uniform-model
        (gcn-distribution-engine ctx cqueue tmp-dir-name uniform-model WGS)
        (gcn-direct-sampler ctx cqueue tmp-dir-name uniform-model WGS)
        beta-model
        (gcn-distribution-engine ctx cqueue tmp-dir-name beta-model WGS)
        (gcn-stretch-factory ctx cqueue tmp-dir-name neanderthal-factory
                             beta-model WGS)
        binomial-model
        (gcn-distribution-engine ctx cqueue tmp-dir-name binomial-model WGS)
        (gcn-stretch-factory ctx cqueue tmp-dir-name neanderthal-factory
                             binomial-model WGS)))))
  ([ctx cqueue]
   (let [dev (queue-device cqueue)]
     (gcn-bayadera-factory ctx cqueue
                           (max-compute-units dev)
                           (max-work-group-size dev)))))
