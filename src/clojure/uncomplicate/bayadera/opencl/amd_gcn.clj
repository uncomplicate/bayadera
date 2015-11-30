(ns uncomplicate.bayadera.opencl.amd-gcn
  (:require [clojure.java.io :as io]
            [me.raynes.fs :as fsc]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.clojurecl.toolbox
             :refer [enq-reduce enq-read-double count-work-groups
                     wrap-int wrap-float]]
            [uncomplicate.neanderthal
             [core :refer [dim sum nrm2]]
             [protocols :as np]
             [block :refer [buffer]]
             [native :refer [sv]]
             [opencl :refer [gcn-single]]]
            [uncomplicate.bayadera.protocols :refer :all]))

(defrecord CLDistributionModel [pdf logpdf functions kernels])

(deftype GCNDirectSampler [cqueue prog]
  Releaseable
  (release [_]
    (release prog))
  RandomSampler
  (sample! [this seed params res]
    (with-release [sample-kernel (kernel prog "sample")]
      (set-args! sample-kernel 0 (buffer params) (wrap-int seed) (buffer res))
      (enq-nd! cqueue sample-kernel (work-size-1d (dim res)))
      this)))

(deftype GCNDistributionEngine [cqueue prog]
  Releaseable
  (release [_]
    (release prog))
  DistributionEngine
  (logpdf! [this params x res]
    (with-release [logpdf-kernel (kernel prog "logpdf")]
      (set-args! logpdf-kernel 0 (buffer params) (buffer x) (buffer res))
      (enq-nd! cqueue logpdf-kernel (work-size-1d (dim x)))
      this))
  (pdf! [this params x res]
    (with-release [pdf-kernel (kernel prog "pdf")]
      (set-args! pdf-kernel 0 (buffer params) (buffer x) (buffer res))
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
           (slurp (io/resource "uncomplicate/bayadera/dataset/opencl/amd-gcn.cl"))]]
  (defn gcn-dataset-engine
    ([ctx cqueue ^long WGS]
     (let [prog (build-program! (program-with-source ctx src)
                                (format "-cl-std=CL2.0 -DWGS=%s" WGS)
                                nil)]
       (->GCNDataSetEngine ctx cqueue prog WGS)))
    ([ctx queue]
     (gcn-dataset-engine ctx queue 256))))

(let [general-source (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/kernels.cl"))]
  (defn gcn-distribution-engine
    ([ctx cqueue model ^long WGS]
     (let [prog (build-program!
                 (program-with-source
                  ctx [(:functions model) general-source])
                 (format "-cl-std=CL2.0 -DLOGPDF=%s -DPDF=%s -DWGS=%s"
                         (:logpdf model) (:pdf model) WGS)
                 nil)]
       (->GCNDistributionEngine cqueue prog)))
    ([ctx queue model]
     (gcn-distribution-engine ctx queue model 256))))

(defn ^:private copy-random123 [include-name tmp-dir-name]
  (io/copy
   (io/input-stream
    (io/resource (format "uncomplicate/bayadera/rng/opencl/include/Random123/%s"
                         include-name)))
   (io/file (format "%s/Random123/%s" tmp-dir-name include-name))))

(defn gcn-direct-sampler
  ([ctx cqueue model ^long WGS]
   (let [tmp-dir-name (fsc/temp-dir "uncomplicate/")]
     (try
       (fsc/mkdirs (format "%s/%s" tmp-dir-name "Random123/features/"))
       (doseq [res-name ["philox.h" "array.h" "features/compilerfeatures.h"
                         "features/openclfeatures.h" "features/sse.h"]]
         (copy-random123 res-name tmp-dir-name))
       (->GCNDirectSampler
        cqueue
        (build-program!
         (program-with-source ctx [(:functions model) (:kernels model)])
         (format "-cl-std=CL2.0 -DWGS=%s -I%s/" WGS tmp-dir-name)
         nil))
       (finally
         (fsc/delete-dir tmp-dir-name)))))
  ([ctx cqueue model]
   (gcn-direct-sampler ctx cqueue model 256)))

;; =========================== Distribution creators ===========================

(defrecord GCNEngineFactory [dataset-eng neanderthal-factory
                             gaussian-eng gaussian-samp
                             uniform-eng uniform-samp]
  Releaseable
  (release [_]
    (and (release dataset-eng)
         (release neanderthal-factory)
         (release gaussian-eng)
         (release gaussian-samp)
         (release uniform-eng)
         (release uniform-samp)))
  DistributionEngineFactory
  (gaussian-engine [_]
    gaussian-eng)
  (uniform-engine [_]
    uniform-eng)
  SamplerFactory
  (gaussian-sampler [_]
    gaussian-samp)
  (uniform-sampler [_]
    uniform-samp)
  DataSetFactory
  (dataset-engine [_]
    dataset-eng)
  np/FactoryProvider
  (factory [_]
    neanderthal-factory))

(let [gaussian-model
      (->CLDistributionModel "gaussian_pdf" "gaussian_logpdf"
                             (str (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/uniform.h"))
                                  "\n"
                                  (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/gaussian.h")))
                             (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/gaussian.cl")))
      uniform-model
      (->CLDistributionModel "uniform_pdf"
                             "uniform_logpdf"
                             (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/uniform.h"))
                             (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/uniform.cl")))]

  (defn gcn-engine-factory
    ([ctx cqueue ^long WGS]
     (->GCNEngineFactory
      (gcn-dataset-engine ctx cqueue WGS)
      (gcn-single ctx cqueue)
      (gcn-distribution-engine ctx cqueue gaussian-model WGS)
      (gcn-direct-sampler ctx cqueue gaussian-model WGS)
      (gcn-distribution-engine ctx cqueue uniform-model WGS)
      (gcn-direct-sampler ctx cqueue uniform-model WGS)))
    ([ctx cqueue]
     (gcn-engine-factory ctx cqueue 256))))
