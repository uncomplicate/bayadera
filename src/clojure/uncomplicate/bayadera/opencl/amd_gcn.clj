(ns uncomplicate.bayadera.opencl.amd-gcn
  (:require [clojure.java.io :as io]
            [me.raynes.fs :as fsc]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.clojurecl.toolbox
             :refer [enq-reduce enq-read-double count-work-groups]]
            [uncomplicate.neanderthal
             [core :refer [dim sum nrm2]]
             [protocols :as np]
             [block :refer [buffer]]
             [native :refer [sv]]
             [opencl :refer [clv clge gcn-single]]]
            [uncomplicate.bayadera.protocols :refer :all])
  (:import [uncomplicate.neanderthal.opencl.amd_gcn GCNVectorEngine]))

(deftype GCNDirectSampler [cqueue sample-kernel]
  Releaseable
  (release [_]
    (release sample-kernel))
  RandomSampler
  (sample! [this seed res]
    (do
      (set-args! sample-kernel 1 (int-array [seed]) (buffer res))
      (enq-nd! cqueue sample-kernel (work-size [(dim res)]))
      this)))

(deftype GCNDistributionEngine [cqueue logpdf-kernel pdf-kernel]
  Releaseable
  (release [_]
    (release logpdf-kernel)
    (release pdf-kernel))
  DistributionEngine
  (logpdf! [this x res]
    (do
      (set-args! logpdf-kernel 1 (buffer x) (buffer res))
      (enq-nd! cqueue logpdf-kernel (work-size [(dim x)]))
      this))
  (pdf! [this x res]
    (do
      (set-args! pdf-kernel 1 (buffer x) (buffer res))
      (enq-nd! cqueue pdf-kernel (work-size [(dim x)]))
      this)))

(defrecord GCNDataSetEngine [cqueue data-vect variance-kernel]
  Releaseable
  (release [_]
    (release variance-kernel))
  Spread
  (mean-variance [this]
    (let [neand-eng ^GCNVectorEngine (np/engine data-vect)
          m (/ (sum data-vect) (dim data-vect))]
      (set-arg! variance-kernel 2 (float-array [m]))
      (enq-reduce cqueue variance-kernel (.sum-reduction-kernel neand-eng)
                  (.WGS neand-eng) (dim data-vect))
      (sv m (/ (enq-read-double cqueue (.reduce-acc neand-eng)) (dec (dim data-vect)))))))

(deftype GCNEngineFactory [ctx cqueue neand-factory prog]
  Releaseable
  (release [_]
    (release prog)
    (release neand-factory))
  np/BlockCreator
  (create-block [_ n]
    (clv neand-factory n))
  (create-block [_ m n]
    (clge neand-factory m n))
  EngineFactory
  (dataset-engine [_ data-vect]
    (let [neand-eng ^GCNVectorEngine (np/engine data-vect)]
      (->GCNDataSetEngine
       cqueue data-vect
       (doto (kernel prog "variance_reduce")
         (set-args! 0 (.reduce-acc neand-eng) (buffer data-vect))))))
  (random-sampler [_ dist-name params]
    (let [cl-params (cl-buffer ctx (* Float/BYTES 2) :read-only)]
      (enq-write! cqueue cl-params (buffer params))
      (->GCNDirectSampler cqueue (doto (kernel prog (str dist-name "_sample"))
                                   (set-arg! 0 cl-params)))))
  (distribution-engine [_ dist-name params]
    (let [cl-params (cl-buffer ctx (* Float/BYTES 2) :read-only)]
      (enq-write! cqueue cl-params (buffer params))
      (->GCNDistributionEngine
       cqueue
       (doto (kernel prog (str dist-name "_logpdf_kernel"))
         (set-arg! 0 cl-params))
       (doto (kernel prog (str dist-name "_pdf_kernel"))
         (set-arg! 0 cl-params))))))

(defn ^:private copy-random123 [include-name tmp-dir-name]
  (io/copy
   (io/input-stream
    (io/resource (format "uncomplicate/bayadera/rng/opencl/include/Random123/%s"
                         include-name)))
   (io/file (format "%s/Random123/%s" tmp-dir-name include-name))))

(defn gcn-engine-factory [ctx cqueue]
  (let [tmp-dir-name (fsc/temp-dir "uncomplicate/")]
    (try
      (fsc/mkdirs (format "%s/%s" tmp-dir-name "Random123/features/"))
      (doseq [res-name ["philox.h" "array.h" "features/compilerfeatures.h"
                        "features/openclfeatures.h" "features/sse.h"]]
        (copy-random123 res-name tmp-dir-name))
      (let [neand-factory (gcn-single ctx cqueue)]
             (->GCNEngineFactory
              ctx cqueue neand-factory
              (build-program!
               (program-with-source
                ctx
                [(slurp (io/resource "uncomplicate/clojurecl/kernels/reduction.cl"))
                 (slurp (io/resource "uncomplicate/bayadera/dataset/opencl/amd-gcn.cl"))
                 (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/sampling.h"))
                 (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/measures.h"))
                 (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/kernels.cl"))])
               (format "-cl-std=CL2.0 -I%s/" tmp-dir-name)
               nil)))
           (finally
             (fsc/delete-dir tmp-dir-name)))))
