(ns uncomplicate.bayadera.opencl.amd-gcn
  (:require [clojure.java.io :as io]
            [me.raynes.fs :as fsc]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.clojurecl.toolbox
             :refer [enq-reduce enq-read-float count-work-groups]]
            [uncomplicate.neanderthal
             [core :refer [dim sum nrm2]]
             [protocols :as np]
             [opencl :refer [clv clge gcn-single]]]
            [uncomplicate.bayadera.protocols :refer :all])
  (:import [uncomplicate.neanderthal.opencl.amd_gcn GCNVectorEngine]))

(deftype GCNDirectSampler [cqueue sample-kernel]
  Releaseable
  (release [_]
    (release sample-kernel))
  RandomSampler
  (sample! [this seed params res]
    (do
      (set-args! sample-kernel (int-array [seed]) (.buffer params) (.buffer res))
      (enq-nd! cqueue sample-kernel (work-size [(dim res)]))
      this)))

(deftype GCNDistributionEngine [cqueue logpdf-kernel pdf-kernel]
  Releaseable
  (release [_]
    (release logpdf-kernel)
    (release pdf-kernel))
  DistributionEngine
  (logpdf! [this params x res]
    (do
      (set-args! logpdf-kernel (.buffer params) (.buffer x) (.buffer res))
      (enq-nd! cqueue logpdf-kernel (work-size [(dim x)]))
      this))
  (pdf! [this params x res]
    (do
      (set-args! pdf-kernel (.buffer params) (.buffer x) (.buffer res))
      (enq-nd! cqueue pdf-kernel (work-size [(dim x)]))
      this)))

(defn enq-reduce-weighted
  [queue main-kernel reduce-kernel max-local-size n]
  (loop [queue (enq-nd! queue main-kernel (work-size [n]))
         global-size (count-work-groups max-local-size n)
         tail-rem (rem n max-local-size)]
    (if (= 1 global-size)
      queue
      (recur
       (do (set-arg! reduce-kernel 2 (int-array [tail-rem]))
           (enq-nd! queue reduce-kernel (work-size [global-size])))
       (count-work-groups max-local-size global-size)
       (rem global-size max-local-size)))))

(defrecord GCNDataSetEngine [cqueue
                             reduction-acc
                             mean-variance-kernel mean-variance-reduction-kernel]
  Releaseable
  (release [_]
    (release reduction-acc)
    (release mean-variance-kernel)
    (release mean-variance-reduction-kernel))
  Spread
  (variance [_ dataset]
    (let [data-vect (data dataset)
          neand-eng ^GCNVectorEngine (np/engine data-vect)
          WGS (.WGS neand-eng)]
      (enq-reduce-weighted cqueue mean-variance-kernel
                           mean-variance-reduction-kernel WGS (dim data-vect))

      [(enq-read-float cqueue (.reduce-acc neand-eng))
       (/ (enq-read-float cqueue reduction-acc) (dim data-vect))])))

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
    (let [neand-eng ^GCNVectorEngine (np/engine data-vect)
          acc-size (* Double/BYTES (count-work-groups (.WGS neand-eng) (dim data-vect)))
          cl-acc (cl-buffer ctx acc-size :read-write)]
      (->GCNDataSetEngine
       cqueue cl-acc
       (doto (kernel prog "mean_variance_reduce")
         (set-args! 0 (.reduce-acc neand-eng)
                    cl-acc (.buffer data-vect)))
       (doto (kernel prog "mean_variance_reduction")
         (set-args! 0 (.reduce-acc neand-eng) cl-acc)))))
  (random-sampler [_ dist-name]
    (->GCNDirectSampler cqueue (kernel prog (str dist-name "_sample"))))
  (distribution-engine [_ dist-name]
    (->GCNDistributionEngine cqueue
                             (kernel prog (str dist-name "_logpdf_kernel"))
                             (kernel prog (str dist-name "_pdf_kernel")))))

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
               (format "-cl-std=CL2.0 -DACCUMULATOR=float -I%s/" tmp-dir-name)
               nil)))
           (finally
             (fsc/delete-dir tmp-dir-name)))))
