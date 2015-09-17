(ns uncomplicate.bayadera.distributions.opencl.amd-gcn
  (:require [clojure.java.io :as io]
            [me.raynes.fs :as fsc]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal.core :refer [dim]]
            [uncomplicate.neanderthal.opencl.amd-gcn :refer [gcn-single]]
            [uncomplicate.bayadera.protocols :refer :all])
  (:import [uncomplicate.neanderthal.protocols Block]))

(defprotocol CLAware
  (get-context [_])
  (get-queue [_]))

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

(deftype GCNEngineFactory [ctx cqueue neanderthal-factory prog]
  Releaseable
  (release [_]
    (release prog)
    (release neanderthal-factory))
  CLAware
  (get-context [_]
    ctx)
  (get-queue [_]
    cqueue)
  DistributionEngineFactory
  (vector-factory [_]
    neanderthal-factory)
  (random-sampler [_ dist-name]
    (->GCNDirectSampler cqueue (kernel prog (str dist-name "_sample"))))
  (dist-engine [_ dist-name]
    (->GCNDistributionEngine cqueue
                             (kernel prog (str dist-name "_logpdf_kernel"))
                             (kernel prog (str dist-name "_pdf_kernel")))))

(defn ^:private copy-random123 [include-name tmp-dir-name]
  (io/copy
   (io/input-stream
    (io/resource (format "uncomplicate/bayadera/rng/opencl/include/Random123/%s"
                         include-name)))
   (io/file (format "%s/Random123/%s" tmp-dir-name include-name))))

(defn gcn-distribution-engine-factory [ctx cqueue]
  (let [tmp-dir-name (fsc/temp-dir "uncomplicate/")]
    (try
      (fsc/mkdirs (format "%s/%s" tmp-dir-name "Random123/features/"))
      (doseq [res-name ["philox.h" "array.h" "features/compilerfeatures.h"
                        "features/openclfeatures.h" "features/sse.h"]]
        (copy-random123 res-name tmp-dir-name))
      (let [neanderthal-factory (gcn-single ctx cqueue)]
        (->GCNEngineFactory
         ctx cqueue neanderthal-factory
         (build-program!
          (program-with-source
           ctx
           [(slurp (io/resource "uncomplicate/bayadera/distributions/opencl/sampling.h"))
            (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/measures.h"))
            (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/kernels.cl"))])
          (format "-cl-std=CL2.0 -I%s/" tmp-dir-name)
          nil)))
      (finally
        (fsc/delete-dir tmp-dir-name)))))
