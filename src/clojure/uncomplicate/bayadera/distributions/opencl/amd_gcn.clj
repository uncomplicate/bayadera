(ns uncomplicate.bayadera.distributions.opencl.amd-gcn
  (:require [clojure.java.io :as io]
            [me.raynes.fs :as fsc]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.bayadera.protocols :refer :all]))

(defprotocol DistributionEngineFactory
  (random-sampler [_])
  (distribution-engine [_]))

(deftype GCNDistributionEngine [cqueue logpdf-kernel pdf-kernel]
  Releaseable
  (release [_]
    (release logpdf-kernel)
    (release pdf-kernel))
  DistributionEngine
  (logpdf! [this n cl-params cl-x cl-res]
    (do
      (set-args! logpdf-kernel cl-params cl-x cl-res)
      (enq-nd! cqueue logpdf-kernel (work-size [n]))
      this))
  (pdf! [this n cl-params cl-x cl-res]
    (do
      (set-args! pdf-kernel cl-params cl-x cl-res)
      (enq-nd! cqueue pdf-kernel (work-size [n]))
      this)))

(deftype GCNDirectSampler [cqueue sample-kernel]
  Releaseable
  (release [_]
    (release sample-kernel))
  RandomSampler
  (sample! [this seed n cl-params cl-res]
    (do
      (set-args! sample-kernel seed cl-params cl-res)
      (enq-nd! cqueue sample-kernel (work-size [n]))
      this)))

(deftype GCNEngineFactory [cqueue prog name]
  Releaseable
  (release [_]
    (release prog))
  DistributionEngineFactory
  (random-sampler [_]
    (->GCNDirectSampler cqueue (kernel prog (str name "_sample"))))
  (distribution-engine [_]
    (->GCNDistributionEngine cqueue
                             (kernel prog "logpdf_kernel")
                             (kernel prog "pdf_kernel"))))

(defn ^:private copy-random123 [include-name tmp-dir-name]
  (io/copy
   (io/input-stream
    (io/resource (format "uncomplicate/bayadera/rng/opencl/include/Random123/%s"
                         include-name)))
   (io/file (format "%s/Random123/%s" tmp-dir-name include-name))))

(defn gcn-distribution-engine-factory [ctx cqueue dist-name]
  (let [tmp-dir-name (fsc/temp-dir "uncomplicate/")]
    (try
      (fsc/mkdirs (format "%s/%s" tmp-dir-name "Random123/features/"))
      (doseq [res-name ["philox.h" "array.h" "features/compilerfeatures.h"
                        "features/openclfeatures.h" "features/sse.h"]]
        (copy-random123 res-name tmp-dir-name))
      (->GCNEngineFactory
       cqueue
       (build-program!
        (program-with-source
         ctx
         [(slurp (io/resource "uncomplicate/bayadera/distributions/opencl/sampling.h"))
          (slurp (io/resource (format "uncomplicate/bayadera/distributions/opencl/%s.h" dist-name)))
          (slurp (io/resource (format "uncomplicate/bayadera/distributions/opencl/%s.cl" dist-name)))
          (slurp (io/resource "uncomplicate/bayadera/distributions/opencl/kernels.cl"))])
        (format "-cl-std=CL2.0 -DDIST_LOGPDF=%s_logpdf -DDIST_PDF=%s_pdf -I%s/"
                dist-name dist-name tmp-dir-name)
        nil)
       dist-name)
      (finally
        (fsc/delete-dir tmp-dir-name)))))
