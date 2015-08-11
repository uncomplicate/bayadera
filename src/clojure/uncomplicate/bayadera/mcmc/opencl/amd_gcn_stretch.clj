(ns uncomplicate.bayadera.mcmc.opencl.amd-gcn-stretch
  (:require [clojure.java.io :as io]
            [me.raynes.fs :as fsc]
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [info durations profiling-info]]]))

(defn ^:private count-work-groups ^long [^long max-local-size ^long n]
  (if (< max-local-size n)
    (quot (+ n (dec max-local-size)) max-local-size)
    1))

(defn ^:private enq-reduce
  [queue main-kernel reduce-kernel max-local-size n]
  (loop [queue (enq-nd! queue main-kernel (work-size [n]))
         global-size (count-work-groups max-local-size n)]
    (if (= 1 global-size)
      queue
      (recur
       (enq-nd! queue reduce-kernel (work-size [global-size]))
       (count-work-groups max-local-size global-size)))))

(defn ^:private enq-read-long ^long [queue cl-buf]
  (let [res (long-array 1)]
    (enq-read! queue cl-buf res)
    (aget res 0)))

(defprotocol MCMC
  (init! [_] [_ cl-buff])
  (move! [this])
  (burn-in! [this n])
  (acc-rate [this]))

(defprotocol MCMCEngineFactory
  (mcmc-engine [this walker-count cl-params]))

(deftype GCNStretch1D [cqueue ^long walker-count wsize ^long WGS step-counter
                       ^ints seed cl-params cl-xs cl-s0 cl-s1 cl-accept cl-accept-acc
                       stretch-move-odd-kernel stretch-move-even-kernel
                       init-walkers-kernel sum-reduction-kernel sum-accept-kernel]
  Releaseable
  (release [_]
    (release cl-params)
    (release cl-xs)
    (release cl-s0)
    (release cl-s1)
    (release cl-accept)
    (release cl-accept-acc)
    (release stretch-move-odd-kernel)
    (release stretch-move-even-kernel)
    (release init-walkers-kernel)
    (release sum-reduction-kernel)
    (release sum-accept-kernel))
  MCMC
  (init! [this]
    (do
      (enq-fill! cqueue cl-accept (int-array 1))
      (set-arg! init-walkers-kernel 0
                (doto seed (aset 0 (long (rand-int Integer/MAX_VALUE)))))
      (enq-nd! cqueue init-walkers-kernel (work-size [(/ walker-count 4)]))
      (compare-and-set! step-counter @step-counter 0)
      this))
  (move! [this]
    (do
      (set-arg! stretch-move-odd-kernel 0
                (doto seed (aset 0 (long (rand-int Integer/MAX_VALUE)))))
      (enq-nd! cqueue stretch-move-odd-kernel wsize)
      (set-arg! stretch-move-even-kernel 0
                (doto seed (aset 0 (long (rand-int Integer/MAX_VALUE)))))
      (enq-nd! cqueue stretch-move-even-kernel wsize)
      (swap! step-counter inc)
      cl-xs))
  (burn-in! [this n]
    (do
      (dotimes [_ n] (move! this))
      this))
  (acc-rate [_]
    (if (pos? @step-counter)
      (do
        (enq-reduce cqueue sum-accept-kernel sum-reduction-kernel
                    WGS (count-work-groups WGS (/ walker-count 2)))
        (double (/ (enq-read-long cqueue cl-accept-acc) (* walker-count (long @step-counter)))))
      Double/NaN)))

(deftype GCNStretch1DEngineFactory [ctx queue prog ^long WGS]
  Releaseable
  (release [_]
    (release prog))
  MCMCEngineFactory
  (mcmc-engine [_ walker-count params]
    (let [walker-count (long walker-count)
          cnt (/ walker-count 2)
          accept-count (count-work-groups WGS cnt)
          accept-acc-count (count-work-groups WGS accept-count)
          bytecount (long (* Float/BYTES cnt))
          seed (int-array 1)
          cl-params (let [par-buf (cl-buffer ctx (* Float/BYTES (alength ^floats  params)) :read-only)]
                      (enq-write! queue par-buf params)
                      par-buf)
          cl-xs (cl-buffer ctx (* 2 bytecount) :read-write)
          cl-s0 (cl-sub-buffer cl-xs 0 bytecount :read-write)
          cl-s1 (cl-sub-buffer cl-xs bytecount bytecount :read-write)
          cl-accept (cl-buffer ctx (* Integer/BYTES accept-count) :read-write)
          cl-accept-acc (cl-buffer ctx (* Long/BYTES accept-acc-count) :read-write)]
      (->GCNStretch1D
       queue walker-count (work-size [(/ walker-count 2)]) WGS (atom 0)
       seed cl-params cl-xs cl-s0 cl-s1 cl-accept cl-accept-acc
       (doto (kernel prog "stretch_move1")
         (set-args! 1 cl-params cl-s0 cl-s1 cl-accept))
       (doto (kernel prog "stretch_move1")
         (set-args! 1 cl-params cl-s1 cl-s0 cl-accept))
       (doto (kernel prog "init_walkers") (set-arg! 1 cl-xs))
       (doto (kernel prog "sum_reduction") (set-arg! 0 cl-accept-acc))
       (doto (kernel prog "sum_accept_reduce") (set-args! 0 cl-accept-acc cl-accept))))))

(defn ^:private copy-random123 [include-name tmp-dir-name]
  (io/copy
   (io/input-stream
    (io/resource (format "uncomplicate/bayadera/mcmc/opencl/include/Random123/%s"
                         include-name)))
   (io/file (format "%s/Random123/%s" tmp-dir-name include-name))))

(defn gcn-stretch-1d-engine-factory [ctx cqueue]
  (let [tmp-dir-name (fsc/temp-dir "uncomplicate/")]
    (try
      (fsc/mkdirs (format "%s/%s" tmp-dir-name "Random123/features/"))
      (doseq [res-name ["philox.h" "array.h" "features/compilerfeatures.h"
                        "features/openclfeatures.h" "features/sse.h"]]
        (copy-random123 res-name tmp-dir-name))
      (->GCNStretch1DEngineFactory
       ctx cqueue
       (build-program!
        (program-with-source
         ctx
         [(slurp (io/resource "uncomplicate/bayadera/mcmc/opencl/kernels/amd_gcn/random.h"))
          (slurp (io/resource "uncomplicate/bayadera/mcmc/opencl/kernels/amd_gcn/stretch-move.cl"))])
        (format "-cl-std=CL2.0 -I%s/" tmp-dir-name)
        nil)
       256)
      (finally
        (fsc/delete-dir tmp-dir-name)))))
