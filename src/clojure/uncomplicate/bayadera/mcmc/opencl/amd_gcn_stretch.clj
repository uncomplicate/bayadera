(ns uncomplicate.bayadera.mcmc.opencl.amd-gcn-stretch
  (:require [clojure.java.io :as io]
            [me.raynes.fs :as fsc]
            [uncomplicate.clojurecl
             [core :refer :all]
             [toolbox :refer [count-work-groups enq-reduce enq-read-long]]]
            [uncomplicate.neanderthal
             [math :refer [power-of-2?]]
             [core :refer [sum dim]]
             [opencl :refer [clv read!]]]
            [uncomplicate.neanderthal.opencl.amd-gcn :refer [gcn-single]]
            [uncomplicate.bayadera.protocols :refer :all]))

(defprotocol MCMCEngineFactory
  (mcmc-engine [this walker-count cl-params]))

(defn ^:private inc! [^ints a]
  (doto a (aset 0 (inc (aget a 0)))))

(deftype GCNStretch1D [ctx cqueue neanderthal-engine
                       ^long walker-count wsize ^long WGS ^ints step-counter
                       cl-params cl-xs cl-s0 cl-s1 cl-accept cl-accept-acc
                       stretch-move-odd-kernel stretch-move-even-kernel
                       init-walkers-kernel
                       sum-accept-reduction-kernel sum-accept-kernel
                       sum-means-kernel subtract-mean-kernel
                       autocovariance-kernel]
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
    (release sum-accept-reduction-kernel)
    (release sum-accept-kernel)
    (release sum-means-kernel)
    (release subtract-mean-kernel)
    (release autocovariance-kernel))
  MCMC
  (init! [this seed];;TODO =============== maybe separate init-walkers and move the rest of the init to reset-counters! and rename that to init? currently I also have a curious time < 0 when seed is 137. Maybe also some variant of init should in fact initialize cl-means to the max size of requested number of sampler runs that can be partially run?
    (if (integer? seed)
      (let [seed (int-array [seed])]
        (do
          (set-arg! init-walkers-kernel 0 seed)
          (enq-nd! cqueue init-walkers-kernel (work-size [(/ walker-count 4)]))
          (set-arg! stretch-move-odd-kernel 0 (inc! seed))
          (set-arg! stretch-move-even-kernel 0 (inc! seed))
          (reset-counters! this)
          this))
      (throw (IllegalArgumentException. "Seed must be an integer."))))
  (init! [this]
   (init! this (rand-int Integer/MAX_VALUE)))
  (reset-counters! [this]
    (do
      (enq-fill! cqueue cl-accept (int-array 1))
      (aset step-counter 0 0)
      this))
  (move! [this]
    (do
      (set-arg! stretch-move-odd-kernel 6 step-counter)
      (enq-nd! cqueue stretch-move-odd-kernel wsize)
      (set-arg! stretch-move-even-kernel 6 step-counter)
      (enq-nd! cqueue stretch-move-even-kernel wsize)
      cl-xs))
  (run-sampler! [this n]
    (let [means-count (long (count-work-groups WGS (/ walker-count 2)))]
      (with-release [means-vec (clv neanderthal-engine n)
                     cl-means (cl-buffer ctx (* Float/BYTES means-count n) :read-write)]
        (reset-counters! this)
        (enq-fill! cqueue cl-means (float-array 1))
        (set-arg! stretch-move-odd-kernel 5 cl-means)
        (set-arg! stretch-move-even-kernel 5 cl-means)
        (dotimes [i n]
          (move! this)
          (inc! step-counter))
        (set-args! sum-means-kernel 0 (.buffer means-vec)
                   cl-means (int-array [means-count]))
        (enq-nd! cqueue sum-means-kernel (work-size [n]))
        (acor this means-vec))))
  (acc-rate [_]
    (if (pos? (aget step-counter 0))
      (do
        (enq-reduce cqueue sum-accept-kernel sum-accept-reduction-kernel
                    WGS (count-work-groups WGS (/ walker-count 2)))
        (/ (double (enq-read-long cqueue cl-accept-acc))
           (* walker-count (aget step-counter 0))))
      Double/NaN))
  (acor [_ sample]
    (let [n (dim sample)
          MAXLAG 64 ;;TODO magic number
          i-max (- n MAXLAG)
          autocov-count (count-work-groups WGS i-max)]
      (if (<= (* MAXLAG 2) n)
        (with-release [c0-vec (clv neanderthal-engine autocov-count)
                       d-vec (clv neanderthal-engine autocov-count)]
          (let [sample-mean (/ (sum sample) n)]
            (set-args! subtract-mean-kernel 0 (.buffer sample)
                       (float-array [sample-mean]))
            (enq-nd! cqueue subtract-mean-kernel (work-size [n]))
            (set-args! autocovariance-kernel 0 (.buffer c0-vec)
                       (.buffer d-vec) (.buffer sample))
            (enq-nd! cqueue autocovariance-kernel (work-size [i-max]))
            (let [d (sum d-vec)]
              (->Autocorrelation (/ d (sum c0-vec)) sample-mean (/ d n i-max)
                                 (* n walker-count) d))))
        (throw (IllegalArgumentException.
                (format "Number of steps (%d) must not be less than %d." n (* 2 MAXLAG))))))))

(deftype GCNStretch1DEngineFactory [ctx queue neanderthal-engine
                                    prog ^long WGS]
  Releaseable
  (release [_]
    (release prog)
    (release neanderthal-engine))
  MCMCEngineFactory
  (mcmc-engine [_ walker-count params]
    (if (and (<= (* 2 WGS) walker-count) (power-of-2? walker-count))
      (let [walker-count (long walker-count)
            cnt (long (/ walker-count 2))
            accept-count (count-work-groups WGS cnt)
            accept-acc-count (count-work-groups WGS accept-count)
            bytecount (long (* Float/BYTES cnt))
            step-counter (int-array 1)
            cl-params (let [par-buf (cl-buffer ctx
                                               (* Float/BYTES
                                                  (alength ^floats params))
                                               :read-only)]
                        (enq-write! queue par-buf params)
                        par-buf)
            cl-xs (cl-buffer ctx (* 2 bytecount) :read-write)
            cl-s0 (cl-sub-buffer cl-xs 0 bytecount :read-write)
            cl-s1 (cl-sub-buffer cl-xs bytecount bytecount :read-write)
            cl-accept (cl-buffer ctx (* Integer/BYTES accept-count) :read-write)
            cl-accept-acc (cl-buffer ctx (* Long/BYTES accept-acc-count) :read-write)]
        (->GCNStretch1D
         ctx queue neanderthal-engine
         walker-count (work-size [(/ walker-count 2)]) WGS step-counter
         cl-params cl-xs cl-s0 cl-s1
         cl-accept cl-accept-acc
         (doto (kernel prog "stretch_move1")
           (set-args! 0 (rand-ints) cl-params cl-s0 cl-s1 cl-accept))
         (doto (kernel prog "stretch_move1")
           (set-args! 0 (rand-ints) cl-params cl-s1 cl-s0 cl-accept))
         (doto (kernel prog "init_walkers") (set-args! (rand-ints) cl-xs))
         (doto (kernel prog "sum_accept_reduction") (set-arg! 0 cl-accept-acc))
         (doto (kernel prog "sum_accept_reduce") (set-args! 0 cl-accept-acc cl-accept))
         (kernel prog "sum_means")
         (kernel prog "subtract_mean")
         (kernel prog "autocovariance")))
      (throw (IllegalArgumentException.
              (format "Number of walkers (%d) must be a power of 2, not less than %d."
                      walker-count (* 2 WGS)))))))

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
      (let [neanderthal-engine (gcn-single ctx cqueue)]
        (->GCNStretch1DEngineFactory
         ctx cqueue neanderthal-engine
         (build-program!
          (program-with-source
           ctx
           [(slurp (io/resource "uncomplicate/clojurecl/kernels/reduction.cl"))
            (slurp (io/resource "uncomplicate/bayadera/mcmc/opencl/kernels/amd_gcn/random.h"))
            (slurp (io/resource "uncomplicate/bayadera/mcmc/opencl/kernels/amd_gcn/stretch-move.cl"))])
          (format "-cl-std=CL2.0 -DACCUMULATOR=float -I%s/" tmp-dir-name)
          nil)
         256))
      (finally
        (fsc/delete-dir tmp-dir-name)))))
