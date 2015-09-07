(ns uncomplicate.bayadera.mcmc.opencl.amd-gcn-stretch
  (:require [clojure.java.io :as io]
            [me.raynes.fs :as fsc]
            [uncomplicate.clojurecl
             [core :refer :all]
             [toolbox :refer [count-work-groups enq-reduce enq-read-long]]]
            [uncomplicate.neanderthal
             [math :refer [sqrt]]
             [core :refer [asum sum dim vect?]]
             [opencl :refer [clv read!]]]
            [uncomplicate.neanderthal.opencl.amd-gcn :refer [gcn-single]]
            [uncomplicate.bayadera.protocols :refer :all]))

(defprotocol MCMCEngineFactory
  (mcmc-engine [this walker-count cl-params]))

(defn ^:private inc! [^ints a]
  (doto a (aset 0 (inc (aget a 0)))))

(deftype GCNStretch1D [ctx cqueue neanderthal-engine
                       ^long walker-count wsize ^long WGS
                       ^ints step-counter
                       cl-params cl-xs cl-s0 cl-s1 cl-accept cl-accept-acc
                       stretch-move-odd-kernel stretch-move-even-kernel
                       stretch-move-odd-bare-kernel stretch-move-even-bare-kernel
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
    (release stretch-move-odd-bare-kernel)
    (release stretch-move-even-bare-kernel)
    (release init-walkers-kernel)
    (release sum-accept-reduction-kernel)
    (release sum-accept-kernel)
    (release sum-means-kernel)
    (release subtract-mean-kernel)
    (release autocovariance-kernel))
  MCMC
  (init-walkers! [this seed cl-walkers]
    (do
      (enq-copy! cqueue cl-walkers cl-xs)))
  (init-walkers! [this seed]
    (do
     (set-arg! init-walkers-kernel 0 seed)
     (enq-nd! cqueue init-walkers-kernel (work-size [(/ walker-count 4)]))))
  (init! [this seed]
    (do
      (set-arg! stretch-move-odd-kernel 0 (inc! seed))
      (set-arg! stretch-move-even-kernel 0 (inc! seed))
      (set-arg! stretch-move-odd-bare-kernel 0 (inc! seed))
      (set-arg! stretch-move-even-bare-kernel 0 (inc! seed))
      (enq-fill! cqueue cl-accept (int-array 1))
      (aset step-counter 0 0)
      this))
  (init! [this]
    (init! this (int-array [(rand-int Integer/MAX_VALUE)])))
  (move! [this]
    (do
      (set-arg! stretch-move-odd-kernel 6 step-counter)
      (set-arg! stretch-move-even-kernel 6 step-counter)
      (enq-nd! cqueue stretch-move-odd-kernel wsize)
      (enq-nd! cqueue stretch-move-even-kernel wsize)
      (inc! step-counter)
      cl-xs))
  (move-bare! [this]
    (do
      (set-arg! stretch-move-odd-bare-kernel 4 step-counter)
      (set-arg! stretch-move-even-bare-kernel 4 step-counter)
      (enq-nd! cqueue stretch-move-odd-bare-kernel wsize)
      (enq-nd! cqueue stretch-move-even-bare-kernel wsize)
      (inc! step-counter)
      cl-xs))
  (burn-in! [this n a]
    (do
      (aset step-counter 0 0)
      (set-arg! stretch-move-odd-bare-kernel 5 a)
      (set-arg! stretch-move-even-bare-kernel 5 a)
      (dotimes [i (dec n)]
        (move-bare! this))
      (let [means-count (long (count-work-groups WGS (/ walker-count 2)))]
        (with-release [cl-means (cl-buffer ctx (* Float/BYTES) :read-write)]
          (aset step-counter 0 0)
          (enq-fill! cqueue cl-accept (int-array 1))
          (set-arg! stretch-move-odd-kernel 5 cl-means)
          (set-arg! stretch-move-even-kernel 5 cl-means)
          (set-arg! stretch-move-odd-kernel 7 a)
          (set-arg! stretch-move-even-kernel 7 a)
          (move! this)
          (acc-rate this)))))
  (run-sampler! [this n a]
    (let [means-count (long (count-work-groups WGS (/ walker-count 2)))]
      (with-release [means-vec (clv neanderthal-engine n)
                     cl-means (cl-buffer ctx (* Float/BYTES means-count (long n))
                                         :read-write)]
        (aset step-counter 0 0)
        (enq-fill! cqueue cl-means (float-array 1))
        (set-arg! stretch-move-odd-kernel 5 cl-means)
        (set-arg! stretch-move-even-kernel 5 cl-means)
        (set-arg! stretch-move-odd-kernel 7 a)
        (set-arg! stretch-move-even-kernel 7 a)

        (dotimes [i n]
          (move! this))
        (set-args! sum-means-kernel 0 (.buffer means-vec)
                   cl-means (int-array [means-count]))
        (enq-nd! cqueue sum-means-kernel (work-size [n]))
        (assoc (acor this means-vec) :acc-rate  (acc-rate this)))))
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
          min-fac 16 ;; TODO magic number
          MAXLAG 64 ;;TODO magic number
          MINLAG 4
          lag (max MINLAG (min (quot n min-fac) MAXLAG))
          i-max (- n lag)
          autocov-count (count-work-groups WGS i-max)]
      (if (<= (* lag min-fac) n)
        (with-release [c0-vec (clv neanderthal-engine autocov-count)
                       d-vec (clv neanderthal-engine autocov-count)]
          (let [sample-mean (/ (float (sum sample)) n)]
            (set-args! subtract-mean-kernel 0 (.buffer sample)
                       (float-array [sample-mean]))
            (enq-nd! cqueue subtract-mean-kernel (work-size [n]))
            (set-args! autocovariance-kernel 0 (int-array [lag]) (.buffer c0-vec)
                       (.buffer d-vec) (.buffer sample))
            (enq-nd! cqueue autocovariance-kernel (work-size [i-max]))
            (let [d (float (sum d-vec))]
              (->Autocorrelation (/ d (float (sum c0-vec))) sample-mean
                                 (sqrt (/ d i-max n))
                                 (* n walker-count) n walker-count lag 0.0))))
        (throw (IllegalArgumentException.
                (format (str "The autocorrelation time is too long relative to the variance."
                             "Number of steps (%d) must not be less than %d.")
                        n (* lag min-fac))))))))

(deftype GCNStretch1DEngineFactory [ctx queue neanderthal-engine prog ^long WGS]
  Releaseable
  (release [_]
    (release prog)
    (release neanderthal-engine))
  MCMCEngineFactory
  (mcmc-engine [_ walker-count params]
    (let [walker-count (long walker-count)]
      (if (and (<= (* 2 WGS) walker-count) (zero? (rem walker-count (* 2 WGS))))
        (let [cnt (long (/ walker-count 2))
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
           (doto (kernel prog "stretch_move1_accu")
             (set-args! 1 cl-params cl-s1 cl-s0 cl-accept))
           (doto (kernel prog "stretch_move1_accu")
             (set-args! 1 cl-params cl-s0 cl-s1 cl-accept))
           (doto (kernel prog "stretch_move1_bare")
             (set-args! 1 cl-params cl-s1 cl-s0))
           (doto (kernel prog "stretch_move1_bare")
             (set-args! 1 cl-params cl-s0 cl-s1))
           (doto (kernel prog "init_walkers") (set-arg! 1 cl-xs))
           (doto (kernel prog "sum_accept_reduction") (set-arg! 0 cl-accept-acc))
           (doto (kernel prog "sum_accept_reduce") (set-args! 0 cl-accept-acc cl-accept))
           (kernel prog "sum_means")
           (kernel prog "subtract_mean")
           (kernel prog "autocovariance")))
        (throw (IllegalArgumentException.
                (format "Number of walkers (%d) must be a multiple of %d."
                        walker-count (* 2 WGS))))))))

(defn ^:private copy-random123 [include-name tmp-dir-name]
  (io/copy
   (io/input-stream
    (io/resource (format "uncomplicate/bayadera/mcmc/opencl/include/Random123/%s"
                         include-name)))
   (io/file (format "%s/Random123/%s" tmp-dir-name include-name))))

(defn gcn-stretch-1d-engine-factory [ctx cqueue a]
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
          (format "-cl-std=CL2.0 -DA=%f  -DACCUMULATOR=float -I%s/" a tmp-dir-name)
          nil)
         256))
      (finally
        (fsc/delete-dir tmp-dir-name)))))
