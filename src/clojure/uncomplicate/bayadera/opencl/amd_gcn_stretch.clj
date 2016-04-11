(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.opencl.amd-gcn-stretch
  (:require [clojure.java.io :as io]
            [uncomplicate.commons.core
             :refer [Releaseable release wrap-float wrap-int with-release double-fn]]
            [uncomplicate.fluokitten.core :refer [op fmap fmap!]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [toolbox :refer [count-work-groups enq-reduce enq-read-long]]]
            [uncomplicate.neanderthal
             [math :refer [sqrt]]
             [core :refer [dim create-raw create scal! copy matrix? ncols vect?]]
             [native :refer [sv]]
             [real :refer [sum nrm2]]
             [block :refer [buffer]]
             [opencl :refer [gcn-single]]]
            [uncomplicate.bayadera.protocols :refer :all]
            [uncomplicate.bayadera.opencl
             [utils :refer [with-philox get-tmp-dir-name]]]))

(defn ^:private inc! [^ints a]
  (doto a (aset 0 (inc (aget a 0)))))

(deftype GCNStretch [ctx cqueue neanderthal-factory
                     ^long walker-count wsize ^long DIM ^long WGS
                     ^ints step-counter
                     cl-params cl-xs cl-s0 cl-s1
                     cl-logpdf-xs cl-logpdf-s0 cl-logpdf-s1
                     cl-accept cl-accept-acc
                     stretch-move-odd-kernel stretch-move-even-kernel
                     stretch-move-odd-bare-kernel stretch-move-even-bare-kernel
                     init-walkers-kernel logpdf-kernel
                     sum-accept-reduction-kernel sum-accept-kernel
                     sum-means-kernel
                     sum-reduction-kernel sum-reduce-kernel
                     scal-kernel
                     subtract-mean-kernel
                     autocovariance-kernel]
  Releaseable
  (release [_]
    (and
     (release cl-xs)
     (release cl-s0)
     (release cl-s1)
     (release cl-logpdf-xs)
     (release cl-logpdf-s0)
     (release cl-logpdf-s1)
     (release cl-accept)
     (release cl-accept-acc)
     (release stretch-move-odd-kernel)
     (release stretch-move-even-kernel)
     (release stretch-move-odd-bare-kernel)
     (release stretch-move-even-bare-kernel)
     (release init-walkers-kernel)
     (release logpdf-kernel)
     (release sum-accept-reduction-kernel)
     (release sum-accept-kernel)
     (release sum-means-kernel)
     (release sum-reduction-kernel)
     (release sum-reduce-kernel)
     (release scal-kernel)
     (release subtract-mean-kernel)
     (release autocovariance-kernel)))
  MCMCStretch
  (move! [this]
    (do
      (set-arg! stretch-move-odd-kernel 8 step-counter)
      (set-arg! stretch-move-even-kernel 8 step-counter)
      (enq-nd! cqueue stretch-move-odd-kernel wsize)
      (enq-nd! cqueue stretch-move-even-kernel wsize)
      (inc! step-counter)
      cl-xs))
  (move-bare! [this]
    (do
      (set-arg! stretch-move-odd-bare-kernel 6 step-counter)
      (set-arg! stretch-move-even-bare-kernel 6 step-counter)
      (enq-nd! cqueue stretch-move-odd-bare-kernel wsize)
      (enq-nd! cqueue stretch-move-even-bare-kernel wsize)
      (inc! step-counter)
      cl-xs))
  (acc-rate [_]
    (if (pos? (aget step-counter 0))
      (do
        (enq-reduce cqueue sum-accept-kernel sum-accept-reduction-kernel
                    (count-work-groups WGS (/ walker-count 2)) WGS)
        (/ (double (enq-read-long cqueue cl-accept-acc))
           (* walker-count (aget step-counter 0))))
      Double/NaN))
  (acor [_ cl-sample]
    (let [n (/ (size cl-sample) (* Float/BYTES DIM))
          min-fac 16
          MINLAG 4
          WINMULT 16
          TAUMAX 16
          lag (max MINLAG (min (quot n min-fac) WGS))
          i-max (- n lag)
          c0 (sv DIM)
          d (sv DIM)
          mean-vec (sv DIM)
          wgsm (min DIM WGS)
          wgsn (long (/ WGS wgsm))
          wg-count (count-work-groups wgsn n)]
      (if (<= (* lag min-fac) n)
        (with-release [cl-acc (cl-buffer ctx (* DIM wg-count Float/BYTES) :read-write)
                       cl-sub-acc (cl-sub-buffer cl-acc 0 (* DIM Float/BYTES))
                       d-acc (cl-buffer ctx (* DIM wg-count Float/BYTES) :read-write)]
          (set-arg! sum-reduction-kernel 0 cl-acc)
          (set-args! sum-reduce-kernel 0 cl-acc cl-sample)
          (enq-reduce cqueue sum-reduce-kernel sum-reduction-kernel
                      DIM n wgsm wgsn)
          (set-args! scal-kernel 0 (wrap-float (/ n)) cl-sub-acc)
          (enq-nd! cqueue scal-kernel (work-size-1d (* DIM (long n))))
          (enq-read! cqueue cl-sub-acc (buffer mean-vec))
          (set-args! subtract-mean-kernel 0 cl-sample cl-sub-acc)
          (enq-nd! cqueue subtract-mean-kernel (work-size-2d DIM n))
          (enq-fill! cqueue cl-acc (int-array 1))
          (enq-fill! cqueue d-acc (int-array 1))
          (set-args! autocovariance-kernel 0 (wrap-int lag) cl-acc
                     d-acc cl-sample (wrap-int i-max))
          (enq-nd! cqueue autocovariance-kernel (work-size-1d n))
          (set-arg! sum-reduction-kernel 0 cl-acc)
          (set-args! sum-reduce-kernel 0 cl-acc cl-acc)
          (enq-reduce cqueue sum-reduce-kernel sum-reduction-kernel
                      DIM wg-count wgsm wgsn)

          (enq-read! cqueue cl-sub-acc (buffer c0))
          (set-arg! sum-reduce-kernel 1 d-acc)
          (enq-reduce cqueue sum-reduce-kernel sum-reduction-kernel
                      DIM wg-count wgsm wgsn)

          (enq-read! cqueue cl-sub-acc (buffer d))
          (->Autocorrelation (fmap (double-fn /) d c0) mean-vec
                             (fmap! sqrt (scal! (/ 1.0 (* i-max n)) (copy d)))
                             (* n walker-count) n walker-count lag 0.0))
        (throw (IllegalArgumentException.
                (format (str "The autocorrelation time is too long relative to the variance."
                             "Number of steps (%d) must not be less than %d.")
                        n (* lag min-fac)))))))
  RandomSampler
  (init! [this seed]
    (let [seed (wrap-int seed)]
      (set-arg! stretch-move-odd-kernel 0 (inc! seed))
      (set-arg! stretch-move-even-kernel 0 (inc! seed))
      (set-arg! stretch-move-odd-bare-kernel 0 (inc! seed))
      (set-arg! stretch-move-even-bare-kernel 0 (inc! seed))
      (enq-fill! cqueue cl-accept (int-array 1))
      (aset step-counter 0 0)
      this))
  (sample! [this res]
    (let [res (cond
                (and (= 1 DIM) (number? res)) (create-raw neanderthal-factory res)
                (and (= 1 DIM) (vect? res)) res
                (and (< 1 DIM) (number? res)) (create-raw neanderthal-factory DIM res)
                (and (< 1 DIM) (matrix? res)) res
                :default (throw (IllegalArgumentException.
                                 "sample! requires valid res argument.")))
          res-buff (buffer res)
          n (if (vect? res) (dim res) (ncols res))
          available (* DIM Float/BYTES walker-count)]
      (do
        (loop [ofst 0 requested (* DIM Float/BYTES (long n))]
          (do (move-bare! this)
              (if (<= requested available)
                (enq-copy! cqueue cl-xs res-buff 0 ofst requested nil nil)
                (do
                  (enq-copy! cqueue cl-xs res-buff 0 ofst available nil nil)
                  (recur (+ ofst available) (- requested available))))))
        res)))
  MCMC
  (set-position! [this position]
    (do
      (if (cl-buffer? position)
        (if (<= (size cl-xs) (size position))
          (enq-copy! cqueue position cl-xs)
          (throw (IllegalArgumentException. "Position buffer too short.")))
        (let [seed (wrap-int position)]
          (set-arg! init-walkers-kernel 0 seed)
          (enq-nd! cqueue init-walkers-kernel
                   (work-size-1d (* DIM (/ walker-count 4))))))
      (enq-nd! cqueue logpdf-kernel (work-size-1d walker-count))
      this))
  (burn-in! [this n a]
    (do
      (aset step-counter 0 0)
      (set-arg! stretch-move-odd-bare-kernel 5 a)
      (set-arg! stretch-move-even-bare-kernel 5 a)
      (dotimes [i (dec (long n))]
        (move-bare! this))
      (let [means-count (long (count-work-groups WGS (/ walker-count 2)))]
        (with-release [cl-means-acc (cl-buffer ctx (* DIM Float/BYTES means-count)
                                               :read-write)]
          (aset step-counter 0 0)
          (enq-fill! cqueue cl-accept (int-array 1))
          (set-args! stretch-move-odd-kernel 6 cl-means-acc a)
          (set-args! stretch-move-even-kernel 6 cl-means-acc a)
          (move! this)
          (acc-rate this)))))
  (run-sampler! [this n a]
    (let [means-count (long (count-work-groups WGS (/ walker-count 2)))
          local-m (min means-count WGS)
          local-n (long (/ WGS local-m))
          acc-count (long (count-work-groups local-m means-count))
          wgsn (min acc-count WGS)
          wgsm (long (/ WGS wgsn))]
      (with-release [cl-means-acc (cl-buffer ctx
                                             (* DIM Float/BYTES means-count (long n))
                                             :read-write)
                     cl-acc (cl-buffer ctx (* DIM Float/BYTES acc-count (long n))
                                       :read-write)
                     cl-means (cl-sub-buffer cl-acc 0 (* DIM Float/BYTES (long n)))]
        (aset step-counter 0 0)
        (enq-fill! cqueue cl-means-acc (float-array 1))
        (set-args! stretch-move-odd-kernel 6 cl-means-acc a)
        (set-args! stretch-move-even-kernel 6 cl-means-acc a)
        (dotimes [i n]
          (move! this))
        (set-arg! sum-reduction-kernel 0 cl-acc)
        (set-args! sum-means-kernel 0 cl-acc cl-means-acc)
        (enq-reduce cqueue sum-means-kernel sum-reduction-kernel
                    means-count (* DIM (long n)) local-m local-n wgsm wgsn)
        (set-args! scal-kernel 0 (wrap-float (/ means-count)) cl-means)
        (enq-nd! cqueue scal-kernel (work-size-1d (* DIM (long n))))
        (assoc (acor this cl-means)
               :acc-rate (acc-rate this))))))

(deftype GCNStretchFactory [ctx queue neanderthal-factory prog
                            ^long DIM ^long WGS]
  Releaseable
  (release [_]
    (release prog))
  MCMCFactory
  (mcmc-sampler [_ walker-count params host-lower host-upper]
    (let [walker-count (long walker-count)]
      (if (and (<= (* 2 WGS) walker-count) (zero? (rem walker-count (* 2 WGS))))
        (let [cnt (long (/ walker-count 2))
              accept-count (count-work-groups WGS cnt)
              accept-acc-count (count-work-groups WGS accept-count)
              bytecount (long (* Float/BYTES cnt DIM))
              step-counter (int-array 1)
              cl-params (buffer params)
              cl-xs (cl-buffer ctx (* 2 bytecount) :read-write)
              cl-s0 (cl-sub-buffer cl-xs 0 bytecount :read-write)
              cl-s1 (cl-sub-buffer cl-xs bytecount bytecount :read-write)
              cl-logpdf-xs (cl-buffer ctx (* 2 bytecount) :read-write)
              cl-logpdf-s0 (cl-sub-buffer cl-logpdf-xs 0 bytecount :read-write)
              cl-logpdf-s1 (cl-sub-buffer cl-logpdf-xs bytecount bytecount :read-write)
              cl-accept (cl-buffer ctx (* Integer/BYTES accept-count) :read-write)
              cl-accept-acc (cl-buffer ctx (* Long/BYTES accept-acc-count) :read-write)
              cl-lower (cl-buffer ctx (* Float/BYTES DIM) :read-only)
              cl-upper (cl-buffer ctx (* Float/BYTES DIM) :read-only)]
          (enq-write! queue cl-lower (buffer host-lower))
          (enq-write! queue cl-upper (buffer host-upper))
          (->GCNStretch
           ctx queue neanderthal-factory
           walker-count (work-size-1d (/ walker-count 2)) DIM WGS step-counter
           cl-params cl-xs cl-s0 cl-s1 cl-logpdf-xs cl-logpdf-s0 cl-logpdf-s1
           cl-accept cl-accept-acc
           (doto (kernel prog "stretch_move_accu")
             (set-args! 1 cl-params cl-s1 cl-s0 cl-logpdf-s0 cl-accept))
           (doto (kernel prog "stretch_move_accu")
             (set-args! 1 cl-params cl-s0 cl-s1 cl-logpdf-s1 cl-accept))
           (doto (kernel prog "stretch_move_bare")
             (set-args! 1 cl-params cl-s1 cl-s0 cl-logpdf-s0))
           (doto (kernel prog "stretch_move_bare")
             (set-args! 1 cl-params cl-s0 cl-s1 cl-logpdf-s1))
           (doto (kernel prog "init_walkers")
             (set-args! 1 cl-lower cl-upper cl-xs))
           (doto (kernel prog "logpdf")
             (set-args! 0 cl-params cl-xs cl-logpdf-xs))
           (doto (kernel prog "sum_accept_reduction")
             (set-arg! 0 cl-accept-acc))
           (doto (kernel prog "sum_accept_reduce")
             (set-args! 0 cl-accept-acc cl-accept))
           (kernel prog "sum_means_vertical")
           (kernel prog "sum_reduction_horizontal")
           (kernel prog "sum_reduce_horizontal")
           (kernel prog "scal")
           (kernel prog "subtract_mean")
           (kernel prog "autocovariance")))
        (throw (IllegalArgumentException.
                (format "Number of walkers (%d) must be a multiple of %d."
                        walker-count (* 2 WGS))))))))

(let [reduction-src (slurp (io/resource "uncomplicate/clojurecl/kernels/reduction.cl"))
      kernels-src (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/dist-kernels.cl"))
      uniform-sample-src (slurp (io/resource "uncomplicate/bayadera/opencl/rng/uniform-sampler.cl"))
      stretch-common-src (slurp (io/resource "uncomplicate/bayadera/opencl/mcmc/amd-gcn-stretch-generic.cl"))
      stretch-move-src (slurp (io/resource "uncomplicate/bayadera/opencl/mcmc/amd-gcn-stretch-move.cl"))
      compiler-options "-cl-std=CL2.0 -DLOGPDF=%s -DACCUMULATOR=float -DREAL=float -DPARAMS_SIZE=%d -DDIM=%d -DWGS=%d -I%s/"]

  (defn gcn-stretch-factory
    ([ctx cqueue tmp-dir-name neanderthal-factory model WGS]
     (let [prog-src (program-with-source
                     ctx (op [uniform-sample-src reduction-src]
                             (source model)
                             [kernels-src stretch-common-src stretch-move-src]))]
       (->GCNStretchFactory
        ctx cqueue neanderthal-factory
        (build-program! prog-src
                        (format compiler-options
                                (mcmc-logpdf model) (params-size model)
                                (dimension model) WGS tmp-dir-name)
                        nil)
        (dimension model) WGS)))
    ([ctx cqueue neanderthal-factory model]
     (let [tmp-dir-name (get-tmp-dir-name)]
       (with-philox tmp-dir-name
         (gcn-stretch-factory ctx cqueue tmp-dir-name neanderthal-factory model 256))))))
