;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.device.amd-gcn
  (:require [clojure.java.io :as io]
            [uncomplicate.commons
             [core :refer [Releaseable release with-release let-release Info info wrap-int]]
             [utils :refer [count-groups dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [max-compute-units max-work-group-size queue-device]]
             [toolbox :refer [enq-reduce! enq-read-long enq-read-double]]]
            [uncomplicate.neanderthal
             [core :refer [vctr ge ncols mrows scal! transfer transfer! raw submatrix dim]]
             [math :refer [sqrt]]
             [vect-math :refer [sqrt! div div!]]
             [block :refer [buffer create-data-source wrap-prim initialize entry-width data-accessor
                            offset stride count-entries]]
             [opencl :refer [opencl-float]]]
            [uncomplicate.neanderthal.internal.api :as na]
            [uncomplicate.neanderthal.internal.device.random123 :refer [temp-dir]]
            [uncomplicate.bayadera.internal.protocols :refer :all]))

;; ============================ Private utillities =============================

(defn ^:private inc! [^ints a]
  (aset a 0 (inc (aget a 0)))
  a)

(defn ^:private add! [^longs a ^long n]
  (aset a 0 (+ (aget a 0) n))
  a)

(defn ^:private add-long ^long [^long x ^long y]
  (+ x y))

(defn ^:private inc-long ^long [^long x]
  (inc x))

;; ============================ Direct sampler =================================

(deftype GCNDirectSamplerEngine [cqueue prog]
  Releaseable
  (release [_]
    (release prog))
  RandomSamplerEngine
  (sample [this seed cl-params res]
    (if (and (= 0 (rem (ncols res) 4)) (= 0 (rem (offset res) 4)))
      (with-release [sample-kernel (kernel prog "sample")]
        (set-args! sample-kernel 0 (buffer cl-params) (wrap-int seed) (buffer res) (wrap-int (/ (offset res) 4)))
        (enq-kernel! cqueue sample-kernel (work-size-1d (/ (ncols res) 4)))
        res)
      (dragan-says-ex "GCN direct sampler supports only matrices with ncols and offset that are multiple of 4."
                      {:ncols (ncols res)
                       :offset (offset res)}))))

;; ============================ Distribution engine ============================

(deftype GCNDistributionEngine [ctx cqueue prog ^long WGS dist-model]
  Releaseable
  (release [_]
    (release prog))
  ModelProvider
  (model [_];;TODO too complicated. models are everywhere. possibly move out?
    dist-model)
  DensityEngine
  (log-density [this cl-params x]
    (let [params-len (long (params-size dist-model))
          data-len (max 0 (- ^long (count-entries (data-accessor cl-params) (buffer cl-params))
                             params-len))]
      (let-release [res (vctr x (ncols x))]
        (with-release [logpdf-kernel (kernel prog "logpdf")]
          (set-args! logpdf-kernel 0 (wrap-int data-len) (wrap-int params-len) (buffer cl-params)
                     (wrap-int (mrows x)) (buffer x) (buffer res))
          (enq-kernel! cqueue logpdf-kernel (work-size-1d (ncols x)))
          res))))
  (density [this cl-params x]
    (let [params-len (long (params-size dist-model))
          data-len (max 0 (- ^long (count-entries (data-accessor cl-params) (buffer cl-params))
                             params-len))]
      (let-release [res (vctr x (ncols x))]
        (with-release [pdf-kernel (kernel prog "pdf")]
          (set-args! pdf-kernel 0 (wrap-int data-len) (wrap-int params-len) (buffer cl-params)
                     (wrap-int (mrows x)) (buffer x) (buffer res))
          (enq-kernel! cqueue pdf-kernel (work-size-1d (ncols x)))
          res)))))

;; ============================ Likelihood engine ============================

(deftype GCNLikelihoodEngine [ctx cqueue prog ^long WGS lik-model]
  Releaseable
  (release [_]
    (release prog))
  ModelProvider
  (model [_]
    lik-model)
  DensityEngine
  (log-density [this cl-data x]
    (let [data-len (wrap-int (count-entries (data-accessor cl-data) (buffer cl-data)))]
      (let-release [res (vctr x (ncols x))]
        (with-release [loglik-kernel (kernel prog "loglik")]
          (set-args! loglik-kernel 0 data-len (buffer cl-data)
                     (wrap-int (mrows x)) (buffer x) (buffer res))
          (enq-kernel! cqueue loglik-kernel (work-size-1d (ncols x)))
          res))))
  (density [this cl-data x]
    (let [data-len (wrap-int (count-entries (data-accessor cl-data) (buffer cl-data)))]
      (let-release [res (vctr x (ncols x))]
        (with-release [lik-kernel (kernel prog "lik")]
          (set-args! lik-kernel 0 data-len (buffer cl-data)
                     (wrap-int (mrows x)) (buffer x) (buffer res))
          (enq-kernel! cqueue lik-kernel (work-size-1d (ncols x)))
          res))))
  LikelihoodEngine
  (evidence [this cl-data x]
    (let [n (ncols x)
          acc-size (* Double/BYTES (count-groups WGS n))
          data-len (wrap-int (count-entries (data-accessor cl-data) (buffer cl-data)))]
      (with-release [evidence-kernel (kernel prog "evidence_reduce")
                     sum-reduction-kernel (kernel prog "sum_reduction")
                     cl-acc (cl-buffer ctx acc-size :read-write)]
        (set-args! evidence-kernel 0 cl-acc data-len (buffer cl-data)
                   (wrap-int (mrows x)) (buffer x))
        (set-arg! sum-reduction-kernel 0 cl-acc)
        (enq-reduce! cqueue evidence-kernel sum-reduction-kernel n WGS)
        (/ (enq-read-double cqueue cl-acc) n)))))

;; ============================ Dataset engine =================================

(deftype GCNDatasetEngine [ctx cqueue prog ^long WGS]
  Releaseable
  (release [_]
    (release prog))
  DatasetEngine
  (data-mean [this data-matrix]
    (let [m (mrows data-matrix)
          n (ncols data-matrix)
          wgsn (min n WGS)
          wgsm (/ WGS wgsn)
          acc-size (max 1 (* m (count-groups wgsn n)))]
      (let-release [res-vec (vctr data-matrix m)]
        (with-release [cl-acc (create-data-source data-matrix acc-size)
                       sum-reduction-kernel (kernel prog "sum_reduction_horizontal")
                       mean-kernel (kernel prog "mean_reduce")]
          (set-arg! sum-reduction-kernel 0 cl-acc)
          (set-args! mean-kernel cl-acc
                     (buffer data-matrix) (wrap-int (offset data-matrix)) (wrap-int (stride data-matrix)))
          (enq-reduce! cqueue mean-kernel sum-reduction-kernel m n wgsm wgsn)
          (enq-copy! cqueue cl-acc (buffer res-vec))
          (scal! (/ 1.0 n) res-vec)))))
  (data-variance [this data-matrix]
    (let [m (mrows data-matrix)
          n (ncols data-matrix)
          wgsn (min n WGS)
          wgsm (/ WGS wgsn)
          acc-size (max 1 (* m (count-groups wgsn n)))]
      (let-release [res (vctr data-matrix m)]
        (with-release [cl-acc (create-data-source data-matrix acc-size)
                       sum-reduction-kernel (kernel prog "sum_reduction_horizontal")
                       mean-kernel (kernel prog "mean_reduce")
                       variance-kernel (kernel prog "variance_reduce")]
          (set-arg! sum-reduction-kernel 0 cl-acc)
          (set-args! mean-kernel cl-acc
                     (buffer data-matrix) (wrap-int (offset data-matrix)) (wrap-int (stride data-matrix)))
          (enq-reduce! cqueue mean-kernel sum-reduction-kernel m n wgsm wgsn)
          (enq-copy! cqueue cl-acc (buffer res))
          (scal! (/ 1.0 n) res)
          (set-args! variance-kernel 0 cl-acc
                     (buffer data-matrix) (wrap-int (offset data-matrix)) (wrap-int (stride data-matrix))
                     (buffer res))
          (enq-reduce! cqueue variance-kernel sum-reduction-kernel m n wgsm wgsn)
          (enq-copy! cqueue cl-acc (buffer res))
          (scal! (/ 1.0 n) res)))))
  EstimateEngine
  (histogram [this data-matrix]
    (let [m (mrows data-matrix)
          n (ncols data-matrix)
          wgsm (min m (long (sqrt WGS)))
          wgsn (long (/ WGS wgsm))
          acc-size (* 2 (max 1 (* m (count-groups wgsn n))))
          claccessor (data-accessor data-matrix)]
      (with-release [cl-min-max (create-data-source claccessor acc-size)
                     uint-res (cl-buffer ctx (* Integer/BYTES WGS m) :read-write)
                     limits (ge data-matrix 2 m)
                     result (ge data-matrix WGS m)
                     bin-ranks (ge data-matrix WGS m)
                     min-max-reduction-kernel (kernel prog "min_max_reduction")
                     min-max-kernel (kernel prog "min_max_reduce")
                     histogram-kernel (kernel prog "histogram")
                     uint-to-real-kernel (kernel prog "uint_to_real")
                     local-sort-kernel (kernel prog "bitonic_local")]
        (set-arg! min-max-reduction-kernel 0 cl-min-max)
        (set-args! min-max-kernel cl-min-max
                   (buffer data-matrix) (wrap-int (offset data-matrix)) (wrap-int (stride data-matrix)))
        (enq-reduce! cqueue min-max-kernel min-max-reduction-kernel m n wgsm wgsn)
        (enq-copy! cqueue cl-min-max (buffer limits))
        (enq-fill! cqueue uint-res (wrap-int 0))
        (set-args! histogram-kernel (buffer limits)
                   (buffer data-matrix) (wrap-int (offset data-matrix)) (wrap-int (stride data-matrix))
                   uint-res)
        (enq-kernel! cqueue histogram-kernel (work-size-2d m n 1 WGS))
        (set-args! uint-to-real-kernel (wrap-prim claccessor (/ WGS n)) (buffer limits)
                   uint-res (buffer result))
        (enq-kernel! cqueue uint-to-real-kernel (work-size-2d WGS m WGS 1))
        (set-args! local-sort-kernel (buffer result) (buffer bin-ranks))
        (enq-kernel! cqueue local-sort-kernel (work-size-1d (* m WGS)))
        (->Histogram (transfer limits) (transfer result) (transfer bin-ranks))))))

(deftype GCNAcorEngine [ctx cqueue prog ^long WGS]
  Releaseable
  (release [_]
    (release prog))
  AcorEngine
  (acor [_ data-matrix]
    (let [n (ncols data-matrix)
          dim (mrows data-matrix)
          min-fac 4
          min-lag 4
          max-lag 256
          lag (max min-lag (min (quot n min-fac) WGS max-lag))
          win-mult 4
          wgsm (min 16 dim WGS)
          wgsn (long (/ WGS wgsm))
          wg-count (count-groups wgsn n)
          native-fact (na/native-factory data-matrix)]
      (if (<= (* lag min-fac) n)
        (let-release [tau (vctr native-fact dim)
                      mean (vctr native-fact dim)
                      sigma (vctr native-fact dim)]
          (with-release [cl-acc (create-data-source data-matrix (* dim wg-count))
                         cl-vec (vctr data-matrix dim)
                         d-acc (create-data-source data-matrix (* dim wg-count))
                         sum-reduction-kernel (kernel prog "sum_reduction_horizontal")
                         sum-reduce-kernel (kernel prog "sum_reduce_horizontal")
                         subtract-mean-kernel (kernel prog "subtract_mean")
                         acor-2d-kernel (kernel prog "acor_2d")
                         acor-kernel (kernel prog "acor")]
            (set-arg! sum-reduction-kernel 0 cl-acc)
            (set-args! sum-reduce-kernel 0 cl-acc (buffer data-matrix))
            (enq-reduce! cqueue sum-reduce-kernel sum-reduction-kernel dim n wgsm wgsn)
            (enq-copy! cqueue cl-acc (buffer cl-vec))
            (scal! (/ 1.0 n) cl-vec)
            (set-args! subtract-mean-kernel 0 (buffer data-matrix) (buffer cl-vec))
            (enq-kernel! cqueue subtract-mean-kernel (work-size-2d dim n))
            (transfer! cl-vec mean)
            (enq-fill! cqueue cl-acc (int-array 1))
            (enq-fill! cqueue d-acc (int-array 1))
            (set-args! acor-2d-kernel 0 (wrap-int lag) cl-acc d-acc (buffer data-matrix))
            (enq-kernel! cqueue acor-2d-kernel (work-size-2d n dim))
            (set-args! acor-kernel (wrap-int n)
                       (wrap-int lag) (wrap-int min-lag) (wrap-int win-mult)
                       cl-acc d-acc (buffer data-matrix))
            (enq-kernel! cqueue acor-kernel (work-size-1d dim (min dim WGS)))
            (enq-read! cqueue cl-acc (buffer tau))
            (enq-read! cqueue d-acc (buffer sigma))
            (->Autocorrelation tau mean sigma n lag)))
        (throw (IllegalArgumentException.
                (format (str "The autocorrelation time is too long relative to the variance. "
                             "Number of steps (%d) must not be less than %d.")
                        n (* lag min-fac))))))))

;; ======================== MCMC engine ========================================

(deftype GCNStretch [ctx cqueue neanderthal-factory claccessor acor-eng
                     ^long walker-count wsize cl-model ^long DIM ^long WGS
                     ^ints move-counter ^ints move-bare-counter iteration-counter
                     ^ints move-seed
                     cl-params cl-xs cl-s0 cl-s1
                     cl-logfn-xs cl-logfn-s0 cl-logfn-s1
                     cl-accept cl-accept-acc cl-acc
                     stretch-move-odd-kernel stretch-move-even-kernel
                     stretch-move-odd-bare-kernel stretch-move-even-bare-kernel
                     init-walkers-kernel logfn-kernel
                     sum-accept-reduction-kernel sum-accept-kernel
                     sum-means-kernel
                     sum-reduction-kernel sum-reduce-kernel
                     min-max-reduction-kernel
                     min-max-kernel
                     histogram-kernel
                     uint-to-real-kernel
                     local-sort-kernel
                     mean-kernel
                     variance-kernel]
  Releaseable
  (release [_]
    (release cl-xs)
    (release cl-s0)
    (release cl-s1)
    (release cl-logfn-xs)
    (release cl-logfn-s0)
    (release cl-logfn-s1)
    (release cl-accept)
    (release cl-accept-acc)
    (release cl-acc)
    (release stretch-move-odd-kernel)
    (release stretch-move-even-kernel)
    (release stretch-move-odd-bare-kernel)
    (release stretch-move-even-bare-kernel)
    (release init-walkers-kernel)
    (release logfn-kernel)
    (release sum-accept-reduction-kernel)
    (release sum-accept-kernel)
    (release sum-means-kernel)
    (release sum-reduction-kernel)
    (release sum-reduce-kernel)
    (release min-max-reduction-kernel)
    (release min-max-kernel)
    (release histogram-kernel)
    (release uint-to-real-kernel)
    (release local-sort-kernel)
    (release mean-kernel)
    (release variance-kernel)
    true)
  Info
  (info [this]
    {:walker-count walker-count
     :iteration-counter @iteration-counter})
  ModelProvider
  (model [this]
    cl-model)
  MCMCStretch
  (init-move! [this cl-means-acc a]
    (let [seed (+ 2 (aget move-seed 0))]
      (aset move-seed 0 seed)
      (set-arg! stretch-move-odd-kernel 0 (wrap-int seed))
      (set-arg! stretch-move-even-kernel 0 (wrap-int (inc (int seed))))
      (aset move-counter 0 0)
      (enq-fill! cqueue cl-accept (int-array 1))
      (initialize claccessor cl-means-acc)
      (set-args! stretch-move-odd-kernel 9 cl-means-acc a)
      (set-args! stretch-move-even-kernel 9 cl-means-acc a))
    this)
  (move! [this]
    (set-arg! stretch-move-odd-kernel 11 move-counter)
    (enq-kernel! cqueue stretch-move-odd-kernel wsize)
    (set-arg! stretch-move-even-kernel 11 move-counter)
    (enq-kernel! cqueue stretch-move-even-kernel wsize)
    (inc! move-counter)
    this)
  (move-bare! [this]
    (set-arg! stretch-move-odd-bare-kernel 10 move-bare-counter)
    (enq-kernel! cqueue stretch-move-odd-bare-kernel wsize)
    (set-arg! stretch-move-even-bare-kernel 10 move-bare-counter)
    (enq-kernel! cqueue stretch-move-even-bare-kernel wsize)
    (inc! move-bare-counter)
    this)
  (set-temperature! [this t]
    (let [beta (wrap-prim claccessor (/ 1.0 ^double t))]
      (set-arg! stretch-move-odd-bare-kernel 9 beta)
      (set-arg! stretch-move-even-bare-kernel 9 beta))
    this)
  RandomSampler
  (sample! [this]
    (sample! this walker-count))
  (sample! [this n-or-res]
   (let [available (* DIM (entry-width claccessor) walker-count)]
      (let-release [res (if (integer? n-or-res)
                          (ge neanderthal-factory DIM n-or-res {:raw true})
                          n-or-res)]
        (set-temperature! this 1.0)
        (loop [ofst 0 requested (* DIM (entry-width claccessor) (ncols res))]
          (move-bare! this)
          (vswap! iteration-counter inc-long)
          (if (<= requested available)
            (enq-copy! cqueue cl-xs (buffer res) 0 ofst requested nil nil)
            (do
              (enq-copy! cqueue cl-xs (buffer res) 0 ofst available nil nil)
              (recur (+ ofst available) (- requested available)))))
        res)))
  MCMC
  (init! [this seed]
    (let [a (wrap-prim claccessor 2.0)]
      (set-arg! stretch-move-odd-bare-kernel 0 (wrap-int seed))
      (set-arg! stretch-move-even-bare-kernel 0 (wrap-int (inc (int seed))))
      (set-arg! stretch-move-odd-bare-kernel 8 a)
      (set-arg! stretch-move-even-bare-kernel 8 a)
      (set-temperature! this 1.0)
      (aset move-bare-counter 0 0)
      (aset move-seed 0 (int seed))
      this))
  (init-position! [this position]
    (let [cl-position (.cl-xs ^GCNStretch position)]
      (enq-copy! cqueue cl-position cl-xs)
      (enq-kernel! cqueue logfn-kernel (work-size-1d walker-count))
      (vreset! iteration-counter 0)
      this))
  (init-position! [this seed limits]
    (let [seed (wrap-int seed)]
      (with-release [cl-limits (transfer neanderthal-factory limits)]
        (set-args! init-walkers-kernel 0 seed (buffer cl-limits))
        (enq-kernel! cqueue init-walkers-kernel
                     (work-size-1d (* DIM (long (/ walker-count 4)))))
        (enq-kernel! cqueue logfn-kernel (work-size-1d walker-count))
        (vreset! iteration-counter 0)
        this)))
  (burn-in! [this n a]
    (let [a (wrap-prim claccessor a)]
      (set-arg! stretch-move-odd-bare-kernel 8 a)
      (set-arg! stretch-move-even-bare-kernel 8 a)
      (set-temperature! this 1.0)
      (dotimes [i n]
        (move-bare! this))
      (vswap! iteration-counter add-long n)
      this))
  (anneal! [this schedule n a]
    (let [a (wrap-prim claccessor a)]
      (set-arg! stretch-move-odd-bare-kernel 8 a)
      (set-arg! stretch-move-even-bare-kernel 8 a)
      (dotimes [i n]
        (set-temperature! this (schedule i))
        (move-bare! this))
      (vswap! iteration-counter add-long n)
      this))
  (run-sampler! [this n a]
    (let [a (wrap-prim claccessor a)
          n (long n)
          means-count (long (count-groups WGS (/ walker-count 2)))
          local-m (min means-count WGS)
          local-n (long (/ WGS local-m))
          acc-count (long (count-groups local-m means-count))]
      (with-release [cl-means-acc (create-data-source claccessor (* DIM means-count n))
                     acc (ge neanderthal-factory DIM (* acc-count n))
                     means (submatrix acc 0 0 DIM n)]
        (init-move! this cl-means-acc a)
        (dotimes [i n]
          (move! this))
        (vswap! iteration-counter add-long n)
        (set-arg! sum-reduction-kernel 0 (buffer acc))
        (set-args! sum-means-kernel 0 (buffer acc) cl-means-acc)
        (enq-reduce! cqueue sum-means-kernel sum-reduction-kernel
                     (* DIM n) means-count local-n local-m)
        (scal! (/ 0.5 (* WGS means-count)) means)
        (enq-reduce! cqueue sum-accept-kernel sum-accept-reduction-kernel means-count WGS)
        {:acceptance-rate (/ (double (enq-read-long cqueue cl-accept-acc)) (* walker-count n))
         :a (get a 0)
         :autocorrelation (acor acor-eng means)})))
  (acc-rate! [this a]
    (let [a (wrap-prim claccessor a)
          means-count (long (count-groups WGS (/ walker-count 2)))]
      (with-release [cl-means-acc (create-data-source claccessor (* DIM means-count))]
        (init-move! this cl-means-acc a)
        (move! this)
        (vswap! iteration-counter inc-long)
        (enq-reduce! cqueue sum-accept-kernel sum-accept-reduction-kernel
                     (count-groups WGS (/ walker-count 2)) WGS)
        (/ (double (enq-read-long cqueue cl-accept-acc)) walker-count))))
  EstimateEngine
  (histogram [this]
    (histogram! this 1))
  (histogram! [this cycles]
    (let [cycles (long cycles)
          n (* cycles walker-count)
          wgsm (min DIM (long (sqrt WGS)))
          wgsn (long (/ WGS wgsm))
          histogram-worksize (work-size-2d DIM walker-count 1 WGS)
          acc-size (* 2 (max 1 (* DIM (count-groups wgsn walker-count))))]
      (with-release [cl-min-max (create-data-source claccessor acc-size)
                     uint-res (cl-buffer ctx (* Integer/BYTES WGS DIM) :read-write)
                     result (ge neanderthal-factory WGS DIM)
                     limits (ge neanderthal-factory 2 DIM)
                     bin-ranks (ge neanderthal-factory WGS DIM)]
        (set-arg! min-max-reduction-kernel 0 cl-min-max)
        (set-args! min-max-kernel cl-min-max cl-xs)
        (enq-reduce! cqueue min-max-kernel min-max-reduction-kernel DIM walker-count wgsm wgsn)
        (enq-copy! cqueue cl-min-max (buffer limits))
        (enq-fill! cqueue uint-res (wrap-int 0))
        (set-args! histogram-kernel (buffer limits) cl-xs)
        (set-arg! histogram-kernel 4 uint-res)
        (enq-kernel! cqueue histogram-kernel histogram-worksize)
        (set-temperature! this 1.0)
        (dotimes [i (dec cycles)]
          (move-bare! this)
          (enq-kernel! cqueue histogram-kernel histogram-worksize))
        (vswap! iteration-counter add-long (dec cycles))
        (set-args! uint-to-real-kernel (wrap-prim claccessor (/ WGS n)) (buffer limits)
                   uint-res (buffer result))
        (enq-kernel! cqueue uint-to-real-kernel (work-size-2d WGS DIM WGS 1))
        (set-args! local-sort-kernel (buffer result) (buffer bin-ranks))
        (enq-kernel! cqueue local-sort-kernel (work-size-1d (* DIM WGS)))
        (->Histogram (transfer limits) (transfer result) (transfer bin-ranks)))))
  Location
  (mean [_]
    (let-release [res-vec (vctr neanderthal-factory DIM)]
      (set-arg! sum-reduction-kernel 0 cl-acc)
      (enq-reduce! cqueue mean-kernel sum-reduction-kernel DIM walker-count 1 WGS)
      (enq-copy! cqueue cl-acc (buffer res-vec))
      (scal! (/ 1.0 walker-count) res-vec)))
  Spread
  (variance [_]
    (let-release [res-vec (vctr neanderthal-factory DIM)]
      (set-arg! sum-reduction-kernel 0 cl-acc)
      (enq-reduce! cqueue mean-kernel sum-reduction-kernel DIM walker-count 1 WGS)
      (enq-copy! cqueue cl-acc (buffer res-vec))
      (scal! (/ 1.0 walker-count) res-vec)
      (set-arg! variance-kernel 4 (buffer res-vec))
      (enq-reduce! cqueue variance-kernel sum-reduction-kernel DIM walker-count 1 WGS)
      (enq-copy! cqueue cl-acc (buffer res-vec))
      (scal! (/ 1.0 walker-count) res-vec)))
  (sd [this]
    (sqrt! (variance this))))

(deftype GCNStretchFactory [ctx queue prog neanderthal-factory acor-eng model ^long DIM ^long WGS]
  Releaseable
  (release [_]
    (release prog))
  SamplerFactory
  (create-sampler [_ seed walker-count params]
    (let [walker-count (long walker-count)]
      (if (and (<= (* 2 WGS) walker-count) (zero? (rem walker-count (* 2 WGS))))
        (let [acc-count (* DIM (count-groups WGS walker-count))
              accept-count (count-groups WGS (/ walker-count 2))
              accept-acc-count (count-groups WGS accept-count)
              claccessor (data-accessor neanderthal-factory)
              sub-bytesize (* DIM (long (/ walker-count 2)) (entry-width claccessor))
              cl-params (buffer params)
              params-len (wrap-int (params-size model))
              data-len (wrap-int (- (count-entries claccessor cl-params) ^long (params-size model)))
              arr-0 (wrap-int 0)
              arr-DIM (wrap-int DIM)]
          (let-release [cl-xs (create-data-source claccessor (* DIM walker-count))
                        cl-s0 (cl-sub-buffer cl-xs 0 sub-bytesize :read-write)
                        cl-s1 (cl-sub-buffer cl-xs sub-bytesize sub-bytesize :read-write)
                        cl-logfn-xs (create-data-source claccessor (* DIM walker-count))
                        cl-logfn-s0 (cl-sub-buffer cl-logfn-xs 0 sub-bytesize :read-write)
                        cl-logfn-s1 (cl-sub-buffer cl-logfn-xs sub-bytesize sub-bytesize :read-write)
                        cl-accept (cl-buffer ctx (* Integer/BYTES accept-count) :read-write)
                        cl-accept-acc (cl-buffer ctx (* Long/BYTES accept-acc-count) :read-write)
                        cl-acc (create-data-source claccessor acc-count)]
            (doto
                (->GCNStretch
                 ctx queue neanderthal-factory claccessor acor-eng walker-count
                 (work-size-1d (/ walker-count 2)) model DIM WGS
                 (int-array 1) (int-array 1) (volatile! 0) (int-array 1)
                 cl-params cl-xs cl-s0 cl-s1 cl-logfn-xs cl-logfn-s0 cl-logfn-s1
                 cl-accept cl-accept-acc cl-acc
                 (set-args! (kernel prog "stretch_move_accu") 1 (wrap-int 1111)
                            data-len params-len cl-params cl-s1 cl-s0 cl-logfn-s0 cl-accept)
                 (set-args! (kernel prog "stretch_move_accu") 1 (wrap-int 2222)
                            data-len params-len cl-params cl-s0 cl-s1 cl-logfn-s1 cl-accept)
                 (set-args! (kernel prog "stretch_move_bare") 1 (wrap-int 3333)
                            data-len params-len cl-params cl-s1 cl-s0 cl-logfn-s0)
                 (set-args! (kernel prog "stretch_move_bare") 1 (wrap-int 4444)
                            data-len params-len cl-params cl-s0 cl-s1 cl-logfn-s1)
                 (set-arg! (kernel prog "init_walkers") 2 cl-xs)
                 (set-args! (kernel prog "logfn") 0 data-len params-len cl-params cl-xs cl-logfn-xs)
                 (set-arg! (kernel prog "sum_accept_reduction") 0 cl-accept-acc)
                 (set-args! (kernel prog "sum_accept_reduce") 0 cl-accept-acc cl-accept)
                 (kernel prog "sum_means_vertical")
                 (kernel prog "sum_reduction_horizontal")
                 (kernel prog "sum_reduce_horizontal")
                 (kernel prog "min_max_reduction")
                 (set-args! (kernel prog "min_max_reduce") 2 arr-0 arr-DIM)
                 (set-args! (kernel prog "histogram") 2 arr-0 arr-DIM)
                 (kernel prog "uint_to_real")
                 (kernel prog "bitonic_local")
                 (set-args! (kernel prog "mean_reduce") 0 cl-acc cl-xs arr-0 arr-DIM)
                 (set-args! (kernel prog "variance_reduce") 0 cl-acc cl-xs arr-0 arr-DIM))
              (init! seed))))
        (throw (IllegalArgumentException.
                (format "Number of walkers (%d) must be a multiple of %d." walker-count (* 2 WGS))))))))

;; ======================== Constructor functions ==============================

(let [reduction-src (slurp (io/resource "uncomplicate/clojurecl/kernels/reduction.cl"))
      estimate-src (slurp (io/resource "uncomplicate/bayadera/internal/device/opencl/engines/amd-gcn-estimate.cl"))
      acor-src (slurp (io/resource "uncomplicate/bayadera/internal/device/opencl/engines/amd-gcn-acor.cl"))
      distribution-src (slurp (io/resource "uncomplicate/bayadera/internal/device/opencl/engines/amd-gcn-distribution.cl"))
      likelihood-src (slurp (io/resource "uncomplicate/bayadera/internal/device/opencl/engines/amd-gcn-likelihood.cl"))
      uniform-sampler-src (slurp (io/resource "uncomplicate/bayadera/internal/device/opencl/rng/uniform-sampler.cl"))
      mcmc-stretch-src (slurp (io/resource "uncomplicate/bayadera/internal/device/opencl/engines/amd-gcn-mcmc-stretch.cl"))
      dataset-options "-DCL_VERSION_2_0=200 -cl-std=CL2.0 -DREAL=float -DREAL2=float2 -DACCUMULATOR=float -DACCUMULATOR=float -DWGS=%d"
      distribution-options "-cl-std=CL2.0 -DREAL=float -DACCUMULATOR=float -DLOGPDF=%s -DWGS=%d"
      likelihood-options "-cl-std=CL2.0 -DREAL=float -DACCUMULATOR=double -DLOGLIK=%s -DWGS=%d"
      stretch-options "-cl-std=CL2.0 -DREAL=float -DREAL2=float2 -DACCUMULATOR=float -DLOGFN=%s -DDIM=%d -DWGS=%d -I%s/"]

  (defn gcn-dataset-engine
    ([ctx cqueue ^long WGS]
     (let-release [prog (build-program! (program-with-source ctx [reduction-src estimate-src])
                                        (format dataset-options WGS) nil)]
       (->GCNDatasetEngine ctx cqueue prog WGS)))
    ([ctx queue]
     (gcn-dataset-engine ctx queue (max-work-group-size (queue-device queue)))))

  (defn gcn-acor-engine
    ([ctx cqueue ^long WGS]
     (let-release [prog (build-program! (program-with-source ctx [reduction-src acor-src])
                                        (format dataset-options WGS) nil)]
       (->GCNAcorEngine ctx cqueue prog WGS)))
    ([ctx queue]
     (gcn-acor-engine ctx queue (max-work-group-size (queue-device queue)))))

  (defn gcn-distribution-engine
    ([ctx cqueue model WGS]
     (let-release [prog (build-program!
                         (program-with-source ctx (conj (source model) distribution-src))
                         (format distribution-options (logpdf model) WGS)
                         nil)]
       (->GCNDistributionEngine ctx cqueue prog WGS model))))

  (defn gcn-likelihood-engine
    ([ctx cqueue model WGS]
     (let-release [prog (build-program!
                         (program-with-source ctx (op (source model) [reduction-src likelihood-src]))
                         (format likelihood-options (loglik model) WGS)
                         nil)]
       (->GCNLikelihoodEngine ctx cqueue prog WGS model))))

  (defn gcn-direct-sampler-engine
    ([ctx cqueue include-dir model WGS]
     (let-release [prog (build-program!
                         (program-with-source ctx (op (source model) (sampler-source model)))
                         (format "-cl-std=CL2.0 -DREAL=float -DWGS=%d -I%s/" WGS include-dir)
                         nil)]
       (->GCNDirectSamplerEngine cqueue prog))))

  (defn gcn-stretch-factory
    [ctx cqueue include-dir neanderthal-factory dataset-eng model WGS]
    (let-release [prog (build-program!
                        (program-with-source
                         ctx (op [uniform-sampler-src reduction-src]
                                 (source model) [estimate-src mcmc-stretch-src]))
                        (format stretch-options (mcmc-logpdf model) (dimension model) WGS include-dir)
                        nil)]
      (->GCNStretchFactory ctx cqueue prog neanderthal-factory dataset-eng model (dimension model) WGS))))

;; =========================== Bayadera factory  ===========================

(defrecord GCNBayaderaFactory [ctx cqueue dev-queue include-dir ^long compute-units ^long WGS
                               neanderthal-factory dataset-eng acor-eng]
  Releaseable
  (release [_]
    (release dataset-eng)
    (release acor-eng)
    (release neanderthal-factory)
    (release dev-queue)
    true)
  na/MemoryContext
  (compatible? [_ o]
    (or (satisfies? DeviceModel o) (na/compatible? neanderthal-factory o)))
  na/FactoryProvider
  (factory [_]
    neanderthal-factory)
  (native-factory [_]
    (na/native-factory neanderthal-factory))
  EngineFactory
  (likelihood-engine [_ model]
    (gcn-likelihood-engine ctx cqueue model WGS))
  (distribution-engine [_ model]
    (gcn-distribution-engine ctx cqueue model WGS))
  (dataset-engine [_]
    dataset-eng)
  (direct-sampler-engine [_ model]
    (gcn-direct-sampler-engine ctx cqueue include-dir model WGS))
  (mcmc-factory [_ model]
    (gcn-stretch-factory ctx cqueue include-dir neanderthal-factory acor-eng model WGS))
  (processing-elements [_]
    (* compute-units WGS)))

(defn gcn-bayadera-factory
  ([ctx cqueue compute-units WGS]
   (let-release [neanderthal-factory (opencl-float ctx cqueue)
                 dataset-eng (gcn-dataset-engine ctx cqueue WGS)
                 acor-eng (gcn-acor-engine ctx cqueue WGS)
                 dev-queue (command-queue ctx (queue-device cqueue)
                                          :queue-on-device-default :queue-on-device
                                          :out-of-order-exec-mode)]
     (->GCNBayaderaFactory ctx cqueue dev-queue temp-dir compute-units WGS
                           neanderthal-factory dataset-eng acor-eng)))
  ([ctx cqueue]
   (let [dev (queue-device cqueue)]
     (gcn-bayadera-factory ctx cqueue (max-compute-units dev) (max-work-group-size dev)))))
