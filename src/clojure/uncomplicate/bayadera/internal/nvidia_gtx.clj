;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.internal.nvidia-gtx
  (:require [clojure.java.io :as io]
            [uncomplicate.commons.core
             :refer [Releaseable release with-release let-release wrap-int double-fn long-fn]]
            [uncomplicate.fluokitten.core :refer [fmap op]]
            [uncomplicate.clojurecuda
             [core :refer :all :as cuda :exclude [parameters]]
             [nvrtc :refer [compile! program]]
             [info :refer [multiprocessor-count max-block-dim-x ctx-device driver-version]]
             [toolbox :refer [launch-reduce! read-long read-double]]]
            [uncomplicate.neanderthal.internal.api :as na]
            [uncomplicate.neanderthal
             [core :refer [vctr ge ncols mrows scal! transfer raw submatrix dim copy!]]
             [math :refer [sqrt]]
             [vect-math :refer [sqrt! div]]
             [block :refer [buffer create-data-source wrap-prim initialize entry-width data-accessor
                            cast-prim]]
             [cuda :refer [cuda-float]]]
            [uncomplicate.bayadera.util :refer [srand-int]]
            [uncomplicate.bayadera.internal
             [protocols :refer :all]
             [models :refer [source sampler-source CLModel]]]))

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

(deftype GTXDirectSampler [ctx modl hstream ^long DIM ^long WGS]
  Releaseable
  (release [_]
    (release modl))
  RandomSampler
  (sample [this seed cu-params n]
    (in-context
     ctx
     (let-release [res (ge cu-params DIM n)]
       (with-release [sample-kernel (function modl "sample")]
         (launch! sample-kernel (grid-1d n WGS) hstream
                  (cuda/parameters (int n) (buffer cu-params) seed (buffer res)))
         res)))))

;; ============================ Distribution engine ============================

(deftype GTXDistributionEngine [ctx modl hstream ^long WGS dist-model
                                logpdf-kernel pdf-kernel evidence-kernel sum-reduction-kernel]
  Releaseable
  (release [_]
    (release logpdf-kernel)
    (release pdf-kernel)
    (release evidence-kernel)
    (release sum-reduction-kernel)
    (release modl))
  ModelProvider
  (model [_]
    dist-model)
  DistributionEngine
  (log-pdf [this cu-params x]
    (in-context
     ctx
     (let-release [res (vctr cu-params (ncols x))]
       (launch! logpdf-kernel (grid-1d (ncols x) WGS) hstream
                (cuda/parameters (ncols x) (buffer cu-params) (buffer x) (buffer res)))
       res)))
  (pdf [this cu-params x]
    (in-context
     ctx
     (let-release [res (vctr cu-params (ncols x))]
       (launch! pdf-kernel (grid-1d (ncols x) WGS) hstream
                (cuda/parameters (ncols x) (buffer cu-params) (buffer x) (buffer res)))
       res)))
  (evidence [this cu-params x]
    (if (satisfies? LikelihoodModel dist-model)
      (in-context
       ctx
       (let [n (ncols x)
             acc-size (* Double/BYTES (blocks-count WGS n))]
         (with-release [cu-acc (mem-alloc acc-size)]
           (launch-reduce! hstream evidence-kernel sum-reduction-kernel
                           [cu-acc (buffer cu-params) (buffer x)] [cu-acc] n WGS)
           (/ (read-double hstream cu-acc) n))))
      Double/NaN)))

;; ============================ Dataset engine =================================

(deftype GTXDatasetEngine [ctx modl hstream ^long WGS]
  Releaseable
  (release [_]
    (release modl))
  DatasetEngine
  (data-mean [this data-matrix]
    (in-context
     ctx
     (let [m (mrows data-matrix)
           n (ncols data-matrix)
           wgsn (min n WGS)
           wgsm (/ WGS wgsn)
           acc-size (max 1 (* m (blocks-count wgsn n)))]
       (with-release [acc (vctr data-matrix acc-size)
                      sum-reduction-kernel (function modl "sum_reduction_horizontal")
                      mean-kernel (function modl "mean_reduce")]
         (launch-reduce! hstream mean-kernel sum-reduction-kernel
                         [(buffer acc) (buffer data-matrix)] [(buffer acc)]
                         m n wgsm wgsn)
         (scal! (/ 1.0 n) (transfer acc))))))
  (data-variance [this data-matrix]
    (in-context
     ctx
     (let [m (mrows data-matrix)
           n (ncols data-matrix)
           wgsn (min n WGS)
           wgsm (/ WGS wgsn)
           acc-size (max 1 (* m (blocks-count wgsn n)))]
       (with-release [res (vctr data-matrix m)
                      cu-acc (create-data-source data-matrix acc-size)
                      sum-reduction-kernel (function modl "sum_reduction_horizontal")
                      mean-kernel (function modl "mean_reduce")
                      variance-kernel (function modl "variance_reduce")]
         (launch-reduce! hstream mean-kernel sum-reduction-kernel
                         [cu-acc (buffer data-matrix)] [cu-acc]
                         m n wgsm wgsn)
         (memcpy! cu-acc (buffer res) hstream)
         (scal! (/ 1.0 n) res)
         (launch-reduce! hstream variance-kernel sum-reduction-kernel
                         [cu-acc (buffer data-matrix) (buffer res)] [cu-acc]
                         m n wgsm wgsn)
         (memcpy! cu-acc (buffer res) hstream)
         (scal! (/ 1.0 n) (transfer res))))))
  (acor [_ data-matrix]
    (in-context
     ctx
     (let [n (ncols data-matrix)
           dim (mrows data-matrix)
           min-fac 4
           min-lag 4
           lag (max min-lag (min (quot n min-fac) WGS))
           win-mult 4;;TODO use this
           tau-max 2;;TODO (double (/ lag win-mult))
           wgsm (min 16 dim WGS)
           wgsn (long (/ WGS wgsm))
           wg-count (blocks-count wgsn n)
           native-fact (na/native-factory data-matrix)]
       (if (<= (* lag min-fac) n)
         (let-release [d (vctr native-fact dim)]
           (with-release [c0 (vctr native-fact dim)
                          cu-acc (create-data-source data-matrix (* dim wg-count))
                          mean-vec (vctr data-matrix dim)
                          d-acc (create-data-source data-matrix (* dim wg-count))
                          sum-reduction-kernel (function modl "sum_reduction_horizontal")
                          sum-reduce-kernel (function modl "sum_reduce_horizontal")
                          subtract-mean-kernel (function modl "subtract_mean")
                          autocovariance-kernel (function modl "autocovariance")
                          sum-reduce-params (make-parameters 4)
                          sum-reduction-params (make-parameters 3)]
             (set-parameter! sum-reduction-params 2 cu-acc)
             (set-parameters! sum-reduce-params 2 cu-acc (buffer data-matrix))
             (launch-reduce! hstream sum-reduce-kernel sum-reduction-kernel
                             sum-reduce-params sum-reduction-params dim n wgsm wgsn)
             (memcpy! cu-acc (buffer mean-vec) hstream)
             (scal! (/ 1.0 n) mean-vec)
             (launch! subtract-mean-kernel (grid-2d dim n wgsm wgsn) hstream
                      (cuda/parameters dim n (buffer data-matrix) (buffer mean-vec)))
             (memset! cu-acc 0 hstream)
             (memset! d-acc 0 hstream)
             (launch! autocovariance-kernel (grid-2d n dim WGS 1) hstream
                      (cuda/parameters n dim (int lag) cu-acc d-acc (buffer data-matrix)))
             (set-parameter! sum-reduce-params 3 cu-acc)
             (launch-reduce! hstream sum-reduce-kernel sum-reduction-kernel
                             sum-reduce-params sum-reduction-params dim wg-count wgsm wgsn)
             (memcpy-host! cu-acc (buffer c0) hstream)
             (set-parameter! sum-reduce-params 3 d-acc)
             (launch-reduce! hstream sum-reduce-kernel sum-reduction-kernel
                             sum-reduce-params sum-reduction-params dim wg-count wgsm wgsn)
             (memcpy-host! cu-acc (buffer d) hstream)
             (->Autocorrelation (div d c0) (transfer mean-vec)
                                (sqrt! (scal! (/ 1.0 (* (- n lag) n)) d)) n lag)))
         (throw (IllegalArgumentException.
                 (format (str "The autocorrelation time is too long relative to the variance. "
                              "Number of steps (%d) must not be less than %d.")
                         n (* lag min-fac))))))))
  EstimateEngine
  (histogram [this data-matrix]
    (in-context
     ctx
     (let [m (mrows data-matrix)
           n (ncols data-matrix)
           wgsn (min n WGS)
           wgsm (/ WGS wgsn)
           acc-size (* 2 (max 1 (* m (blocks-count wgsn n))))
           cuaccessor (data-accessor data-matrix)]
       (with-release [cu-min-max (create-data-source data-matrix acc-size)
                      uint-res (mem-alloc (* Integer/BYTES WGS m))
                      limits (ge data-matrix 2 m)
                      result (ge data-matrix WGS m)
                      bin-ranks (ge data-matrix WGS m)
                      min-max-reduction-kernel (function modl "min_max_reduction")
                      min-max-kernel (function modl "min_max_reduce")
                      histogram-kernel (function modl "histogram")
                      uint-to-real-kernel (function modl "uint_to_real")
                      local-sort-kernel (function modl "bitonic_local")]
         (launch-reduce! hstream min-max-kernel min-max-reduction-kernel
                         [cu-min-max (buffer data-matrix)] [cu-min-max]
                         m n wgsm wgsn)
         (memcpy! cu-min-max (buffer limits) hstream)
         (memset! uint-res 0 hstream)
         (launch! histogram-kernel (grid-2d m n 1 WGS) hstream
                  (cuda/parameters (int m) (int n) (buffer limits) (buffer data-matrix) uint-res))
         (launch! uint-to-real-kernel (grid-2d WGS m WGS 1) hstream
                  (cuda/parameters WGS m (cast-prim cuaccessor (/ WGS n)) (buffer limits)
                                   uint-res (buffer result)))
         (launch! local-sort-kernel (grid-1d (* m WGS) WGS) hstream
                  (cuda/parameters (* m WGS) (buffer result) (buffer bin-ranks)))
         (->Histogram (transfer limits) (transfer result) (transfer bin-ranks)))))))

;; ======================== MCMC engine ========================================

(deftype GTXStretch [ctx hstream neanderthal-factory cuaccessor dataset-eng
                     ^long walker-count wsize cu-model ^long DIM ^long WGS
                     ^ints move-counter ^ints move-bare-counter iteration-counter
                     ^ints move-seed
                     cu-params cu-xs cu-s0 cu-s1
                     cu-logfn-xs cu-logfn-s0 cu-logfn-s1
                     cu-accept cu-accept-acc cu-acc
                     stretch-move-odd-kernel stretch-move-odd-params
                     stretch-move-even-kernel stretch-move-even-params
                     stretch-move-odd-bare-kernel stretch-move-odd-bare-params
                     stretch-move-even-bare-kernel stretch-move-even-bare-params
                     init-walkers-kernel init-walkers-params
                     logfn-kernel logfn-params
                     sum-accept-reduction-kernel sum-accept-reduction-params
                     sum-accept-kernel sum-accept-params
                     sum-means-kernel
                     sum-reduction-kernel sum-reduce-kernel
                     min-max-reduction-kernel
                     min-max-kernel
                     histogram-kernel
                     uint-to-real-kernel
                     local-sort-kernel
                     mean-kernel mean-params
                     variance-kernel variance-params]
  Releaseable
  (release [_]
    (release cu-xs)
    (release cu-s0)
    (release cu-s1)
    (release cu-logfn-xs)
    (release cu-logfn-s0)
    (release cu-logfn-s1)
    (release cu-accept)
    (release cu-accept-acc)
    (release cu-acc)
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
  ModelProvider
  (model [this]
    cu-model)
  MCMCStretch
  (init-move! [this cu-means-acc a]
    (let [seed (int (+ 2 (aget move-seed 0)))]
      (aset move-seed 0 seed)
      (set-parameter! stretch-move-odd-params 1 seed)
      (set-parameter! stretch-move-even-params 1 (inc seed))
      (aset move-counter 0 0)
      (memset! cu-accept 1 hstream)
      (initialize cuaccessor cu-means-acc)
      (set-parameters! stretch-move-odd-params 8 cu-means-acc a)
      (set-parameters! stretch-move-even-params 8 cu-means-acc a))
    this)
  (move! [this]
    (set-parameter! stretch-move-odd-params 10 (aget move-counter 0))
    (launch! stretch-move-odd-kernel wsize hstream stretch-move-odd-params)
    (set-parameter! stretch-move-even-params 10 (aget move-counter 0))
    (launch! stretch-move-even-kernel wsize hstream stretch-move-even-params)
    (inc! move-counter)
    this)
  (move-bare! [this]
    (set-parameter! stretch-move-odd-bare-params 9 (aget move-bare-counter 0))
    (launch! stretch-move-odd-bare-kernel wsize hstream stretch-move-odd-bare-params)
    (set-parameter! stretch-move-even-bare-params 9 (aget move-bare-counter 0))
    (launch! stretch-move-even-bare-kernel wsize hstream stretch-move-even-bare-params)
    (inc! move-bare-counter)
    this)
  (set-temperature! [this t]
    (let [beta (cast-prim cuaccessor (/ 1.0 ^double t))]
      (set-parameter! stretch-move-odd-bare-params 8 beta)
      (set-parameter! stretch-move-even-bare-params 8 beta))
    this)
  (info [this]
    {:walker-count walker-count
     :iteration-counter @iteration-counter})
  RandomSampler
  (init! [this seed]
    (let [a (cast-prim cuaccessor 2.0)]
      (set-parameter! stretch-move-odd-bare-params 1 (int seed))
      (set-parameter! stretch-move-even-bare-params 1 (inc (int seed)))
      (set-parameter! stretch-move-odd-bare-params 7 a)
      (set-parameter! stretch-move-even-bare-params 7 a)
      (set-temperature! this 1.0)
      (aset move-bare-counter 0 0)
      (aset move-seed 0 (int seed))
      this))
  (sample [this]
    (sample this walker-count))
  (sample [this n]
    (in-context
     ctx
     (if (<= (long n) walker-count)
       (let-release [res (ge neanderthal-factory DIM n)]
         (memcpy! cu-xs (buffer res) hstream)
         res)
       (throw (IllegalArgumentException.
               (format "For number of samples greater than %d, use sample! method." walker-count)) ))))
  (sample! [this]
    (sample! this walker-count))
  (sample! [this n]
    (in-context
     ctx
     (let [available (* DIM (entry-width cuaccessor) walker-count)]
       (set-temperature! this 1.0)
       (let-release [res (ge neanderthal-factory DIM n)]
         (loop [ofst 0 requested (* DIM (entry-width cuaccessor) (long n))]
           (move-bare! this)
           (vswap! iteration-counter inc-long)
           (if (<= requested available)
             (memcpy! cu-xs (buffer res) 0 ofst requested hstream)
             (do
               (memcpy! cu-xs (buffer res) 0 ofst available hstream)
               (recur (+ ofst available) (- requested available)))))
         res))))
  MCMC
  (init-position! [this position]
    (in-context
     ctx
     (let [cu-position (.cu-xs ^GTXStretch position)]
       (memcpy! cu-position cu-xs hstream)
       (launch! logfn-kernel (grid-1d walker-count WGS) hstream logfn-params)
       (vreset! iteration-counter 0)
       this)))
  (init-position! [this seed limits]
    (in-context
     ctx
     (with-release [cu-limits (transfer neanderthal-factory limits)]
       (set-parameters! init-walkers-params 1 (int seed) (buffer cu-limits))
       (launch! init-walkers-kernel (grid-1d (* DIM (long (/ walker-count 4))) WGS) hstream
                init-walkers-params)
       (launch! logfn-kernel (grid-1d walker-count WGS) hstream logfn-params)
       (vreset! iteration-counter 0)
       this)))
  (burn-in! [this n a]
    (in-context
     ctx
     (let [a (cast-prim cuaccessor a)]
       (set-parameter! stretch-move-odd-bare-params 7 a)
       (set-parameter! stretch-move-even-bare-params 7 a)
       (set-temperature! this 1.0)
       (dotimes [i n]
         (move-bare! this))
       (vswap! iteration-counter add-long n)
       this)))
  (anneal! [this schedule n a]
    (in-context
     ctx
     (let [a (cast-prim cuaccessor a)]
       (set-parameter! stretch-move-odd-bare-params 7 a)
       (set-parameter! stretch-move-even-bare-params 7 a)
       (dotimes [i n]
         (set-temperature! this (schedule i))
         (move-bare! this))
       (vswap! iteration-counter add-long n)
       this)))
  (run-sampler! [this n a]
    (in-context
     ctx
     (let [a (cast-prim cuaccessor a)
           n (long n)
           means-count (long (blocks-count WGS (/ walker-count 2)))
           local-m (min means-count WGS)
           local-n (long (/ WGS local-m))
           acc-count (long (blocks-count local-m means-count))]
       (with-release [cu-means-acc (create-data-source cuaccessor (* DIM means-count n))
                      acc (ge neanderthal-factory DIM (* acc-count n))
                      means (submatrix acc 0 0 DIM n)]
         (init-move! this cu-means-acc a)
         (dotimes [i n]
           (move! this))
         (vswap! iteration-counter add-long n)
         (launch-reduce! hstream sum-means-kernel sum-reduction-kernel
                         [(buffer acc) cu-means-acc] [(buffer acc)]
                         (* DIM n) means-count local-n local-m)
         (scal! (/ 0.5 (* WGS means-count)) means)
         (launch-reduce! hstream sum-accept-kernel sum-accept-reduction-kernel
                         sum-accept-params sum-accept-reduction-params means-count WGS)
         {:acceptance-rate (/ (double (read-long hstream cu-accept-acc)) (* walker-count n))
          :a a
          :autocorrelation (acor dataset-eng means)}))))
  (acc-rate! [this a]
    (in-context
     ctx
     (let [a (cast-prim cuaccessor a)
           means-count (long (blocks-count WGS (/ walker-count 2)))]
       (with-release [cu-means-acc (create-data-source cuaccessor (* DIM means-count))]
         (init-move! this cu-means-acc a)
         (move! this)
         (vswap! iteration-counter inc-long)
         (launch-reduce! hstream sum-accept-kernel sum-accept-reduction-kernel ;;TODO
                         sum-accept-params sum-accept-reduction-params
                         (blocks-count WGS (/ walker-count 2)) WGS)
         (/ (double (read-long hstream cu-accept-acc)) walker-count)))))
  EstimateEngine
  (histogram [this]
    (histogram! this 1))
  (histogram! [this cycles]
    (in-context
     ctx
     (let [cycles (long cycles)
           n (* cycles walker-count)
           wgsm (min DIM (long (sqrt WGS)))
           wgsn (long (/ WGS wgsm))
           histogram-worksize (grid-2d DIM walker-count 1 WGS)
           acc-size (* 2 (max 1 (* DIM (blocks-count WGS walker-count))))]
       (with-release [cu-min-max (create-data-source cuaccessor acc-size)
                      uint-res (mem-alloc (* Integer/BYTES WGS DIM))
                      result (ge neanderthal-factory WGS DIM)
                      limits (ge neanderthal-factory 2 DIM)
                      bin-ranks (ge neanderthal-factory WGS DIM)]
         (let [histogram-params (cuda/parameters DIM walker-count (buffer limits) cu-xs uint-res)]
           (launch-reduce! hstream min-max-kernel min-max-reduction-kernel
                           [cu-min-max cu-xs] [cu-min-max] DIM walker-count wgsm wgsn)
           (memcpy! cu-min-max (buffer limits) hstream)
           (memset! uint-res 1 hstream)
           (launch! histogram-kernel histogram-worksize hstream histogram-params)
           (set-temperature! this 1.0)
           (dotimes [i (dec cycles)]
             (move-bare! this)
             (launch! histogram-kernel histogram-worksize hstream histogram-params))
           (vswap! iteration-counter add-long (dec cycles))
           (launch! uint-to-real-kernel (grid-2d WGS DIM WGS 1) hstream
                    (cuda/parameters WGS DIM (cast-prim cuaccessor (/ WGS n)) (buffer limits)
                                     uint-res (buffer result)))
           (launch! local-sort-kernel (grid-1d (* DIM WGS) WGS) hstream
                    (cuda/parameters (* DIM WGS) (buffer result) (buffer bin-ranks)))
           (->Histogram (transfer limits) (transfer result) (transfer bin-ranks)))))))
  Location
  (mean [_]
    (in-context
     ctx
     (let-release [res-vec (vctr (na/native-factory neanderthal-factory) DIM)]
       (launch-reduce! hstream mean-kernel sum-reduction-kernel
                       mean-params [cu-acc] DIM walker-count 1 WGS)
       (memcpy-host! cu-acc (buffer res-vec) hstream)
       (scal! (/ 1.0 walker-count) res-vec))))
  Spread
  (variance [_]
    (in-context
     ctx
     (let-release [res-vec (vctr neanderthal-factory DIM)]
       (launch-reduce! hstream mean-kernel sum-reduction-kernel mean-params [cu-acc]
                       DIM walker-count 1 WGS)
       (memcpy! cu-acc (buffer res-vec) hstream)
       (scal! (/ 1.0 walker-count) res-vec)
       (set-parameters! variance-params 4 (buffer res-vec))
       (launch-reduce! hstream variance-kernel sum-reduction-kernel variance-params [cu-acc]
                       DIM walker-count 1 WGS)
       (memcpy! cu-acc (buffer res-vec) hstream)
       (scal! (/ 1.0 walker-count) (transfer res-vec)))))
  (sd [this]
    (in-context
     ctx
     (sqrt! (variance this)))))

(deftype GTXStretchFactory [ctx modl hstream neanderthal-factory dataset-eng model ^long DIM ^long WGS]
  Releaseable
  (release [_]
    (release modl))
  MCMCFactory
  (mcmc-sampler [_ walker-count params]
    (in-context
     ctx
     (let [walker-count (int walker-count)]
       (if (and (<= (* 2 WGS) walker-count) (zero? (rem walker-count (* 2 WGS))))
         (let [acc-count (* DIM (blocks-count WGS walker-count))
               accept-count (blocks-count WGS (/ walker-count 2))
               accept-acc-count (blocks-count WGS accept-count)
               cuaccessor (data-accessor neanderthal-factory)
               sub-bytesize (* DIM (long (/ walker-count 2)) (entry-width cuaccessor))
               cu-params (buffer params)
               half-walker-count (int (/ walker-count 2))]
           (let-release [cu-xs (create-data-source cuaccessor (* DIM walker-count))
                         cu-s0 (mem-sub-region cu-xs 0 sub-bytesize)
                         cu-s1 (mem-sub-region cu-xs sub-bytesize sub-bytesize)
                         cu-logfn-xs (create-data-source cuaccessor (* DIM walker-count))
                         cu-logfn-s0 (mem-sub-region cu-logfn-xs 0 sub-bytesize)
                         cu-logfn-s1 (mem-sub-region cu-logfn-xs sub-bytesize sub-bytesize)
                         cu-accept (mem-alloc (* Integer/BYTES accept-count))
                         cu-accept-acc (mem-alloc (* Long/BYTES accept-acc-count))
                         cu-acc (create-data-source cuaccessor acc-count)]
             (->GTXStretch
              ctx hstream neanderthal-factory cuaccessor dataset-eng walker-count
              (grid-1d (/ walker-count 2) WGS) model DIM WGS
              (int-array 1) (int-array 1) (volatile! 0) (int-array 1)
              cu-params cu-xs cu-s0 cu-s1 cu-logfn-xs cu-logfn-s0 cu-logfn-s1
              cu-accept cu-accept-acc cu-acc
              (function modl "stretch_move_accu")
              (cuda/parameters half-walker-count 1 (int 1111) cu-params cu-s1 cu-s0 cu-logfn-s0 cu-accept 8 9 10)
              (function modl "stretch_move_accu") ;;TODO unnecessary. remove. kernels are stateless in CUDA.
              (cuda/parameters half-walker-count 1 (int 2222) cu-params cu-s0 cu-s1 cu-logfn-s1 cu-accept 8 9 10)
              (function modl "stretch_move_bare")
              (cuda/parameters half-walker-count 1 (int 3333) cu-params cu-s1 cu-s0 cu-logfn-s0 7 8 9)
              (function modl "stretch_move_bare")
              (cuda/parameters half-walker-count 1 (int 4444) cu-params cu-s0 cu-s1 cu-logfn-s1 7 8 9)
              (function modl "init_walkers")
              (cuda/parameters (int (/ (* DIM walker-count) 4)) 1 2 cu-xs)
              (function modl "logfn")
              (cuda/parameters walker-count cu-params cu-xs cu-logfn-xs)
              (function modl "sum_accept_reduction")
              (cuda/parameters walker-count cu-accept-acc)
              (function modl "sum_accept_reduce")
              (cuda/parameters walker-count cu-accept-acc cu-accept)
              (function modl "sum_means_vertical")
              (function modl "sum_reduction_horizontal")
              (function modl "sum_reduce_horizontal")
              (function modl "min_max_reduction")
              (function modl "min_max_reduce")
              (function modl "histogram")
              (function modl "uint_to_real")
              (function modl "bitonic_local")
              (function modl "mean_reduce")
              (cuda/parameters DIM walker-count cu-acc cu-xs)
              (function modl "variance_reduce")
              (cuda/parameters DIM walker-count cu-acc cu-xs 4))))
         (throw (IllegalArgumentException.
                 (format "Number of walkers (%d) must be a multiple of %d." walker-count (* 2 WGS)))))))))

;; ======================== Constructor functions ==============================

(defn ^:private dataset-options [wgs]
  ["-DREAL=float" "-DREAL2=float2" "-DACCUMULATOR=float" "-arch=compute_30" "-default-device"
   "-use_fast_math" (format "-DWGS=%d" wgs)])

(defn ^:private distribution-options [logpdf dim wgs cudart-version]
  ["-DREAL=float" "-DACCUMULATOR=float" "-arch=compute_30" "-default-device" "-use_fast_math"
   (format "-DLOGPDF=%s" logpdf) (format "-DDIM=%d" dim) (format "-DWGS=%d" wgs)
   (format "-DCUDART_VERSION=%s" cudart-version)])

(defn ^:private posterior-options [logpdf loglik dim wgs cudart-version]
  ["-DREAL=float" "-DACCUMULATOR=double" "-arch=compute_30" "-default-device" "-use_fast_math"
   (format "-DLOGPDF=%s" logpdf) (format "-DLOGLIK=%s" loglik) (format "-DDIM=%d" dim)
   (format "-DWGS=%d" wgs) (format "-DCUDART_VERSION=%s" cudart-version)])

(defn ^:private stretch-options [logfn dim wgs cudart-version]
  ["-DREAL=float" "-DREAL2=float2" "-DACCUMULATOR=float" "-arch=compute_30" "-default-device"
   "-use_fast_math" (format "-DLOGFN=%s" logfn) (format "-DDIM=%d" dim) (format "-DWGS=%d" wgs)
   (format "-DCUDART_VERSION=%s" cudart-version)])

(defn ^:private direct-sampler-options [wgs cudart-version]
  ["-DREAL=float" (format "-DWGS=%d" wgs) "-use_fast_math" "-default-device"
   (format "-DCUDART_VERSION=%s" cudart-version)])


(let [reduction-src (slurp (io/resource "uncomplicate/clojurecuda/kernels/reduction.cu"))
      estimate-src (slurp (io/resource "uncomplicate/bayadera/internal/cuda/engines/nvidia-gtx-estimate.cu"))
      distribution-src (slurp (io/resource "uncomplicate/bayadera/internal/cuda/engines/nvidia-gtx-distribution.cu"))
      likelihood-src (slurp (io/resource "uncomplicate/bayadera/internal/cuda/engines/nvidia-gtx-likelihood.cu"))
      uniform-sampler-src (slurp (io/resource "uncomplicate/bayadera/internal/cuda/rng/uniform-sampler.cu"))
      mcmc-stretch-src (slurp (io/resource "uncomplicate/bayadera/internal/cuda/engines/nvidia-gtx-mcmc-stretch.cu"))
      standard-headers {"stdint.h" (slurp (io/resource "uncomplicate/clojurecuda/include/jitify/stdint.h"))}
      philox-headers
      (merge standard-headers
             {"Random123/philox.h"
              (slurp (io/resource "uncomplicate/bayadera/internal/include/Random123/philox.h"))
              "features/compilerfeatures.h"
              (slurp (io/resource "uncomplicate/bayadera/internal/include/Random123/features/compilerfeatures.h"))
              "nvccfeatures.h"
              (slurp (io/resource "uncomplicate/bayadera/internal/include/Random123/features/nvccfeatures.h"))
              "array.h" (slurp (io/resource "uncomplicate/bayadera/internal/include/Random123/array.h"))})]

  (defn gtx-dataset-engine
    ([ctx hstream ^long WGS]
     (in-context
      ctx
      (with-release [prog (compile! (program (format "%s\n%s" reduction-src estimate-src)
                                             standard-headers)
                                    (dataset-options WGS))]
        (let-release [modl (module prog)]
          (->GTXDatasetEngine ctx modl hstream WGS)))))
    ([ctx hstream]
     (gtx-dataset-engine ctx hstream 1024)))

  (defn gtx-distribution-engine
    ([ctx hstream model WGS cudart-version]
     (in-context
      ctx
      (let [include-likelihood (satisfies? LikelihoodModel model)]
        (with-release [prog (compile! (program (format "%s\n%s" (apply str (source model)) distribution-src)
                                               philox-headers)
                                      (distribution-options (logpdf model) (dimension model) WGS cudart-version))]
          (let-release [modl (module prog)
                        logpdf-kernel (function modl "logpdf")
                        pdf-kernel (function modl "pdf")
                        evidence-kernel (if include-likelihood (function modl "evidence_reduce") nil)
                        sum-reduction-kernel (if include-likelihood (function modl "sum_reduction") nil)]
            (->GTXDistributionEngine ctx modl hstream WGS model
                                     logpdf-kernel pdf-kernel evidence-kernel sum-reduction-kernel)))))))

  (defn gtx-posterior-engine
    ([ctx hstream model WGS cudart-version]
     (in-context
      ctx
      (with-release [prog (compile! (program (format "%s\n%s\n%s\n%s"
                                                     (apply str (source model))
                                                     reduction-src likelihood-src distribution-src)
                                             philox-headers)
                                    (posterior-options (logpdf model) (loglik model) (dimension model)
                                                       WGS cudart-version))]
        (let-release [modl (module prog)
                      logpdf-kernel (function modl "logpdf")
                      pdf-kernel (function modl "pdf")
                      evidence-kernel (function modl "evidence_reduce")
                      sum-reduction-kernel (function modl "sum_reduction")]
          (->GTXDistributionEngine ctx modl hstream WGS model
                                   logpdf-kernel pdf-kernel evidence-kernel sum-reduction-kernel))))))

  (defn gtx-direct-sampler
    ([ctx hstream model WGS cudart-version]
     (in-context
      ctx
      (with-release [prog (compile! (program (format "%s\n%s"
                                                     (apply str (source model))
                                                     (apply str (sampler-source model)))
                                             philox-headers)
                                    (direct-sampler-options WGS cudart-version))]
        (let-release [modl (module prog)]
          (->GTXDirectSampler ctx modl hstream (dimension model) WGS))))))

  (defn gtx-stretch-factory [ctx hstream neanderthal-factory dataset-eng model WGS cudart-version]
    (in-context
     ctx
     (with-release [prog (compile! (program (format "%s\n%s\n%s\n%s\n%s"
                                                    uniform-sampler-src reduction-src
                                                    (apply str (source model))
                                                    estimate-src mcmc-stretch-src)
                                            philox-headers)
                                   (stretch-options (mcmc-logpdf model) (dimension model) WGS cudart-version))]
       (let-release [modl (module prog)]
         (->GTXStretchFactory ctx modl hstream neanderthal-factory dataset-eng model (dimension model) WGS))))))

;; =========================== Bayadera factory  ===========================

(defn ^:private release-deref [ds]
  (if (sequential? ds)
    (doseq [d ds]
      (when (realized? d) (release @d)))
    (when (realized? ds) (release ds))))

(defrecord GTXBayaderaFactory [ctx hstream ^long compute-units ^long WGS cudart-version
                               dataset-eng neanderthal-factory distribution-engines
                               direct-samplers mcmc-factories]
  Releaseable
  (release [_]
    (in-context
     ctx
     (release dataset-eng)
     (release neanderthal-factory)
     (release-deref (vals distribution-engines))
     (release-deref (vals direct-samplers))
     (release-deref (vals mcmc-factories))
     true))
  na/MemoryContext
  (compatible? [_ o]
    (or (satisfies? CLModel o) (na/compatible? neanderthal-factory o)))
  DistributionEngineFactory
  (distribution-engine [_ model]
    (if-let [eng (distribution-engines model)]
      @eng
      (gtx-distribution-engine ctx hstream model WGS)))
  (posterior-engine [_ model]
    (gtx-posterior-engine ctx hstream model WGS cudart-version))
  SamplerFactory
  (direct-sampler [_ id]
    (deref (direct-samplers id)))
  (mcmc-factory [_ model]
    (if-let [factory (mcmc-factories model)]
      @factory
      (gtx-stretch-factory ctx hstream neanderthal-factory dataset-eng model WGS cudart-version)))
  (processing-elements [_]
    (* compute-units WGS))
  DatasetFactory
  (dataset-engine [_]
    dataset-eng)
  na/FactoryProvider
  (factory [_]
    neanderthal-factory)
  (native-factory [_]
    (na/native-factory neanderthal-factory)))

(defn gtx-bayadera-factory
  ([distributions samplers ctx hstream compute-units WGS]
   (in-context
    ctx
    (let [cudart-version (driver-version)]
      (let-release [neanderthal-factory (cuda-float ctx hstream)
                    dataset-eng (gtx-dataset-engine ctx hstream WGS)]
        (->GTXBayaderaFactory
         ctx hstream compute-units WGS cudart-version dataset-eng neanderthal-factory
         (fmap #(delay (gtx-distribution-engine ctx hstream % WGS cudart-version)) distributions)
         (fmap #(delay (gtx-direct-sampler ctx hstream % WGS cudart-version))
               (select-keys distributions (keys samplers)))
         (fmap #(delay (gtx-stretch-factory ctx hstream neanderthal-factory dataset-eng % WGS cudart-version))
               distributions))))))
  ([distributions samplers ctx hstream]
   (in-context
    ctx
    (let [dev (ctx-device)]
      (gtx-bayadera-factory distributions samplers ctx hstream
                            (multiprocessor-count dev) (max-block-dim-x dev))))))
