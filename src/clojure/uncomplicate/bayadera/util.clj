;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
     uncomplicate.bayadera.util
  (:require [uncomplicate.commons.core :refer [release with-release let-release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [math :refer [sqrt]]
             [core :refer [transfer dim subvector col ge copy]]
             [real :refer [entry entry! asum]]
             [auxil :refer [sort+!]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.internal.api :as na]
            [uncomplicate.bayadera.internal.protocols :as pr])
  (:import [java.security SecureRandom]
           [java.nio ByteBuffer]
           [java.util Arrays]
           [clojure.lang IFn$DD IFn$DDDD IFn$LD]
           uncomplicate.bayadera.internal.protocols.Histogram))

(let [random (SecureRandom.)]
  (defn srand-buffer [^long n]
    (let [b (ByteBuffer/allocate n)]
      (.nextBytes random (.array b))
      b)))

(defn range-mapper
  (^IFn$DD [^double start1 ^double end1 ^double start2 ^double end2]
   (fn ^double [^double value]
     (+ start2 (* (- end2 start2) (/ (- value start1) (- end1 start1))))))
  (^IFn$DDDD [^double start1 ^double end1]
   (fn ^double [^double value ^double start2 ^double end2]
     (+ start2 (* (- end2 start2) (/ (- value start1) (- end1 start1)))))))

(defn bin-mapper
  (^IFn$LD [^long bin-count ^double lower ^double upper]
   (bin-mapper bin-count 0.5 lower upper))
  (^IFn$LD [^long bin-count ^double offset ^double lower ^double upper]
   (let [bin-width (/ (- upper lower) bin-count)]
     (fn
       (^double [^long i]
        (+ lower (* bin-width (+ i offset))))
       (^long []
        bin-count)))))

(defn hdi-rank-count
  "Counts the smallest number of entries in `pdf` whose (scaled) mass is larger than mass.

  `bin-rank` contains the information about the decreasing order of entries in pdf by mass.
  This function does not explicitly check bin-rank and pdf for compatibility or errors."
  (^long [bin-rank pdf]
   (hdi-rank-count 0.95 bin-rank pdf))
  (^long [^double mass bin-rank pdf]
   (let [n (dim bin-rank)
         density (* mass (asum pdf))]
     (loop [i 0 acc 0.0]
       (if (and (< i n) (< acc density))
         (recur (inc i) (+ acc (entry pdf (entry bin-rank i))))
         i)))))

(defn hdi-bins
  "Groups the ranked bins from `bin-rank` into distinct regions.

  Returns a vector `[start0 end0 start1 end1 ...]`.
  This function does not explicitly check its parameters for sanity."
  [bin-rank ^long hdi-cnt]
  (with-release [hdi-vector (sort+! (copy (subvector bin-rank 0 hdi-cnt)))]
    (loop [i 1
           last-bin (entry hdi-vector 0)
           regions (transient [last-bin])]
      (if (< i hdi-cnt)
        (let [bin (entry hdi-vector i)]
          (recur (inc i) bin
                 (if (< 1.5 (- bin last-bin))
                   (conj! (conj! regions last-bin) bin)
                   regions)))
        (persistent! (conj! regions (entry hdi-vector (dec hdi-cnt))))))))

(defn hdi-regions
  "Creates a ge matrix `2 x number-of-distinct-regions` that contains regions within limits to
  which the first `hdi-cnt` binns from `bin-rank` belong.

  See also [[hdi-bins]]."
  [limits bin-rank ^long hdi-cnt]
  (let [lower (entry limits 0)
        upper (entry limits 1)
        bin-width (/ (- upper lower) (dim bin-rank))
        hdi-vector (hdi-bins bin-rank hdi-cnt)
        cnt (long (/ (count hdi-vector) 2))
        regions (ge (na/factory bin-rank) 2 cnt)]
    (dotimes [i cnt]
      (entry! regions 0 i (+ lower (* bin-width (double (hdi-vector (* 2 i))))))
      (entry! regions 1 i (+ lower (* bin-width (inc (double (hdi-vector (inc (* 2 i)))))))))
    regions))

(defn hdi
  "Creates hdi regions that contain `mass` density for the particular `index` variable in `histogram`."
  ([^Histogram histogram ^double mass ^long index]
   (let [limits (col (.limits histogram) index)
         bin-rank (col (.bin-ranks histogram) index)
         pdf (col (.pdf histogram) index)]
     (hdi-regions limits bin-rank (hdi-rank-count mass bin-rank pdf))))
  ([histogram ^long index]
   (hdi histogram 0.95 index)))
