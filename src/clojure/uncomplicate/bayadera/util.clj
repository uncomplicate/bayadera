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
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [transfer dim subvector col]]
             [real :refer [entry entry! asum]]
             [native :refer [sge]]
             [block :refer [buffer]]])
  (:import [java.security SecureRandom]
           [java.nio ByteBuffer]
           [java.util Arrays]
           [clojure.lang IFn$DD IFn$DDDD IFn$LD]
           [uncomplicate.bayadera.protocols Histogram]))

(let [random (SecureRandom.)]
  (defn srand-buffer [^long n]
    (let [b (ByteBuffer/allocate n)]
      (.nextBytes random (.array b))
      b)))

(defn srand-int []
  (.getInt ^ByteBuffer (srand-buffer 4) 0))

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
  ([bin-rank pdf]
   (hdi-rank-count 0.95 bin-rank pdf))
  ([^double mass bin-rank pdf]
   (let [n (dim bin-rank)
         density (* mass (asum pdf))]
     (loop [i 0 acc 0.0]
       (if (and (< i n) (< acc density))
         (recur (inc i) (+ acc (entry pdf (entry bin-rank i))))
         i)))))

(defn hdi-bins
  [bin-rank ^long hdi-cnt]
  (let [hdi-array (float-array hdi-cnt)]
    (.get (.asFloatBuffer ^ByteBuffer (buffer (subvector bin-rank 0 hdi-cnt))) hdi-array)
    (Arrays/sort hdi-array)
    (loop [i 1
           last-bin (aget hdi-array 0)
           regions (transient [last-bin])]
      (if (< i hdi-cnt)
        (let [bin (aget hdi-array i)]
          (recur (inc i) bin
                 (if (< 1.0 (- bin last-bin))
                   (conj! (conj! regions last-bin) bin)
                   regions)))
        (persistent! (conj! regions (aget hdi-array (dec hdi-cnt))))))))

(defn hdi-regions [limits bin-rank ^long hdi-cnt]
  (let [lower (entry limits 0)
        upper (entry limits 1)
        bin-width (/ (- upper lower) (dim bin-rank))
        hdi-vector (hdi-bins bin-rank hdi-cnt)
        cnt (long (/ (count hdi-vector) 2))
        regions (sge 2 cnt)]
    (dotimes [i cnt]
      (entry! regions 0 i (+ lower (* bin-width (double (hdi-vector (* 2 i))))))
      (entry! regions 1 i (+ lower (* bin-width (inc (double (hdi-vector (inc (* 2 i)))))))))
    regions))

(defn hdi
  ([^Histogram histogram ^double mass ^long index]
   (let [limits (col (.limits histogram) index)
         bin-rank (col (.bin-ranks histogram) index)
         pdf (col (.pdf histogram) index)]
     (hdi-regions limits bin-rank (hdi-rank-count mass bin-rank pdf))))
  ([histogram ^long index]
   (hdi histogram 0.95 index)))
