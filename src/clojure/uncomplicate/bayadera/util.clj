(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.util
  (:require [uncomplicate.commons.core :refer [release with-release let-release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [protocols :as np]
             [math :refer [sqrt]]
             [core :refer [transfer dim]]
             [real :refer [entry]]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [impl :refer :all]
             [math :refer [log-beta]]])
  (:import [clojure.lang IFn$DD IFn$DDDD IFn$LD]))

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

(defn hdi-rank-index
  ([bin-rank pdf]
   (hdi-rank 0.95 bin-rank pdf))
  ([^double mass bin-rank pdf]
   (let [n (dim bin-rank)
         density (* mass n)]
     (loop [i 0 acc 0.0]
       (if (and (< i n) (< acc density))
         (recur (inc i) (+ acc (entry pdf (entry bin-rank i))))
         (dec i))))))
