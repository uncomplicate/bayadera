(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.toolbox.scaling
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [dim imin imax]]
             [real :refer [entry]]
             [math :refer [magnitude floor ceil]]]
            [uncomplicate.bayadera.util :refer [range-mapper]])
  (:import [clojure.lang IFn$DDDD IFn$DD IFn$LD]))

;;================== Finding nice axis scaling =============================

(defn nice ^double [^double x]
  (let [mag (magnitude x)
        fraction (/ x mag)]
    (* (double (cond
                 (<= fraction 1.0) 1.0
                 (<= fraction 2.0) 2.0
                 (<= fraction 5.0) 5.0
                 :default 10.0))
       mag)))

(defn nice-round ^double [^double x]
  (let [mag (magnitude x)
        fraction (/ x mag)]
    (* (double (cond
                 (< fraction 1.5) 1.0
                 (< fraction 3.0) 2.0
                 (< fraction 7.0) 5.0
                 :default 10.0))
       mag)))

(defn nice-spacing
  (^double [^double lower ^double upper ^double max-ticks]
   (nice-round (/ (nice (- upper lower)) (dec max-ticks))))
  (^double [^double lower ^double upper]
   (nice-spacing lower upper 10.0)))

(defn nice-limit ^double [^IFn$DD limit-fn ^double spacing ^double x]
  (* (.invokePrim limit-fn (/ x spacing)) spacing))

;; ====================== Axis ===========================================

(defrecord Axis [^double lower ^double upper ^double spacing])

(defn lower ^double [^Axis axis]
  (.lower axis))

(defn upper ^double [^Axis axis]
  (.upper axis))

(defn spacing ^double [^Axis axis]
  (.spacing axis))

(defn axis
  (^Axis [^double lower ^double upper]
   (axis lower upper 5.0))
  (^Axis [^double lower ^double upper ^double max-ticks]
   (let [spacing (nice-spacing lower upper max-ticks)]
     (->Axis (nice-limit floor spacing lower)
             (nice-limit ceil spacing upper)
             spacing))))

(defn vector-axis
  (^Axis [x ^double max-ticks]
   (axis (entry x (imin x)) (entry x (imax x)) max-ticks))
  (^Axis [x]
   (vector-axis x 5.0)))

(defn axis-mapper
  (^IFn$DD [^Axis axis ^double start ^double end]
   (range-mapper (.lower axis) (.upper axis) start end))
  (^IFn$DDDD [^Axis axis]
   (range-mapper (.lower axis) (.upper axis))))
