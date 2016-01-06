(ns uncomplicate.bayadera.visual
  (:require [uncomplicate.clojurecl.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [dim]]
             [real :refer [entry]]
             [native :refer [dv]]]
            [quil.core :as q])
  (:import [processing.core PGraphics PConstants PApplet]
           [clojure.lang IFn$DDDD IFn$DD]))

(defrecord HSBColor [^float h ^float s ^float b])

(defn range-mapper
  (^IFn$DD [^double start1 ^double end1 ^double start2 ^double end2]
   (fn ^double [^double value]
     (+ start2 (* (- end2 start2) (/ (- value start1) (- end1 start1))))))
  (^IFn$DDDD [^double start1 ^double end1]
   (fn ^double [^double value ^double start2 ^double end2]
     (+ start2 (* (- end2 start2) (/ (- value start1) (- end1 start1)))))))

(defrecord Axis [^double lower ^double upper])

(defn axis
  ([^double lower ^double upper]
   (->Axis lower upper)))

(defn axis-mapper
  (^IFn$DD [^Axis axis ^double start ^double end]
   (range-mapper (.lower axis) (.upper axis) start end))
  (^IFn$DDDD [^Axis axis]
   (range-mapper (.lower axis) (.upper axis))))

(defn offset ^double [^Axis axis ^long partitions]
  (/ (- (.upper axis) (.lower axis)) (double partitions)))

(defn partitions ^long [^Axis axis ^double offset]
  (inc (long (/ (- (.upper axis) (.lower axis)) offset))))

(defrecord Style [^HSBColor color ^float weight])

(defn frame [^PGraphics g]
  (do
    (.beginDraw g)
    (.rectMode g PConstants/CORNER)
    (.noFill g)
    (.rect g 0 0 (dec (.width g)) (dec (.height g)))
    (.endDraw g)
    g))

(defn bars [^Axis axis ^double offset ^PGraphics g]
  (let [height (.height g)
        map-range (axis-mapper axis 0 (dec (.width g)))
        cnt (partitions axis offset)]
    (.beginDraw g)
    (dotimes [i cnt]
      (let [x (Math/floor (map-range (+ (.lower axis ) (* (double i) offset))))]
        (.line g x 0 x height)))
    (.endDraw g)
    g))

(defn labels
  ([^Axis axis ^double ofst nf ^PGraphics g]
   (let [height (float (.height g))
         cnt (partitions axis ofst)
         left-padding (/ (.textWidth g ^String (nf (.lower axis))) 2.0)
         right-padding (/ (.textWidth g ^String (nf (.upper axis))) 2.0)
         map-range (axis-mapper axis left-padding (- (.width g) right-padding))]
     (.beginDraw g)
     (.textAlign g PConstants/CENTER)
     (dotimes [i cnt]
       (let [value (+ (.lower axis) (* (double i) ofst))]
         (.text g ^String (nf value) (float (map-range value)) height)))
     (.endDraw g)
     g))
  ([^Axis axis ^double ofst ^PGraphics g]
   (labels axis ofst str g)))

(defn style
  ([^PGraphics g ^HSBColor color ^long weight]
   (doto g
     (.beginDraw)
     (.clear)
     (.colorMode PConstants/HSB 360 100 100)
     (.strokeWeight weight)
     (.stroke (.h color) (.s color) (.b color))
     (.endDraw))
   g)
  ([g color]
   (style g color 1)))

(defn points [^PGraphics g ^Axis x-axis ^Axis y-axis xs ys]
  (let [map-x (axis-mapper x-axis 0 (dec (.width g)))
        map-y (axis-mapper y-axis (dec (.height g)) 0)]
    (.beginDraw g)
    (.clear g)
    (dotimes [i (dim xs)]
      (.point g (map-x (entry xs i)) (map-y (entry ys i))))
    (.endDraw g)
    g))
