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
      (let [x (Math/floor (map-range (* (double i) offset)))]
        (.line g x 0 x height)))
    (.endDraw g)
    g))

(defn labels
  ([^Axis axis ^double ofst nf ^PGraphics g]
   (let [height (.height g)
         cnt (partitions axis ofst)
         left-padding (/ (.textWidth g ^String (nf (.lower axis))) 2.0)
         right-padding (/ (.textWidth g ^String (nf (.upper axis))) 2.0)
         map-range ^IFn$DD (axis-mapper axis left-padding (- (.width g) right-padding))]
     (.beginDraw g)
     (.textAlign g PConstants/CENTER)
     (dotimes [i cnt]
       (.text g ^String (nf (* (double i) ofst)) (float (map-range (* (double i) ofst))) (float height)))
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


#_(defn labels [^PGraphics g padding x-lower x-upper x-offset y-lower y-upper y-offset]
  (let [padding (long padding)
        width (- (.width g) padding)
        height (- (.height g) padding)
        x-lower (float x-lower)
        x-upper (float x-upper)
        y-lower (float y-lower)
        y-upper (float y-upper)
        x-offset (float x-offset)
        y-offset (float y-offset)
        map-x (range-mapper x-lower x-upper padding width)]
    (.beginDraw g)
    (loop [x x-lower]
      (when (<= x x-upper)
        (do (.text g x (float (map-x x)) (float (- height padding)))
            (recur (+ x x-offset)))))
    (.endDraw g)
    g))

(defn points [^PGraphics g xs ys x-lower x-upper y-lower y-upper]
  (let [map-x (range-mapper x-lower x-upper 0 (.width g))
        map-y (range-mapper y-lower y-upper (.height g) 0)]
    (.beginDraw g)
    (.clear g)
    (dotimes [i (dim xs)]
      (.point g (map-x (entry xs i)) (map-y (entry ys i))))
    (.endDraw g)
    g))

#_(defn circles [^PGraphics g xs ys x-lower x-upper y-lower y-upper]
  (let [map-x (range-mapper x-lower x-upper 0 (.width g))
        map-y (range-mapper y-lower y-upper (.height g) 0)]
    (.beginDraw g)
    (.clear g)
    (dotimes [i (dim xs)]
      (.ellipse g (map-range (entry xs i)) (map-range (entry ys i)) 2 2))
    (.endDraw g)
    g))

#_(defrecord Chart2D [^Axis x-axis ^Axis y-axis
                    ^long left-margin ^long right-margin
                    ^long top-margin ^long bottom-margin
                    ^long left-padding ^long right-padding
                    ^long top-padding ^long bottom-padding]
  Chart
  (frame [this ^PGraphics g]
    (do
      (.beginDraw g)
      (.rectMode g PConstants/CORNER)
      (.noFill g)
      (.rect g left-margin top-margin
             (- (.width g) 1 right-margin)
             (- (.height g) 1 bottom-margin))
      (.endDraw g)
      g))
  (x-grid [this ^PGraphics g]
    (let [x-max (- (.width g) 1 right-margin)
          y-max (- (.height g) 1 bottom-margin)
          map-x (axis-mapper x-axis (+ left-margin left-padding) (- x-max right-padding))]
      (.beginDraw g)
      (dotimes [i (.short-ticks x-axis)]
        (let [x (map-x (* i (offset x-axis)))]
          (.line g x top-margin x y-max)))
      (.endDraw g)
      g))
  (y-grid [this ^PGraphics g]
    (let [x-max (- (.width g) 1 right-margin)
          y-max (- (.height g) 1 bottom-margin)
          map-y (axis-mapper y-axis (- y-max bottom-padding) (+ top-margin top-padding))]
      (.beginDraw g)
      (dotimes [i (.short-ticks y-axis)]
        (let [y (map-y (* i (offset y-axis)))]
          (.line g left-margin y x-max y)))
      (.endDraw g)
      g))

  )
