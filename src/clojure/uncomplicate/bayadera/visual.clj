(ns uncomplicate.bayadera.visual
  (:require [uncomplicate.clojurecl.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [dim]]
             [real :refer [entry]]
             [native :refer [dv]]]
            [quil.core :as q])
  (:import [processing.core PGraphics PConstants]))

(defrecord HSBColor [^float h ^float s ^float b])

(defn range-mapper
  ([^double start1 ^double end1 ^double start2 ^double end2]
   (fn ^double [^double value]
     (+ start2 (* (- end2 start2) (/ (- value start1) (- end1 start1))))))
  ([^double start1 ^double end1]
   (fn ^double [^double value ^double start2 ^double end2]
     (+ start2 (* (- end2 start2) (/ (- value start1) (- end1 start1)))))))

(defrecord Frame2D [^long margin ^long padding
                    ^long tick-length
                    ^double x-lower ^double x-upper
                    ^long x-offset ^long y-offset]
  )

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

(defn grid [^PGraphics g ^long padding ^long x-offset ^long y-offset]
  (let [x-max (dec (.width g))
        y-max (dec (.height g))]
    (do
      (.beginDraw g)
      (dotimes [i (inc (int (/ (- x-max padding) x-offset)))]
        (.line g (+ padding (* i x-offset)) y-max
               (+ padding (* i x-offset)) 0))
      (dotimes [i (inc (int (/ (- y-max padding) y-offset)))]
        (.line g 0 (- y-max padding (* i y-offset))
               x-max (- y-max padding (* i y-offset))))
      (.endDraw g)
      g)))

(defn frame [^PGraphics g ^long margin]
  (do
    (.beginDraw g)
    (.rectMode g PConstants/CORNER)
    (.noFill g)
    (.rect g margin margin
           (- (dec (.width g)) (* 2 margin)) (- (dec (.height g)) (* 2 margin)))
    (.endDraw g)
    g))

(defn ticks [^PGraphics g margin padding tick-length x-offset y-offset]
  (let [x-max (dec (.width g))
        y-max (dec (.height g))
        margin (int margin)
        gap (+ margin (int padding))
        tick-length (int tick-length)
        x-offset (int x-offset)
        y-offset (int y-offset)]
    (do
      (.beginDraw g)
      (dotimes [i (inc (int (/ (- x-max gap margin) x-offset)))]
        (.line g (+ gap (* i x-offset)) (- y-max margin)
               (+ gap (* i x-offset)) (- y-max (- margin tick-length))))
      (dotimes [i (inc (int (/ (- y-max gap margin) y-offset)))]
        (.line g margin (- y-max gap (* i y-offset))
               (- margin tick-length) (- y-max gap (* i y-offset))))
      (.endDraw g)
      g)))

(defn labels [^PGraphics g padding x-lower x-upper x-offset y-lower y-upper y-offset]
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

(defn circles [^PGraphics g xs ys x-lower x-upper y-lower y-upper]
  (let [map-x (range-mapper x-lower x-upper 0 (.width g))
        map-y (range-mapper y-lower y-upper (.height g) 0)]
    (.beginDraw g)
    (.clear g)
    (dotimes [i (dim xs)]
      (.ellipse g (map-range (entry xs i)) (map-range (entry ys i)) 2 2))
    (.endDraw g)
    g))
