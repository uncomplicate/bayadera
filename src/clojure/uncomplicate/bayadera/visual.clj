(ns uncomplicate.bayadera.visual
  (:require [uncomplicate.clojurecl.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [dim]]
             [real :refer [entry]]
             [native :refer [dv]]]
            [quil.core :as q])
  (:import [processing.core PGraphics PConstants]))

(defn map-range ^double [^double value range-vector]
  (let [start1 (entry range-vector 0)
        start2 (entry range-vector 2)]
    (+ start2
       (* (- (entry range-vector 3) start2)
          (/ (- value start1)
             (- (entry range-vector 1) start1))))))

(defrecord HSBColor [^float h ^float s ^float b])

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
        y-offset (float y-offset)]
    (with-release [x-range (dv x-lower x-upper padding width)
                   y-range (dv y-lower y-upper (- height) padding)]
      (.beginDraw g)
      (loop [x x-lower]
        (when (<= x x-upper)
          (do (.text g x (float (map-range x x-range)) (float (- height padding)))
              (recur (+ x x-offset)))))
      (.endDraw g)
      g)))



(defn points [^PGraphics g xs ys x-lower x-upper y-lower y-upper]
  (with-release [x-range (dv x-lower x-upper 0 (.width g))
                 y-range (dv y-lower y-upper (.height g) 0)]
    (.beginDraw g)
    (.clear g)
    (dotimes [i (dim xs)]
      (.point g (map-range (entry xs i) x-range) (map-range (entry ys i) y-range)))
    (.endDraw g)
    g))

(defn circles [^PGraphics g xs ys x-lower x-upper y-lower y-upper]
  (with-release [x-range (dv x-lower x-upper 0 (.width g))
                 y-range (dv y-lower y-upper (.height g) 0)]
    (.beginDraw g)
    (.clear g)
    (dotimes [i (dim xs)]
      (.ellipse g
                (map-range (entry xs i) x-range)
                (map-range (entry ys i) y-range)
                2 2))
    (.endDraw g)
    g))
