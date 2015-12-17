(ns uncomplicate.bayadera.visual
  (:require [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal.core :refer [dim entry]]
            [quil.core :as q])
  (:import [processing.core PGraphics PConstants PApplet]))

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

(defn points [^PGraphics g xs ys x-lower x-upper y-lower y-upper]
  (let [x-max (.width g)
        y-max (.height g)
        x-lower (float x-lower)
        x-upper (float x-upper)
        y-lower (float y-lower)
        y-upper (float y-upper)]
    (.beginDraw g)
    (.clear g)
    (dotimes [i (dim xs)]
      (.point g
              (PApplet/map (entry xs i) x-lower x-upper 0 x-max)
              (PApplet/map (entry ys i) y-lower y-upper y-max 0)))
    (.endDraw g)))

(defn circles [^PGraphics g xs ys x-lower x-upper y-lower y-upper]
  (let [x-max (.width g)
        y-max (.height g)
        x-lower (float x-lower)
        x-upper (float x-upper)
        y-lower (float y-lower)
        y-upper (float y-upper)]
    (.beginDraw g)
    (.clear g)
    (dotimes [i (dim xs)]
      (.ellipse g
              (PApplet/map (entry xs i) x-lower x-upper 0 x-max)
              (PApplet/map (entry ys i) y-lower y-upper y-max 0)
              2 2))
    (.endDraw g)))
