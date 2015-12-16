(ns uncomplicate.bayadera.visual
  (:require [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal.core :refer [dim entry]]
            [quil.core :as q])
  (:import [processing.core PGraphics PConstants PApplet]))

(defrecord HSBColor [^float h ^float s ^float b])

(defn grid-style [^PGraphics g ^HSBColor color]
  (doto g
    (.beginDraw)
    (.clear)
    (.colorMode PConstants/HSB 360 100 100)
    (.stroke (.h color) (.s color) (.b color))
    (.endDraw))
  g)

(defn grid [^PGraphics g ^long cell-width ^long cell-height]
  (let [width (.width g)
        height (.height g)
        x-count (int (/ width cell-width))
        y-count (int (/ height cell-height))]
    (do
      (.beginDraw g)
      (loop [x 0]
        (if (< x width)
          (do (.line g x height x 0)
              (recur (+ x cell-width)))))
      (loop [y (dec height)]
        (if (<= 0 y)
          (do (.line g 0 y width y)
              (recur (- y cell-height)))))
      (.endDraw g)
      g)))

(defn x-axis [^PGraphics g ^long mark-height ^long cell-width]
  (let [width (.width g)
        x-count (int (/ width cell-width))]
    (do
      (.beginDraw g)
      (.line g 0 0 width 0)
      (loop [x 0]
        (if (< x width)
          (do (.line g x 0 x mark-height)
              (recur (+ x cell-width)))))
      (.endDraw g)
      g)))

(defn y-axis [^PGraphics g ^long mark-width ^long cell-height]
  (let [height (.height g)
        width (dec (.width g))
        mark-start (- width mark-width)
        y-count (int (/ height cell-height))]
    (do
      (.beginDraw g)
      (.line g width 0 width height)
      (loop [y (dec height)]
        (if (<= 0 y)
          (do (.line g mark-start y width y)
              (recur (- y cell-height)))))
      (.endDraw g)
      g)))

(defn points [^PGraphics g xs ys x-lower x-upper y-lower y-upper]
  (let [height (.height g)
        width (.width g)
        x-lower (float x-lower)
        x-upper (float y-upper)
        y-lower (float y-lower)
        y-upper (float y-upper)]
    (.beginDraw g)
    (.clear g)
    (dotimes [i (dim xs)]
      (.point g
              (PApplet/map (entry xs i) x-lower x-upper 0 width)
              (PApplet/map (entry ys i) y-lower y-upper height 0)))
    (.endDraw g)))
