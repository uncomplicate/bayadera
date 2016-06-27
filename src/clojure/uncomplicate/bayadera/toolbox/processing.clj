(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.toolbox.processing
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [imin imax]]
             [real :refer [entry]]]
            [uncomplicate.bayadera.util :refer [range-mapper]]
            [uncomplicate.bayadera.toolbox
             [theme :refer [cyberpunk-theme]]
             [scaling :refer [axis lower upper spacing axis-mapper]]]
            [quil.core :as q]
            [quil.applet :refer [resolve-renderer]])
  (:import [processing.core PGraphics PConstants PApplet]
           [uncomplicate.bayadera.toolbox.theme RGBColor Colormap Style Theme]))

(defn frame! [^PGraphics g]
  (do
    (.beginDraw g)
    (.rectMode g PConstants/CORNER)
    (.noFill g)
    (.rect g 0 0 (dec (.width g)) (dec (.height g)))
    (.endDraw g)
    g))

(defn bars! [^PGraphics g axis ^double density]
  (let [height (.height g)
        ofst (/ (spacing axis) density)
        upper (upper axis)
        map-range (axis-mapper axis 0 (dec (.width g)))]
    (.beginDraw g)
    (loop [value (lower axis)]
      (if (<= value upper)
        (let [x (Math/floor (map-range value))]
          (.line g x 0 x height)
          (recur (+ value ofst)))))
    (.endDraw g)
    g))

(defn format-number [^double n]
  (PApplet/nf n))

(defn labels!
  ([^PGraphics g axis nf ^long density]
   (let [height (float (.height g))
         ofst (* (spacing axis) density)
         upper (upper axis)
         left-padding (/ (.textWidth g ^String (nf (lower axis))) 2.0)
         right-padding (/ (.textWidth g ^String (nf upper)) 2.0)
         map-range (axis-mapper axis left-padding (- (.width g) right-padding))]
     (.beginDraw g)
     (.textAlign g PConstants/CENTER)
     (loop [value (lower axis)]
       (if (<= value upper)
         (let [x (Math/floor (map-range value))]
           (.text g ^String (nf value)
                  (float (map-range value)) (- height (* 2 (.textDescent g))))
           (recur (+ value ofst)))))
     (.endDraw g)
     g))
  ([^PGraphics g axis ^double density]
   (labels! g axis format-number density)))

(defn style! [^PGraphics g ^Style style]
  (let [color ^RGBColor (.color style)]
    (doto g
      (.beginDraw)
      (.strokeWeight (.weight style))
      (.stroke (.r color) (.g color) (.b color))
      (.endDraw))
    g))

(defn fill! [^PGraphics g ^Style style]
  (let [color ^RGBColor (.color style)]
    (doto g
      (.beginDraw)
      (.fill (.r color) (.g color) (.b color))
      (.endDraw))
    g))

(defn clear! [^PGraphics g]
  (.beginDraw g)
  (.clear g)
  (.endDraw g)
  g)

(defn points!
  ([^PGraphics g x-axis y-axis xs ys]
   (let [map-x (axis-mapper x-axis 0 (dec (.width g)))
         map-y (axis-mapper y-axis (dec (.height g)) 0)]
     (.beginDraw g)
     (dotimes [i (xs)]
       (.point g (map-x (xs i)) (map-y (ys i))))
     (.endDraw g)
     g))
  ([^PGraphics g ^Colormap colormap x-axis y-axis xs ys]
   (let [map-x (axis-mapper x-axis 0 (dec (.width g)))
         map-y (axis-mapper y-axis (dec (.height g)) 0)
         map-01 (range-mapper (entry ys (imin ys)) (entry ys (imax ys)) 0.0 1.0)]
     (.beginDraw g)
     (dotimes [i (xs)]
       (let [y (ys i)
             y-01 (map-01 y)]
         (.stroke g (.r colormap y-01) (.g colormap y-01) (.b colormap y-01))
         (.point g (map-x (xs i)) (map-y y))))
     (.endDraw g)
     g))
  ([^PGraphics g ^Colormap colormap x-axis y-axis xs ys zs]
   (let [map-x (axis-mapper x-axis 0 (dec (.width g)))
         map-y (axis-mapper y-axis (dec (.height g)) 0)
         map-01 (range-mapper (entry zs (imin zs)) (entry zs (imax zs)) 0.0 1.0)]
     (.beginDraw g)
     (dotimes [i (xs)]
       (let [z-01 (map-01 (zs i))]
         (.stroke g (.r colormap z-01) (.g colormap z-01) (.b colormap z-01))
         (.point g (map-x (xs i)) (map-y (ys i)))))
     (.endDraw g)
     g)))

(defn x-sidelines! [^PGraphics g x-axis y-axis xs]
  (let [map-x (axis-mapper x-axis 0 (dec (.width g)))
        y0 (dec (.height g))]
    (.beginDraw g)
    (dotimes [i (xs)]
      (.line g (map-x (xs 0 i)) y0 (map-x (xs 1 i)) y0))
    (.endDraw g)
    g))

(defn y-sidelines! [^PGraphics g x-axis y-axis ys]
  (let [map-y (axis-mapper y-axis (dec (.height g)) 0)]
    (.beginDraw g)
    (dotimes [i (ys)]
      (.line g 0 (map-y (ys 0 i)) 0 (map-y (ys 1 i))))
    (.endDraw g)
    g))

(defprotocol Plot
  (width [this])
  (height [this])
  (render-frame [this options])
  (render-data [this options])
  (show [this] [this options]))

(defn render [plot options]
  (render-frame plot options)
  (render-data plot options))

(defrecord Plot2D [^PApplet applet ^Theme theme
                   ^long w ^long h
                   ^long x-density ^long y-density ^long grid-density
                   ^PGraphics main ^PGraphics g-frame ^PGraphics g-data
                   ^PGraphics x-ticks ^PGraphics y-ticks
                   ^PGraphics x-labels ^PGraphics y-labels
                   ^PGraphics x-grid ^PGraphics y-grid]
  Plot
  (width [this]
    w)
  (height [this]
    h)
  (render-frame [this options]
    (let [{:keys [x-axis y-axis]
           :or {x-axis (axis -1.0 1.0)
                y-axis (axis -1.0 1.0)}} options]
      (clear! x-ticks)
      (clear! y-ticks)
      (clear! x-labels)
      (clear! y-labels)
      (clear! x-grid)
      (clear! y-grid)
      (clear! g-data)
      (style! g-frame (.frame theme))
      (style! x-ticks (.ticks theme))
      (style! y-ticks (.ticks theme))
      (fill! x-labels (.labels theme))
      (fill! y-labels (.labels theme))
      (style! x-grid (.grid theme))
      (style! y-grid (.grid theme))
      (frame! g-frame)
      (bars! x-ticks x-axis x-density)
      (labels! x-labels x-axis x-density)
      (bars! x-grid x-axis (* x-density grid-density))
      (bars! y-ticks y-axis y-density)
      (labels! y-labels y-axis y-density)
      (bars! y-grid y-axis (* y-density grid-density))
      this))
  (render-data [this options]
    (let [{:keys [x y z x-axis y-axis style x-sidelines y-sidelines]
           :or {x-axis (axis -1.0 1.0)
                y-axis (axis -1.0 1.0)
                style (.data theme)}} options]
      (clear! g-data)
      (style! g-data style)
      (if z
        (points! g-data (.colormap theme) x-axis y-axis x y z)
        (points! g-data x-axis y-axis x y))
      (style! g-data (.sidelines theme))
      (when x-sidelines
        (x-sidelines! g-data x-axis y-axis x-sidelines))
      (when y-sidelines
        (y-sidelines! g-data x-axis y-axis y-sidelines))

      this))
  (show [this]
    (show this nil))
  (show [this options]
    (let [{:keys [frame left-ticks right-ticks top-ticks bottom-ticks
                  left-labels right-labels top-labels bottom-labels
                  vertical-grid horizontal-grid data]
           :or {frame true
                left-ticks true right-ticks true top-ticks true bottom-ticks true
                left-labels true right-labels true top-labels true bottom-labels true
                vertical-grid true horizontal-grid true data true}}
          options
          margin (/ (- (.width main) (.width g-frame)) 2.0)
          padding (/ (- (.width g-frame) (.width x-ticks)) 2.0)]
      (.beginDraw main)
      (.clear main)
      (when vertical-grid
        (.image main x-grid (+ margin padding) margin))
      (when top-ticks
        (.image main x-ticks (+ margin padding) (- margin (.height x-ticks))))
      (when bottom-ticks
        (.image main x-ticks (+ margin padding) (- (.height main) margin)))
      (when top-labels
        (.image main x-labels margin 0))
      (when bottom-labels
        (.image main x-labels margin (- (.height main) (.height x-labels))))
      (.pushMatrix main)
      (.translate main 0 (.height main))
      (.rotate main (- (/ PConstants/PI 2.0)))
      (when horizontal-grid
        (.image main y-grid (+ margin padding) margin))
      (when left-ticks
        (.image main y-ticks (+ margin padding) (- margin (.height y-ticks))))
      (when right-ticks
        (.image main y-ticks (+ margin padding) (- (.width main) margin)))
      (when left-labels
        (.image main y-labels margin 0))
      (when right-labels
        (.image main y-labels margin (- (.width main) (.height y-labels))))
      (.popMatrix main)
      (when frame (.image main g-frame margin margin))
      (when data
        (.image main g-data (+ margin padding) (+ margin padding)))
      (.endDraw main)
      main)))

(defn plot2d
  ([^PApplet applet
    {:keys [theme width height padding tick-length
            x-density y-density grid-density renderer]
     :or {padding 4 tick-length 4
          x-density 1.0 y-density 1.0 grid-density 2.0
          width (.sketchWidth applet) height (.sketchHeight applet)
          theme cyberpunk-theme renderer :p2d}}]
   (let [width (int width)
         height (int height)
         padding (int padding)
         tick-length (int tick-length)
         renderer (resolve-renderer renderer)
         main (.createGraphics applet width height renderer)
         text-descent (.textDescent main)
         text-height (+ (.textAscent main) (* 3 text-descent))
         margin (+ text-height tick-length)
         g-frame (.createGraphics applet
                                  (- width margin margin)
                                  (- height margin margin)
                                  renderer)
         x-ticks (.createGraphics applet
                                  (- width margin padding margin padding)
                                  tick-length
                                  renderer)
         y-ticks (.createGraphics applet
                                  (- height margin padding margin padding)
                                  tick-length
                                  renderer)
         x-labels (.createGraphics applet
                                   (- width margin margin) text-height
                                   renderer)
         y-labels (.createGraphics applet (- height margin margin) text-height
                                   renderer)
         x-grid (.createGraphics applet
                                 (- width margin padding margin padding)
                                 (- height margin margin)
                                 renderer)
         y-grid (.createGraphics applet
                                 (- height margin padding margin padding)
                                 (- width margin margin)
                                 renderer)
         g-data (.createGraphics applet
                                 (- width margin padding margin padding)
                                 (- height margin padding margin padding)
                                 renderer)]
     (->Plot2D applet theme width height x-density y-density grid-density
               main g-frame g-data x-ticks y-ticks x-labels y-labels
               x-grid y-grid)))
  ([applet]
   (plot2d applet {})))
