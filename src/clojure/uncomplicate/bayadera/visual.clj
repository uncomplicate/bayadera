(ns uncomplicate.bayadera.visual
  (:require [uncomplicate.clojurecl.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [dim]]
             [real :refer [entry]]
             [native :refer [dv]]]
            [quil.core :as q]
            [quil.applet :refer [resolve-renderer]])
  (:import [processing.core PGraphics PConstants PApplet]
           [clojure.lang IFn$DDDD IFn$DD]))

(defrecord HSBColor [^float h ^float s ^float b])
(defrecord Style [^HSBColor color ^float weight])
(defrecord Theme [^Style frame ^Style ticks ^Style labels ^Style grid ^Style data])

(let [frame-style (->Style (->HSBColor 200 50 60) 1)
      data-style (->Style (->HSBColor 320 100 100) 4)
      grid-style (->Style (->HSBColor 60 30 10) 1)
      label-style (->Style (->HSBColor 200 40 70) 10)]
  (def cyberpunk-theme
    (->Theme frame-style frame-style label-style grid-style data-style)))

(defn range-mapper
  (^IFn$DD [^double start1 ^double end1 ^double start2 ^double end2]
   (fn ^double [^double value]
     (+ start2 (* (- end2 start2) (/ (- value start1) (- end1 start1))))))
  (^IFn$DDDD [^double start1 ^double end1]
   (fn ^double [^double value ^double start2 ^double end2]
     (+ start2 (* (- end2 start2) (/ (- value start1) (- end1 start1)))))))

(defrecord Axis [^double lower ^double upper])

(defn axis [^double lower ^double upper]
  (->Axis lower upper))

(defn axis-mapper
  (^IFn$DD [^Axis axis ^double start ^double end]
   (range-mapper (.lower axis) (.upper axis) start end))
  (^IFn$DDDD [^Axis axis]
   (range-mapper (.lower axis) (.upper axis))))

(defn offset ^double [^Axis axis ^long density]
  (/ (- (.upper axis) (.lower axis)) (double density)))

(defn frame! [^PGraphics g]
  (do
    (.beginDraw g)
    (.rectMode g PConstants/CORNER)
    (.noFill g)
    (.rect g 0 0 (dec (.width g)) (dec (.height g)))
    (.endDraw g)
    g))

(defn bars! [^PGraphics g ^Axis axis ^long density]
  (let [height (.height g)
        ofst (offset axis density)
        map-range (axis-mapper axis 0 (dec (.width g)))]
    (.beginDraw g)
    (dotimes [i (inc density)]
      (let [x (Math/floor (map-range (+ (.lower axis ) (* (double i) ofst))))]
        (.line g x 0 x height)))
    (.endDraw g)
    g))

(defn format-number [^double n]
  (PApplet/nf n))

(defn labels!
  ([^PGraphics g ^Axis axis nf ^long density]
   (let [height (float (.height g))
         ofst (offset axis density)
         left-padding (/ (.textWidth g ^String (nf (.lower axis))) 2.0)
         right-padding (/ (.textWidth g ^String (nf (.upper axis))) 2.0)
         map-range (axis-mapper axis left-padding (- (.width g) right-padding))]
     (.beginDraw g)
     (.textAlign g PConstants/CENTER)
     (dotimes [i (inc density)]
       (let [value (+ (.lower axis) (* (double i) ofst))]
         (.text g ^String (nf value)
                (float (map-range value)) (- height (* 2 (.textDescent g))))))
     (.endDraw g)
     g))
  ([^PGraphics g ^Axis axis ^double density]
   (labels! g axis format-number density)))

(defn style! [^PGraphics g ^Style style]
  (let [color ^HSBColor (.color style)]
    (doto g
      (.beginDraw)
      (.colorMode PConstants/HSB 360 100 100)
      (.strokeWeight (.weight style))
      (.stroke (.h color) (.s color) (.b color))
      (.endDraw))
    g))

(defn fill! [^PGraphics g ^Style style]
  (let [color ^HSBColor (.color style)]
    (doto g
      (.beginDraw)
      (.colorMode PConstants/HSB 360 100 100)
      (.fill (.h color) (.s color) (.b color))
      (.endDraw))
    g))

(defn clear! [^PGraphics g]
  (.beginDraw g)
  (.clear g)
  (.endDraw g)
  g)

(defn points! [^PGraphics g ^Axis x-axis ^Axis y-axis xs ys]
  (let [map-x (axis-mapper x-axis 0 (dec (.width g)))
        map-y (axis-mapper y-axis (dec (.height g)) 0)]
    (.beginDraw g)
    (dotimes [i (dim xs)]
      (.point g (map-x (entry xs i)) (map-y (entry ys i))))
    (.endDraw g)
    g))

(defprotocol Plot
  (render-frame [this options])
  (render-data [this options])
  (show [this] [this options]))

(defn render [plot options]
  (render-frame plot options)
  (render-data plot options))

(defrecord Plot2D [^PApplet applet ^Theme theme
                   ^long x-density ^long y-density ^long grid-density
                   ^PGraphics main ^PGraphics g-frame ^PGraphics g-data
                   ^PGraphics x-ticks ^PGraphics y-ticks
                   ^PGraphics x-labels ^PGraphics y-labels
                   ^PGraphics x-grid ^PGraphics y-grid]
  Plot
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
    (let [{:keys [x y x-axis y-axis style]
           :or {x-axis (axis -1.0 1.0)
                y-axis (axis -1.0 1.0)
                style (.data theme)}} options]
      (clear! g-data)
      (style! g-data style)
      (points! g-data x-axis y-axis x y)
      this))
  (show [this]
    (show this {}))
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
      (.background main 0)
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
          x-density 10 y-density 10 grid-density 2
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
     (->Plot2D applet theme x-density y-density grid-density
               main g-frame g-data x-ticks y-ticks x-labels y-labels
               x-grid y-grid)))
  ([applet]
   (plot2d applet {})))
