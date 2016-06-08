(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.visual
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [dim imin imax]]
             [real :refer [entry]]
             [math :refer [magnitude floor ceil]]]
            [quil.core :as q]
            [quil.applet :refer [resolve-renderer]])
  (:import [java.awt Color]
           [processing.core PGraphics PConstants PApplet]
           [clojure.lang IFn$DDDD IFn$DD]))

;; ============= Color mapping functions =================================

(definterface Colormap
  (^float r [^float x])
  (^float g [^float x])
  (^float b [^float x]))

(defmacro ^:private cube-helix-color [p0 p1 gamma s r h x]
  `(let [xg# (Math/pow ~x ~gamma)
         a# (* ~h xg# (- 1 xg#) 0.5)
         phi# (* 2 Math/PI (+ (/ ~s 3) (* ~r ~x)))]
     (* (float 255.0) (+ xg# (* a# (+ (* ~p0 (Math/cos phi#)) (* ~p1 (Math/sin phi#))))))))

(deftype CubeHelix [^float gamma ^float start-color ^float rotations ^float hue]
  Colormap
  (r [_ x]
    (cube-helix-color (float -0.14861) (float 1.78277)
                      gamma start-color rotations hue x))
  (g [_ x]
    (cube-helix-color (float -0.29227) (float -0.90649)
                      gamma start-color rotations hue x))
  (b [_ x]
    (cube-helix-color (float 1.97294) (float 0.0)
                      gamma start-color rotations hue x)))

(defn cube-helix
  ([^double gamma ^double start-color ^double rotations ^double hue]
   (CubeHelix. gamma start-color rotations hue))
  ([]
   (cube-helix 1.0 0.5 -1.5 1.0)))

(definterface RGBColor
  (^float r [])
  (^float g [])
  (^float b []))

(deftype ConstantColor [^float red ^float green ^float blue]
  RGBColor
  (r [_] red)
  (g [_] green)
  (b [_] blue)
  Colormap
  (r [_ x] red)
  (g [_ x] green)
  (b [_ x] blue))

(defn rgb-color [^double red ^double green ^double blue]
  (ConstantColor. red green blue))

(defn hsb-color [^double h ^double s ^double b]
  (let [color (Color/getHSBColor (/ h 360.0) (/ s 100.0) (/ b 100.0))]
    (ConstantColor. (.getRed color) (.getGreen color) (.getBlue color))))

;; ============= Styles and themes ========================================

(defrecord Style [^RGBColor color ^float weight])
(defrecord Theme [^Style frame ^Style ticks ^Style labels ^Style grid
                  ^Style data ^Colormap colormap])

(let [frame-style (->Style (hsb-color 200 50 60) 1)
      data-style (->Style (hsb-color 320 100 100) 2)
      grid-style (->Style (hsb-color 60 30 10) 1)
      label-style (->Style (hsb-color 180 40 100) 10)]
  (def cyberpunk-theme
    (->Theme frame-style frame-style label-style grid-style data-style (cube-helix))))

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

(defn range-mapper
  (^IFn$DD [^double start1 ^double end1 ^double start2 ^double end2]
   (fn ^double [^double value]
     (+ start2 (* (- end2 start2) (/ (- value start1) (- end1 start1))))))
  (^IFn$DDDD [^double start1 ^double end1]
   (fn ^double [^double value ^double start2 ^double end2]
     (+ start2 (* (- end2 start2) (/ (- value start1) (- end1 start1)))))))

(defrecord Axis [^double lower ^double upper ^double spacing])

(defn axis
  (^Axis [^double lower ^double upper]
   (axis lower upper 10.0))
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

(defn frame! [^PGraphics g]
  (do
    (.beginDraw g)
    (.rectMode g PConstants/CORNER)
    (.noFill g)
    (.rect g 0 0 (dec (.width g)) (dec (.height g)))
    (.endDraw g)
    g))

(defn bars! [^PGraphics g ^Axis axis ^double density]
  (let [height (.height g)
        ofst (/ (.spacing axis) density)
        upper (.upper axis)
        map-range (axis-mapper axis 0 (dec (.width g)))]
    (.beginDraw g)
    (loop [value (.lower axis)]
      (if (<= value upper)
        (let [x (floor (map-range value))]
          (.line g x 0 x height)
          (recur (+ value ofst)))))
    (.endDraw g)
    g))

(defn format-number [^double n]
  (PApplet/nf n))

(defn labels!
  ([^PGraphics g ^Axis axis nf ^long density]
   (let [height (float (.height g))
         ofst (* (.spacing axis) density)
         upper (.upper axis)
         left-padding (/ (.textWidth g ^String (nf (.lower axis))) 2.0)
         right-padding (/ (.textWidth g ^String (nf upper)) 2.0)
         map-range (axis-mapper axis left-padding (- (.width g) right-padding))]
     (.beginDraw g)
     (.textAlign g PConstants/CENTER)
     (loop [value (.lower axis)]
       (if (<= value upper)
         (let [x (floor (map-range value))]
           (.text g ^String (nf value)
                  (float (map-range value)) (- height (* 2 (.textDescent g))))
           (recur (+ value ofst)))))
     (.endDraw g)
     g))
  ([^PGraphics g ^Axis axis ^double density]
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
  ([^PGraphics g ^Axis x-axis ^Axis y-axis xs ys]
   (let [map-x (axis-mapper x-axis 0 (dec (.width g)))
         map-y (axis-mapper y-axis (dec (.height g)) 0)]
     (.beginDraw g)
     (dotimes [i (dim xs)]
       (.point g (map-x (entry xs i)) (map-y (entry ys i))))
     (.endDraw g)
     g))
  ([^PGraphics g ^Colormap colormap ^Axis x-axis ^Axis y-axis xs ys]
   (let [map-x (axis-mapper x-axis 0 (dec (.width g)))
         map-y (axis-mapper y-axis (dec (.height g)) 0)
         map-01 (range-mapper (entry ys (imin ys)) (entry ys (imax ys)) 0.0 1.0)]
     (.beginDraw g)
     (dotimes [i (dim xs)]
       (let [y (entry ys i)
             y-01 (map-01 y)]
         (.stroke g (.r colormap y-01) (.g colormap y-01) (.b colormap y-01))
         (.point g (map-x (entry xs i)) (map-y y))))
     (.endDraw g)
     g))
  ([^PGraphics g ^Colormap colormap ^Axis x-axis ^Axis y-axis xs ys zs]
   (let [map-x (axis-mapper x-axis 0 (dec (.width g)))
         map-y (axis-mapper y-axis (dec (.height g)) 0)
         map-01 (range-mapper (entry zs (imin zs)) (entry zs (imax zs)) 0.0 1.0)]
     (.beginDraw g)
     (dotimes [i (dim xs)]
       (let [z-01 (map-01 (entry zs i))]
         (.stroke g (.r colormap z-01) (.g colormap z-01) (.b colormap z-01))
         (.point g (map-x (entry xs i)) (map-y (entry ys i)))))
     (.endDraw g)
     g)))

(defn vertical-lines! [^PGraphics g ^Axis x-axis ^Axis y-axis marks]
  (let [map-x (axis-mapper x-axis 0 (dec (.width g)))
        map-y (axis-mapper y-axis (dec (.height g)) 0)
        y-0 (map-y 0)]
    (.beginDraw g)
    (doseq [[x y] marks]
      (.line g (map-x x) y-0 (map-x x) (map-y y)))
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
    (let [{:keys [x y z x-axis y-axis style vertical-marks]
           :or {x-axis (axis -1.0 1.0)
                y-axis (axis -1.0 1.0)
                style (.data theme)
                vertical-marks []}} options]
      (clear! g-data)
      (style! g-data style)
      (if z
        (points! g-data (.colormap theme) x-axis y-axis x y z)
        (points! g-data x-axis y-axis x y))
      (style! g-data (.ticks theme))
      (vertical-lines! g-data x-axis y-axis vertical-marks)
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
     (->Plot2D applet theme x-density y-density grid-density
               main g-frame g-data x-ticks y-ticks x-labels y-labels
               x-grid y-grid)))
  ([applet]
   (plot2d applet {})))
