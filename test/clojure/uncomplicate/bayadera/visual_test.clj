(ns uncomplicate.bayadera.visual-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.bayadera.visual :refer :all]
            [uncomplicate.neanderthal
             [math :refer [log exp]]
             [core :refer [dim copy fmap!]]
             [native :refer [sv]]]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares.pause-on-error :refer [pause-on-error]])
  (:import [processing.core PGraphics PConstants PApplet]
           [clojure.lang IFn$DDDD IFn$DD]))

(defn setup []
  (with-release [rand-vect (fmap! (fn ^double [^double x] (rand 10.0)) (sv 100))
                 pdf-vect (fmap! (fn ^double [^double x] (log (inc x))) (copy rand-vect))]
    (let [plot (plot2d (qa/current-applet))]
      (q/background 55)
      (render plot {:x-axis (axis 0 10) :y-axis (axis -3 3) :x rand-vect :y pdf-vect})
      (q/image (show plot {}) 0 0))))

(defn draw [])

(q/defsketch diagrams
  :renderer :opengl
  :size :fullscreen
  :display 3
  :setup setup
  :draw draw
  :middleware [pause-on-error])
