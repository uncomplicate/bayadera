(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.toolbox.visual-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release double-fn]]
            [uncomplicate.fluokitten.core :refer [fmap fmap!]]
            [uncomplicate.neanderthal
             [math :refer [log exp]]
             [core :refer [dim copy imax imin]]
             [real :refer [entry]]
             [native :refer [sv]]]
            [uncomplicate.bayadera.toolbox.visual :refer :all]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares.pause-on-error :refer [pause-on-error]])
  (:import [processing.core PGraphics PConstants PApplet]
           [clojure.lang IFn$DDDD IFn$DD]))

(defn setup []
  (with-release [rand-vect (fmap! (fn ^double [^double x] (rand 10.0)) (sv 100))
                 pdf-vect (fmap (fn ^double [^double x] (log (inc x))) rand-vect)]
    (q/background 0)
    (q/image (-> (plot2d (qa/current-applet))
                 (render {:x-axis (vector-axis rand-vect)
                          :y-axis (vector-axis pdf-vect)
                          :x rand-vect
                          :y pdf-vect})
                 show)
             0 0)))

(defn display-sketch []
  (q/defsketch diagrams
    :renderer :opengl
    :size :fullscreen
    :display 3
    :setup setup
    :middleware [pause-on-error]))
