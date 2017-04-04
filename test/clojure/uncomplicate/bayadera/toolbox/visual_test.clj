;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.toolbox.visual-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release double-fn]]
            [uncomplicate.fluokitten.core :refer [fmap fmap!]]
            [uncomplicate.neanderthal
             [math :refer [log exp]]
             [core :refer [dim copy imax imin]]
             [real :refer [entry]]
             [native :refer [fv fge]]]
            [uncomplicate.bayadera.toolbox
             [processing :refer [plot2d render show]]
             [scaling :refer [vector-axis]]]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares.pause-on-error :refer [pause-on-error]])
  (:import [processing.core PGraphics PConstants PApplet]
           [clojure.lang IFn$DDDD IFn$DD]))

(defn setup []
  (with-release [rand-vect (fmap! (fn ^double [^double x] (rand 10.0)) (fv 100))
                 pdf-vect (fmap (fn ^double [^double x] (log (inc x))) rand-vect)]
    (q/background 0)
    (q/image (-> (plot2d (qa/current-applet))
                 (render {:x-axis (vector-axis rand-vect)
                          :y-axis (vector-axis pdf-vect)
                          :x rand-vect
                          :y pdf-vect
                          :y-sidelines (fge 2 1 [1.0 8.0])
                          :x-sidelines (fge 2 3 [1.3 1.8 3.5 4.2 5.0 9.0])})
                 show)
             0 0)))

(defn display-sketch []
  (q/defsketch diagrams
    :renderer :opengl
    :size :fullscreen
    :display 2
    :setup setup
    :middleware [pause-on-error]))
