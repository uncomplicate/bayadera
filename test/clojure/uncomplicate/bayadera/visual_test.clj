(ns uncomplicate.bayadera.visual-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.bayadera.visual :refer :all]
            [uncomplicate.neanderthal
             [math :refer [log exp]]
             [core :refer [dim sum copy fmap! entry]]
             [native :refer [sv]]]
            [quil.core :as q]))

(def rand-vect (fmap! (fn ^double [^double x] (rand 10.0)) (sv 10000)))
(def pdf-vect (fmap! (fn ^double [^double x] (log x)) (copy rand-vect)))

(def grid-color (->HSBColor 110 30 10))
(def axis-color (->HSBColor 200 60 80))
(def points-color (->HSBColor 320 100 100))

(def graphics (atom nil))

(defn setup []
  (let [g (q/create-graphics 1090 630 :p2d)
        x (q/create-graphics 1090 20 :p2d)
        y (q/create-graphics 20 630 :p2d)
        p (q/create-graphics 1090 630 :p2d)]
    (grid-style g grid-color)
    (grid g 20 20)
    (grid-style x axis-color)
    (x-axis x 4 100)
    (grid-style y axis-color)
    (y-axis y 4 100)
    (grid-style p points-color)
    (points p rand-vect pdf-vect 0 10 (- 5) 5)
    (reset! graphics
            {:grid g
             :x-axis x
             :y-axis y
             :points p})))

(defn draw []
  (let [gr @graphics]
    (q/background 0)
    (q/translate 400 400)
    (grid-style (:points gr) points-color)
    (points (:points gr) rand-vect pdf-vect 0 10 (- 3) 1)
    (q/image (:grid gr) 0 0)
    (q/image (:y-axis gr) -19 0)
    (q/image (:x-axis gr) 0 629)
    (q/image (:points gr) 0 0)
    )

  )

(q/defsketch diagrams
  :renderer :opengl
  :size :fullscreen
  :display 3
  :setup setup
  :draw draw)
