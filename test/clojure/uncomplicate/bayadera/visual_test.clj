(ns uncomplicate.bayadera.visual-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.bayadera.visual :refer :all]
            [uncomplicate.neanderthal
             [math :refer [log exp]]
             [core :refer [dim copy fmap!]]
             [native :refer [sv]]]
            [quil.core :as q]))

(def rand-vect (fmap! (fn ^double [^double x] (rand 10.0)) (sv 300)))
(def pdf-vect (fmap! (fn ^double [^double x] (log (inc x))) (copy rand-vect)))

(def grid-color (->HSBColor 60 30 10))
(def frame-color (->HSBColor 200 50 60))
(def points-color (->HSBColor 320 100 100))

(def graphics (atom nil))

(defn setup []
  (let [g (q/create-graphics 310 250 :p2d)
        f (q/create-graphics 350 290 :p2d)
        p (q/create-graphics 302 242 :p2d)]
    (style g grid-color)
    (grid g 5 20 20)
    (style f frame-color)
    (frame f 20)
    (ticks f 20 5 3 100 100)
    (labels f 5 0 10 1 0 3 0.5)
    (style p points-color 4)
    (points p rand-vect pdf-vect 0 10 0 3)
    (reset! graphics
            {:grid g
             :frame f
             :points p})))

(defn draw []
  (let [gr @graphics]
    (q/background 0)
    (q/translate 100 100)

    ;;(grid-style (:grid gr) grid-color)
    ;;(grid (:grid gr) 5 20 20)
    ;;(grid-style (:frame gr) frame-color)
    ;;(frame (:frame gr) 20)
    ;;(ticks (:frame gr) 20 5 3 100 100)
    ;;(grid-style (:points gr) points-color)
    (points (:points gr) rand-vect pdf-vect 0 10 0 3)
    ;;(ticks (:frame gr) 20 40 60)
    (q/image (:grid gr) 20 20)
    (q/image (:frame gr) 0 0)
    (q/image (:points gr) 24 24)
    )

  )

(q/defsketch diagrams
  :renderer :opengl
  :size :fullscreen
  :display 3
  :setup setup
  :draw draw)
