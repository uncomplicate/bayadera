(ns uncomplicate.bayadera.visual-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.bayadera.visual :refer :all]
            [uncomplicate.neanderthal
             [math :refer [log exp]]
             [core :refer [dim copy fmap!]]
             [native :refer [sv]]]
            [quil.core :as q]))

(def rand-vect (fmap! (fn ^double [^double x] (rand 10.0)) (sv 1000)))
(def pdf-vect (fmap! (fn ^double [^double x] (log (inc x))) (copy rand-vect)))

(def x-axis (axis 0 10))
(def y-axis (axis -3 3))

(def grid-color (->HSBColor 60 30 10))
(def frame-color (->HSBColor 200 50 60))
(def points-color (->HSBColor 320 100 100))

(def graphics (atom nil))

(defn setup []
  (let [g-x (q/create-graphics 800 640 :p2d)
        g-y (q/create-graphics 600 840 :p2d)
        g-tx (q/create-graphics 800 4 :p2d)
        g-ty (q/create-graphics 600 4 :p2d)
        g-lx (q/create-graphics 840 20 :p2d)
        g-ly (q/create-graphics 640 20 :p2d)
        f (q/create-graphics 840 640 :p2d)
        p (q/create-graphics 800 600 :p2d)]
    (style g-x grid-color)
    (style g-y grid-color)
    ;;(grid g 5 20 20)

    (bars x-axis  1.0 g-x)

    (bars y-axis  1.0 g-y)

    (style f frame-color)
    (frame f)
    (style g-tx frame-color)
    (bars x-axis  2.0 g-tx)
    (style g-ty frame-color)
    (bars y-axis 2.0 g-ty)

    (labels x-axis 2.0  g-lx)
    (labels y-axis 1.0  g-ly)
    (style p points-color 2)
    (points p x-axis y-axis rand-vect pdf-vect)

    (reset! graphics
            {:grid-x g-x
             :grid-y g-y
             :ticks-x g-tx
             :ticks-y g-ty
             :labels-x g-lx
             :labels-y g-ly
             :frame f
             :points p})))

(defn draw []
  (let [gr @graphics]
    (q/background 0)
    (q/translate 200 200)

    ;;(grid-style (:grid gr) grid-color)
    ;;(grid (:grid gr) 5 20 20)
    ;;(grid-style (:frame gr) frame-color)
    ;;(frame (:frame gr) 20)
    ;;(ticks (:frame gr) 20 5 3 100 100)
    ;;(grid-style (:points gr) points-color)
    ;;(points (:points gr) rand-vect pdf-vect 0 10 0 3)
    ;;(ticks (:frame gr) 20 40 60)
    (q/image (:grid-x gr) 20 0)

    (q/push-matrix)
    ;;Each graph will be inside its own graphics...
    (q/translate 860 20)
    (q/rotate (/ (double q/PI) 2.0))
    (q/image (:grid-y gr) 0 20)
    (q/pop-matrix)


    (q/image (:frame gr) 0 0)

    (q/image (:ticks-x gr) 20 640)

    (q/push-matrix)
    (q/translate 20 20)
    (q/rotate (/ (double q/PI) 2.0))
    (q/image (:ticks-y gr) 0 20)
    (q/pop-matrix)

    ;;(.beginDraw (:labels-x gr))
    ;;(.background (:labels-x gr) 22)
    ;;(.endDraw (:labels-y gr))

    (q/image (:labels-x gr) 0 640)

    (q/push-matrix)
    (q/translate (- 30) 640)
    (q/rotate (- (/ (double q/PI) 2.0)))
    (q/image (:labels-y gr) 0 0)
    (q/pop-matrix)

    (q/image (:points gr) 20 20)))

(q/defsketch diagrams
  :renderer :opengl
  :size :fullscreen
  :display 3
  :setup setup
  :draw draw)
