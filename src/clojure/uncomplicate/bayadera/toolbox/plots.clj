(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.toolbox.plots
  (:require [uncomplicate.neanderthal.core :refer [col]]
            [uncomplicate.bayadera.util :refer [bin-mapper]]
            [uncomplicate.bayadera.toolbox
             [scaling :refer [axis vector-axis]]
             [processing :refer [render]]]))

(defn render-sample
  ([plot xs ps]
   (render plot {:x-axis (vector-axis xs) :x xs
                 :y-axis (vector-axis ps) :y ps}))
  ([plot xs ys ps]
   (render plot {:x-axis (vector-axis xs) :x xs
                 :y-axis (vector-axis ys) :y ys
                 :z ps})))

(defn render-estimate
  ([plot estimate ^long dimension]
   (render-estimate plot estimate dimension false))
  ([plot estimate ^long dimension rotate]
   (let [limits (col (:limits estimate) dimension)
         ps (col (:histogram estimate) dimension)
         [x-axis y-axis x y] (if rotate
                               [:y-axis :x-axis :y :x]
                               [:x-axis :y-axis :x :y])]
     (render plot {x-axis (axis (limits 0) (limits 1))
                   x (bin-mapper (ps) (limits 0) (limits 1))
                   y-axis (vector-axis ps)
                   y ps}))))
