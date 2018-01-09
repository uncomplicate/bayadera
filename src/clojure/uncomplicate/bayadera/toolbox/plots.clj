(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.toolbox.plots
  (:require [uncomplicate.neanderthal.core :refer [col]]
            [uncomplicate.bayadera.util
             :refer [bin-mapper hdi-regions hdi-bins hdi-rank-count]]
            [uncomplicate.bayadera.toolbox
             [scaling :refer [axis vector-axis]]
             [processing :refer [render]]])
  (:import [uncomplicate.bayadera.internal.protocols Histogram]))

(defn render-sample
  ([plot xs ps]
   (render plot {:x-axis (vector-axis xs) :x xs
                 :y-axis (vector-axis ps) :y ps}))
  ([plot xs ys ps]
   (render plot {:x-axis (vector-axis xs) :x xs
                 :y-axis (vector-axis ys) :y ys
                 :z ps})))

(defn render-histogram
  ([plot histogram ^long index]
   (render-histogram plot histogram index false 0.95))
  ([plot histogram ^long index rotate]
   (render-histogram plot histogram index rotate 0.95))
  ([plot ^Histogram histogram index rotate hdi-mass]
   (let [limits (col (.limits histogram) index)
         ps (col (.pdf histogram) index)
         bin-rank (col (.bin-ranks histogram) index)
         [x-axis y-axis x y sidelines] (if rotate
                                         [:y-axis :x-axis :y :x :y-sidelines]
                                         [:x-axis :y-axis :x :y :x-sidelines])]
     (render plot {x-axis (axis (limits 0) (limits 1))
                   x (bin-mapper (ps) (limits 0) (limits 1))
                   y-axis (vector-axis ps)
                   y ps
                   sidelines (hdi-regions limits bin-rank (hdi-rank-count hdi-mass bin-rank ps))}))))
