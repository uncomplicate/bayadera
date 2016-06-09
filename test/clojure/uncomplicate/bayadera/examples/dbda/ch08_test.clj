(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.examples.dbda.ch08-test
  (:require [midje.sweet :refer :all]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares
             [pause-on-error :refer [pause-on-error]]
             [fun-mode :refer [fun-mode]]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core
             :refer [row transfer dot entry imax imin scal!]]
            [uncomplicate.bayadera
             [protocols :as p]
             [core :refer :all]
             [opencl :refer [with-default-bayadera]]]
            [uncomplicate.bayadera.opencl.models
             :refer [binomial-likelihood beta-model]]
            [uncomplicate.bayadera.toolbox.visual :refer :all]
            [clojure.java.io :as io]))

(defn render-sample
  ([plot xs ps]
   (let [imax-p (imax ps)]
     (render plot {:x-axis (vector-axis xs) :x xs
                   :y-axis (vector-axis ps) :y ps
                   :vertical-lines [[(entry xs imax-p) (entry ps imax-p)]]}))))

(defmulti plot-distribution
  (fn [plot xs ps options]
    [(class xs) (class ps)]))

(defmethod plot-distribution [uncomplicate.neanderthal.opencl.clblock.CLBlockVector
                              uncomplicate.neanderthal.opencl.clblock.CLBlockVector]
  [plot xs ps options]
  (with-release [host-xs (transfer xs)
                 host-ps (transfer ps)]
    (render-sample plot host-xs host-ps)))

(def plots (atom nil))

(defn analysis [prior-plot posterior-plot]
  (with-default-bayadera
    (let [sample-count (* 256 44)
          a 1 b 1
          z 15 N 50]
      (with-release [prior-dist (beta a b)
                     prior-sample (dataset (sample (sampler prior-dist) sample-count))
                     prior-pdf (pdf prior-dist prior-sample)
                     post (posterior (posterior-model binomial-likelihood beta-model))
                     post-dist (post (binomial-lik-params N z) (beta-params a b))
                     post-sampler (time (sampler post-dist))
                     post-sample (dataset (sample post-sampler sample-count))
                     post-pdf (scal! (/ 1.0 (evidence post-dist prior-sample))
                                     (pdf post-dist post-sample))]

        (plot-distribution prior-plot (row (p/data prior-sample) 0) prior-pdf {})
        (plot-distribution posterior-plot (row (p/data post-sample) 0) post-pdf {})))))

(defn setup []
  (reset! plots
          {:prior (plot2d (qa/current-applet) {:width 1000 :height 700})
           :posterior (plot2d (qa/current-applet) {:width 1000 :height 700})}))

(defn draw []
  (when (:changed @plots)
    (do
      (q/background 0)
      (analysis (:prior @plots) (:posterior @plots))
      (q/image (show (:prior @plots)) 0 0)
      (q/image (show (:posterior @plots)) 0 720)
      (reset! plots (assoc @plots :changed false)))))

(defn display-sketch []
  (q/defsketch diagrams
    :renderer :p2d
    :size :fullscreen
    :display 2
    :setup setup
    :draw draw
    :middleware [pause-on-error]))
