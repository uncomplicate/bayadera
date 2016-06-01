(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.examples.dbda.ch09-hierarchical-models-test
  (:require [midje.sweet :refer :all]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares
             [pause-on-error :refer [pause-on-error]]
             [fun-mode :refer [fun-mode]]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [row transfer dot entry imax imin scal!]]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [core :refer :all]
             [visual :refer :all]]
            [uncomplicate.bayadera.opencl :refer [with-default-bayadera]]
            [uncomplicate.bayadera.opencl.models
             :refer [binomial-likelihood cl-distribution-model]]
            [clojure.java.io :as io]))

(defn render-sample
  ([plot xs ys ps]
   (render plot {:x-axis (vector-axis xs) :x xs
                 :y-axis (vector-axis ys) :y ys
                 :z ps})))

(defmulti plot-distribution
  (fn [plot xs ys ps options]
    [(class xs) (class ys)]))

(defmethod plot-distribution [uncomplicate.neanderthal.opencl.clblock.CLBlockVector
                              uncomplicate.neanderthal.opencl.clblock.CLBlockVector]
  [plot xs ys ps options]
  (with-release [host-xs (transfer xs)
                 host-ys (transfer ys)
                 host-ps (transfer ps)]
    (render-sample plot host-xs host-ys host-ps)))

(def plots (atom nil))

(def ch09-1mint-1coin-model
  (cl-distribution-model [(slurp (io/resource "uncomplicate/bayadera/opencl/distributions/beta.h"))
                          (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch09-1mint-1coin.h"))]
                         :name "ch09_1mint_1coin" :params-size 3 :dimension 2
                         :lower (sv 0 0) :upper (sv 1 1)))

(defn analysis [prior-plot posterior-plot]
  (with-default-bayadera
    (let [sample-count (* 256 44)
          a 1 b 1
          z 9 N 12]
      (with-release [prior (distribution ch09-1mint-1coin-model)
                     prior-dist (prior (sv 2 2 100))
                     prior-sample (dataset (sample (sampler prior-dist {:warm-up 8092 :iterations 8092 :walkers sample-count}) sample-count))
                     prior-pdf (pdf prior-dist prior-sample)
                     post (posterior "posterior_ch09" binomial-likelihood prior-dist)
                     post-dist (post (binomial-lik-params N z))
                     post-sampler (time (sampler post-dist {:warm-up 80920 :iterations 8092 :walkers sample-count}))
                     post-sample (dataset (sample post-sampler sample-count))
                     post-pdf (scal! (/ 1.0 (evidence post-dist prior-sample))
                                     (pdf post-dist post-sample))]

        (plot-distribution prior-plot (row (p/data prior-sample) 0) (row (p/data prior-sample) 1) prior-pdf {})
        (plot-distribution posterior-plot (row (p/data post-sample) 0) (row (p/data post-sample) 1) post-pdf {})))))

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
