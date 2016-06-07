(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.examples.dbda.ch09-hierarchical-models-test
  (:require [midje.sweet :refer :all]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares
             [pause-on-error :refer [pause-on-error]]
             [fun-mode :refer [fun-mode]]]
            [uncomplicate.commons.core :refer [with-release let-release]]
            [uncomplicate.neanderthal
             [core :refer [row transfer dot imax imin scal! col]]
             [real :refer [entry entry!]]
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
  ([plot xs ps]
   (render plot {:x-axis (vector-axis xs) :x xs
                 :y-axis (vector-axis ps) :y ps}))
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

(defn bin-centers [limits ^long bin-count]
  (let [lower (entry limits 0)
        upper (entry limits 1)
        bin-width (/ (- upper lower) bin-count)
        start (+ lower (* bin-width 0.5))]
    (let-release [res (sv bin-count)]
      (dotimes [i bin-count]
        (entry! res i (+ start (* i bin-width))))
      res)))

(defn analysis [plots]
  (with-default-bayadera
    (let [sample-count (* 256 44 8)
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
        (let [prior-histogram (p/histogram (.dataset-eng prior-sample) (p/data prior-sample))
              post-histogram (p/histogram (.dataset-eng post-sample) (p/data post-sample))]

          (plot-distribution (:prior @plots)
                             (row (p/data prior-sample) 0)
                             (row (p/data prior-sample) 1) prior-pdf {})
          (render-sample (:prior-omega @plots)
                         (bin-centers (col (:limits prior-histogram) 0) 256)
                         (col (:pmf prior-histogram) 0))
          (render-sample (:prior-theta @plots)
                         (bin-centers (col (:limits prior-histogram) 1) 256)
                         (col (:pmf prior-histogram) 1))
          (plot-distribution (:posterior @plots) (row (p/data post-sample) 0)
                             (row (p/data post-sample) 1) post-pdf {})
          (render-sample (:posterior-omega @plots)
                         (bin-centers (col (:limits post-histogram) 0) 256)
                         (col (:pmf post-histogram) 0))
          (render-sample (:posterior-theta @plots)
                         (bin-centers (col (:limits post-histogram) 1) 256)
                         (col (:pmf post-histogram) 1)))))))

(defn setup []
  (reset! plots
          {:prior (plot2d (qa/current-applet) {:width 400 :height 400})
           :prior-omega (plot2d (qa/current-applet) {:width 400 :height 400})
           :prior-theta (plot2d (qa/current-applet) {:width 400 :height 400})
           :posterior (plot2d (qa/current-applet) {:width 400 :height 400})
           :posterior-omega (plot2d (qa/current-applet) {:width 400 :height 400})
           :posterior-theta (plot2d (qa/current-applet) {:width 400 :height 400})}))

(defn draw []
  (when (:changed @plots)
    (do
      (q/background 0)
      (analysis plots)
      (q/image (show (:prior @plots)) 0 0)
      (q/image (show (:prior-omega @plots)) 420 0)
      (q/image (show (:prior-theta @plots)) 0 420)
      (q/image (show (:posterior @plots)) 0 840)
      (q/image (show (:posterior-omega @plots)) 420 840)
      (q/image (show (:posterior-theta @plots)) 0 1260)
      (reset! plots (assoc @plots :changed false)))))

(defn display-sketch []
  (q/defsketch diagrams
    :renderer :p2d
    :size :fullscreen
    :display 2
    :setup setup
    :draw draw
    :middleware [pause-on-error]))
