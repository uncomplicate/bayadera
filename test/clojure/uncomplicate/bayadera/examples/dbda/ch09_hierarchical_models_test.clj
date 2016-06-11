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
             [core :refer [row transfer dot imax imin scal! col submatrix]]
             [real :refer [entry entry!]]
             [native :refer [sv sge]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [core :refer :all]
             [util :refer [bin-mapper]]
             [opencl :refer [with-default-bayadera]]]
            [uncomplicate.bayadera.opencl.models
             :refer [binomial-likelihood cl-distribution-model]]
            [uncomplicate.bayadera.toolbox
             [processing :refer :all]
             [plots :refer [render-sample render-histogram]]]
            [clojure.java.io :as io]))

(def all-data (atom {}))
(def state (atom nil))

(def ch09-1mint-1coin-model
  (cl-distribution-model [(slurp (io/resource "uncomplicate/bayadera/opencl/distributions/beta.h"))
                          (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch09-1mint-1coin.h"))]
                         :name "ch09_1mint_1coin" :params-size 3 :dimension 2
                         :limits (sge 2 2 [0 1 0 1])))

(defn analysis []
  (with-default-bayadera
    (let [walker-count (* 256 44 32)
          sample-count (* 16 walker-count)
          z 9 N 12]
      (with-release [prior (distribution ch09-1mint-1coin-model)
                     prior-dist (prior (sv 2 2 100))
                     prior-sampler (sampler prior-dist)
                     prior-sample (dataset (sample prior-sampler sample-count))
                     prior-pdf (pdf prior-dist prior-sample)
                     post (posterior "posterior_ch09" binomial-likelihood prior-dist)
                     post-dist (post (binomial-lik-params N z))
                     post-sampler (time (sampler post-dist))
                     post-sample (dataset (sample post-sampler sample-count))
                     post-pdf (scal! (/ 1.0 (evidence post-dist prior-sample))
                                     (pdf post-dist post-sample))]
        (println (p/diagnose prior-sampler))
        (println (p/diagnose post-sampler))
        {:prior {:sample (transfer (submatrix (p/data prior-sample) 0 0 2 walker-count))
                 :pdf (transfer prior-pdf)
                 :histogram (p/histogram prior-sampler (* 10 walker-count))}
         :posterior {:sample (transfer (submatrix (p/data post-sample) 0 0 2 walker-count))
                     :pdf (transfer post-pdf)
                     :histogram (time (p/histogram post-sampler (* 100 walker-count)))}}))))

(defn setup []
  (reset! state
          {:data @all-data
           :plots (repeatedly 6 (partial plot2d (qa/current-applet) {:width 400 :height 400}))}))

(defn draw-plots [[scatterplot omega theta] data ^long x-position ^long y-position]
  (q/image (show (render-sample scatterplot
                                (row (:sample data) 0)
                                (row (:sample data) 1)
                                (:pdf data)))
           x-position y-position)
  (q/image (show (render-histogram omega (:histogram data) 0 :rotate))
           (+ x-position 20 (width scatterplot)) y-position)
  (q/image (show (render-histogram omega (:histogram data) 1))
           x-position (+ y-position 20 (height scatterplot))))
(defn draw []
  (when-not (= @all-data (:data @state))
    (swap! state assoc :data @all-data)
    (q/background 0)
    (draw-plots (:plots @state) (:prior @all-data) 0 0)
    (draw-plots (drop 3 (:plots @state)) (:posterior @all-data) 0 840)))

(defn display-sketch []
  (q/defsketch diagrams
    :renderer :p2d
    :size :fullscreen
    :display 2
    :setup setup
    :draw draw
    :middleware [pause-on-error]))
