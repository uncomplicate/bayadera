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
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [core :refer :all]
             [opencl :refer [with-default-bayadera]]]
            [uncomplicate.bayadera.opencl.models
             :refer [binomial-likelihood cl-distribution-model]]
            [uncomplicate.bayadera.toolbox.visual :refer :all]
            [clojure.java.io :as io]))

(defn reconstruct-values [limits ^long bin-count]
  (let [lower (entry limits 0)
        upper (entry limits 1)
        bin-width (/ (- upper lower) bin-count)
        start (+ lower (* bin-width 0.5))]
    (let-release [res (sv bin-count)]
      (dotimes [i bin-count]
        (entry! res i (+ start (* i bin-width))))
      res)))

(defn render-sample
  ([plot xs ps]
   (render plot {:x-axis (vector-axis xs) :x xs
                 :y-axis (vector-axis ps) :y ps}))
  ([plot xs ys ps]
   (render plot {:x-axis (vector-axis xs) :x xs
                 :y-axis (vector-axis ys) :y ys
                 :z ps})))

(def all-data (atom {}))
(def state (atom nil))

(def ch09-1mint-1coin-model
  (cl-distribution-model [(slurp (io/resource "uncomplicate/bayadera/opencl/distributions/beta.h"))
                          (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch09-1mint-1coin.h"))]
                         :name "ch09_1mint_1coin" :params-size 3 :dimension 2
                         :lower (sv 0 0) :upper (sv 1 1)))

(defn analysis []
  (with-default-bayadera
    (let [walker-count (* 256 44 32)
          sample-count (* 16 walker-count)
          a 1 b 1
          z 9 N 12]
      (with-release [prior (distribution ch09-1mint-1coin-model)
                     prior-dist (prior (sv 2 2 100))
                     prior-sample (dataset (sample (sampler prior-dist) sample-count))
                     prior-pdf (pdf prior-dist prior-sample)
                     post (posterior "posterior_ch09" binomial-likelihood prior-dist)
                     post-dist (post (binomial-lik-params N z))
                     post-sampler (time (sampler post-dist))
                     post-sample (dataset (sample post-sampler sample-count))
                     post-pdf (scal! (/ 1.0 (evidence post-dist prior-sample))
                                     (pdf post-dist post-sample))]
        {:prior {:sample (transfer (submatrix (p/data prior-sample) 0 0 2 walker-count))
                 :pdf (transfer prior-pdf)
                 :histogram (p/histogram (.dataset-eng prior-sample) (p/data prior-sample))}
         :posterior {:sample (transfer (submatrix (p/data post-sample) 0 0 2 walker-count))
                     :pdf (transfer post-pdf)
                     :histogram (p/histogram (.dataset-eng post-sample) (p/data post-sample))}}))))

(defn setup []
  (reset! state
          {:data @all-data
           :prior {:scatterplot (plot2d (qa/current-applet) {:width 400 :height 400})
                   :omega (plot2d (qa/current-applet) {:width 400 :height 400})
                   :theta (plot2d (qa/current-applet) {:width 400 :height 400})}
           :posterior {:scatterplot (plot2d (qa/current-applet) {:width 400 :height 400})
                       :omega (plot2d (qa/current-applet) {:width 400 :height 400})
                       :theta (plot2d (qa/current-applet) {:width 400 :height 400})}}))

(defn draw-plots [g data ^long x-position ^long y-position]
  (let [scatterplot (:scatterplot g)
        omega (:omega g)
        theta (:theta g)]
    (q/image (show (render-sample scatterplot
                                  (row (:sample data) 0)
                                  (row (:sample data) 1)
                                  (:pdf data)))
             x-position y-position)
    (q/image (show (render-sample omega
                                  (col (:pdf (:histogram data)) 0)
                                  (reconstruct-values (col (:limits (:histogram data)) 0) 256)))
             (+ x-position 20 (.width (.main scatterplot))) y-position)
    (q/image (show (render-sample theta
                                  (reconstruct-values (col (:limits (:histogram data)) 1) 256)
                                  (col (:pdf (:histogram data)) 1)))
             x-position (+ y-position 20 (.height (.main scatterplot))))))

(defn draw []
  (when-not (= @all-data (:data @state))
    (swap! state assoc :data @all-data)
    (q/background 0)
    (draw-plots (:prior @state) (:prior @all-data) 0 0)
    (draw-plots (:posterior @state) (:posterior @all-data) 0 840)))

(defn display-sketch []
  (q/defsketch diagrams
    :renderer :p2d
    :size :fullscreen
    :display 2
    :setup setup
    :draw draw
    :middleware [pause-on-error]))
