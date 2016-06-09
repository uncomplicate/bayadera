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
             [visual :refer :all]]
            [uncomplicate.bayadera.opencl :refer [with-default-bayadera]]
            [uncomplicate.bayadera.opencl.models
             :refer [binomial-likelihood cl-distribution-model]]
            [clojure.java.io :as io]))

(defn bin-centers [limits ^long bin-count]
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
        {:prior-sample (transfer (submatrix (p/data prior-sample) 0 0 2 walker-count))
         :prior-pdf (transfer prior-pdf)
         :prior-histogram (p/histogram (.dataset-eng prior-sample) (p/data prior-sample))
         :posterior-sample (transfer (submatrix (p/data post-sample) 0 0 2 walker-count))
         :posterior-pdf (transfer post-pdf)
         :posterior-histogram (p/histogram (.dataset-eng post-sample) (p/data post-sample))}))))

(defn setup []
  (reset! state
          {:data @all-data
           :prior (plot2d (qa/current-applet) {:width 400 :height 400})
           :prior-omega (plot2d (qa/current-applet) {:width 400 :height 400})
           :prior-theta (plot2d (qa/current-applet) {:width 400 :height 400})
           :posterior (plot2d (qa/current-applet) {:width 400 :height 400})
           :posterior-omega (plot2d (qa/current-applet) {:width 400 :height 400})
           :posterior-theta (plot2d (qa/current-applet) {:width 400 :height 400})}))

(defn draw []
  (when-not (= @all-data (:data @state))
    (swap! state assoc :data @all-data)
    (q/background 0)
    (q/image (show (render-sample (:prior @state)
                                  (row (:prior-sample @all-data) 0)
                                  (row (:prior-sample @all-data) 1)
                                  (:prior-pdf @all-data)))
             0 0)
    (q/image (show (render-sample (:prior-omega @state)
                                  (col (:pmf (:prior-histogram @all-data)) 0)
                                  (bin-centers (col (:limits (:prior-histogram @all-data)) 0) 256)
                                  (col (:pmf (:prior-histogram @all-data)) 0)))
             420 0)
    (q/image (show (render-sample (:prior-theta @state)
                                  (bin-centers (col (:limits (:prior-histogram @all-data)) 1) 256)
                                  (col (:pmf (:prior-histogram @all-data)) 1)))
             0 420)
    (q/image (show (render-sample (:prior @state)
                                  (row (:posterior-sample @all-data) 0)
                                  (row (:posterior-sample @all-data) 1)
                                  (:posterior-pdf @all-data)))
             0 840)
    (q/image (show (render-sample (:posterior-omega @state)
                                  (col (:pmf (:posterior-histogram @all-data)) 0)
                                  (bin-centers (col (:limits (:posterior-histogram @all-data)) 0) 256)))
             420 840)
    (q/image (show (render-sample (:posterior-theta @state)
                                  (bin-centers (col (:limits (:posterior-histogram @all-data)) 1) 256)
                                  (col (:pmf (:posterior-histogram @all-data)) 1)))
             0 1260)))

(defn display-sketch []
  (q/defsketch diagrams
    :renderer :p2d
    :size :fullscreen
    :display 2
    :setup setup
    :draw draw
    :middleware [pause-on-error]))
