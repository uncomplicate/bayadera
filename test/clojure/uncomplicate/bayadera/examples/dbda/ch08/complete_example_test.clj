(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.examples.dbda.ch08.complete-example-test
  (:require [midje.sweet :refer :all]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares
             [pause-on-error :refer [pause-on-error]]
             [fun-mode :refer [fun-mode]]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core
             :refer [row native scal!]]
            [uncomplicate.bayadera
             [protocols :as p]
             [core :refer :all]
             [distributions :refer [beta-params binomial-lik-params]]
             [mcmc :refer [mix!]]
             [opencl :refer [with-default-bayadera]]]
            [uncomplicate.bayadera.opencl.models
             :refer [distributions likelihoods]]
            [uncomplicate.bayadera.toolbox
             [processing :refer :all]
             [scaling :refer [axis vector-axis]]
             [plots :refer [render-sample]]]
            [clojure.java.io :as io]))

(def all-data (atom {}))
(def plots (atom nil))

(defn analysis []
  (with-default-bayadera
    (let [sample-count (* 256 44 16)
          a 1 b 1
          z 15 N 50]
      (with-release [prior-dist (beta a b)
                     prior-sample (dataset (sample! (sampler prior-dist) sample-count))
                     prior-pdf (pdf prior-dist prior-sample)
                     post (posterior (posterior-model (:binomial likelihoods) (:beta distributions)))
                     post-dist (post (binomial-lik-params N z) (beta-params a b))
                     post-sampler (time (doto (sampler post-dist) (mix!)))
                     post-sample (dataset (sample! post-sampler sample-count))
                     post-pdf (scal! (/ 1.0 (evidence post-dist prior-sample))
                                     (pdf post-dist post-sample))]
        (println (uncomplicate.bayadera.mcmc/info post-sampler))
        {:prior {:sample (native (row (p/data prior-sample) 0))
                 :pdf (native prior-pdf)}
         :posterior {:sample (native (row (p/data post-sample) 0))
                     :pdf (native post-pdf)}}))))

(defn setup []
  (reset! plots
          {:data @all-data
           :prior (plot2d (qa/current-applet) {:width 1000 :height 700})
           :posterior (plot2d (qa/current-applet) {:width 1000 :height 700})}))

(defn draw []
  (when-not (= @all-data (:data @plots))
    (swap! plots assoc :data @all-data)
    (q/background 0)
    (q/image (show (render (:prior @plots)
                           {:x-axis (axis 0 1) :x (:sample (:prior @all-data))
                            :y-axis (axis 0 2) :y (:pdf (:prior @all-data))})) 0 0)
    (q/image (show (render-sample (:posterior @plots)
                                  (:sample (:posterior @all-data))
                                  (:pdf (:posterior @all-data)))) 0 720)))

(defn display-sketch []
  (q/defsketch diagrams
    :renderer :p2d
    :size :fullscreen
    :display 2
    :setup setup
    :draw draw
    :middleware [pause-on-error]))
