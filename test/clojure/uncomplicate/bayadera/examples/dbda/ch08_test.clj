(ns uncomplicate.bayadera.examples.dbda.ch08-test
  (:require [midje.sweet :refer :all]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares
             [pause-on-error :refer [pause-on-error]]
             [fun-mode :refer [fun-mode]]]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [math :refer [log exp]]
             [core :refer [dim sum nrm2 fmap! copy transfer! dot entry imax imin scal!]]
             [native :refer [sv]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [core :refer :all]
             [distributions :refer [beta-pdf]]
             [impl :refer :all]
             [math :refer [log-beta]]
             [visual :refer :all]]
            [uncomplicate.bayadera.opencl :refer [with-default-bayadera]]
            [uncomplicate.bayadera.opencl
             [generic :refer [binomial-likelihood beta-model]]]
            [clojure.java.io :as io])
  (:import [uncomplicate.neanderthal.opencl.clblock CLBlockVector]
           [uncomplicate.bayadera.impl UnivariateDataSet]))

(defn render-sample
  ([plot xs ps x-min x-max]
   (let [imax-p (imax ps)]
     (render plot {:x-axis (axis x-min x-max) :x xs
                   :y-axis (axis 0 10) :y ps
                   :vertical-lines [[(entry xs imax-p) (entry ps imax-p)]]}))))

(defmulti plot-distribution
  (fn [plot xs ps options]
    [(class xs) (class ps)]))

(defmethod plot-distribution [uncomplicate.bayadera.impl.UnivariateDataSet CLBlockVector]
  [plot sample-xs ps options]
  (let [xs (p/data sample-xs)]
    (with-release [host-xs (transfer! xs (sv (dim xs)))
                   host-ps (transfer! ps (sv (dim ps)))]
      (render-sample plot host-xs host-ps 0 1
                     #_(entry host-xs (imin xs))
                     #_(entry host-xs (imax xs))))))

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

        (plot-distribution prior-plot prior-sample prior-pdf {})
        (plot-distribution posterior-plot post-sample post-pdf {})))))

(defn setup []
  (reset! plots
          {:prior (plot2d (qa/current-applet) {:width 1000 :height 700})
           :posterior (plot2d (qa/current-applet) {:width 1000 :height 700})}))

(defn draw []
  (when (:changed @plots)
    (do
      (q/background 0)
      (analysis (:prior @plots) (:posterior @plots))
      (q/image  (-> (:prior @plots) (show)) 0 0)
      (q/image  (-> (:posterior @plots) (show)) 0 720)
      (reset! plots (assoc @plots :changed false)))))

(defn display-sketch []
  (q/defsketch diagrams
    :renderer :p2d
    :size :fullscreen
    :display 2
    :setup setup
    :draw draw
    :middleware [pause-on-error]))
