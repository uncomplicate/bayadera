;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.examples.dbda.ch09.single-coin-test
  (:require [midje.sweet :refer :all]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares.pause-on-error :refer [pause-on-error]]
            [uncomplicate.commons.core :refer [with-release let-release wrap-float]]
            [uncomplicate.neanderthal
             [core :refer [row native dot imax imin scal! col submatrix]]
             [real :refer [entry entry!]]
             [native :refer [fv fge]]]
            [uncomplicate.bayadera
             [core :refer :all]
             [distributions :refer [binomial-lik-params]]
             [util :refer [bin-mapper hdi]]
             [opencl :refer [with-default-bayadera]]
             [mcmc :refer [mix! info anneal! burn-in! acc-rate! run-sampler!]]]
            [uncomplicate.bayadera.internal.protocols :as p]
            [uncomplicate.bayadera.internal.models
             :refer [likelihoods source-library cl-distribution-model]]
            [uncomplicate.bayadera.toolbox
             [processing :refer :all]
             [plots :refer [render-sample render-histogram]]]
            [clojure.java.io :as io]))

(def all-data (atom {}))
(def state (atom nil))

(def single-coin-model
  (cl-distribution-model [(:beta source-library)
                          (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch09/single-coin.h"))]
                         :name "single_coin" :params-size 3 :dimension 2 :limits (fge 2 2 [0 1 0 1])))

(defn analysis []
  (with-default-bayadera
    (let [walker-count (* 256 44 32)
          sample-count (* 16 walker-count)
          z 9 N 12]
      (with-release [prior (distribution single-coin-model)
                     prior-dist (prior (fv 2 2 100))
                     prior-sampler (time (doto (sampler prior-dist) (mix! {:a 2.68})))
                     prior-sample (dataset (sample! prior-sampler sample-count))
                     prior-pdf (pdf prior-dist prior-sample)
                     post (posterior "posterior" (:binomial likelihoods) prior-dist)
                     post-dist (post (fv (binomial-lik-params N z)))
                     post-sampler (time (doto (sampler post-dist) (mix!)))
                     post-sample (dataset (sample! post-sampler sample-count))
                     post-pdf (scal! (/ 1.0 (evidence post-dist prior-sample))
                                     (pdf post-dist post-sample))]

        {:prior {:sample (native (submatrix (p/data prior-sample) 0 0 2 walker-count))
                 :pdf (native prior-pdf)
                 :histogram (histogram! prior-sampler 100)}
         :posterior {:sample (native (submatrix (p/data post-sample) 0 0 2 walker-count))
                     :pdf (native post-pdf)
                     :histogram (time (histogram! post-sampler 100))}}))))

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
  (q/image (show (render-histogram omega (:histogram data) 1 :rotate))
           (+ x-position 20 (width scatterplot)) y-position)
  (q/image (show (render-histogram theta (:histogram data) 0))
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
    :setup setup
    :draw draw
    :middleware [pause-on-error]))
