;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.examples.dbda.ch19.sex-and-death-test
  (:require [midje.sweet :refer :all]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares.pause-on-error :refer [pause-on-error]]
            [uncomplicate.commons.core :refer [with-release let-release wrap-float]]
            [uncomplicate.fluokitten.core :refer [op fmap]]
            [uncomplicate.neanderthal
             [core :refer [dim row subvector sum]]
             [real :refer [entry entry!]]
             [math :refer [sqrt]]
             [native :refer [fv fge]]]
            [uncomplicate.bayadera
             [core :refer :all]
             [util :refer [bin-mapper hdi]]
             [opencl :refer [with-default-bayadera]]
             [mcmc :refer [mix! burn-in! pow-n acc-rate! run-sampler!]]]
            [uncomplicate.bayadera.internal.models
             :refer [source-library  cl-likelihood-model cl-distribution-model]]
            [uncomplicate.bayadera.toolbox
             [processing :refer :all]
             [plots :refer [render-sample render-histogram]]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]))

(def all-data (atom {}))
(def state (atom nil))

(defn read-data [in-file]
  (loop [c 0 data (drop 1 (csv/read-csv in-file)) acc (transient [])]
    (if data
      (let [[longevity group thorax] (first data)]
        (recur (inc c) (next data)
               (-> acc
                   (conj! (case group
                            "None0" 0
                            "Pregnant1" 1
                            "Pregnant8" 2
                            "Virgin1" 3
                            "Virgin8" 4))
                   (conj! (double (read-string thorax)))
                   (conj! (double (read-string longevity))))))
      (fv (op [c] (persistent! acc))))))

(def ff-data (read-data (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch19/fruitfly-data-reduced.csv"))))
(def ff-matrix (fge 3 (first ff-data) (drop 1 ff-data)))
(def y-sd (double (sd (row ff-matrix 2))))
(def y-mean (double (mean (row ff-matrix 2))))
(def x-sd (double (sd (row ff-matrix 1))))
(def params (fv (op [(mean (row ff-matrix 1))] ff-data)))

(def ff-prior
  (cl-distribution-model [(:gaussian source-library)
                          (:uniform source-library)
                          (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch19/fruitfly.h"))]
                         :name "ff" :mcmc-logpdf "ff_mcmc_logpdf" :params-size 11 :dimension 8))

(defn ff-likelihood [n]
  (cl-likelihood-model (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch19/fruitfly.h"))
                       :name "ff" :params-size n))

(defn analysis []
  (with-default-bayadera
    (with-release [prior (distribution ff-prior)
                   prior-dist (prior (fv (op [(/ y-sd 100.0) (* y-sd 10.0)
                                              y-mean (* y-sd 5)]
                                             (vec (repeat 5 (* y-sd 5)))
                                             [0 (/ (* 2 y-sd) x-sd)])))
                   prior-sampler (sampler prior-dist
                                          {:limits (fge 2 8 (op [0 150 0 100]
                                                                (vec (take 10 (cycle [-20 50])))
                                                                [-500 500]))})
                   post (posterior "ff" (ff-likelihood  (dim params)) prior-dist)
                   post-dist (post params)
                   post-sampler (sampler post-dist {:position prior-dist
                                                    :limits (fge 2 8 (op [0 150 0 100]
                                                                         (vec (take 10 (cycle [-20 50])))
                                                                         [-500 500]))})]
      (println (time (mix! prior-sampler)))
      (println (time (mix! post-sampler {:cooling-schedule (pow-n 1.5)})))
      (time (histogram! post-sampler 100)))))

#_(defn setup []
  (reset! state
          {:data @all-data
           :nu (plot2d (qa/current-applet) {:width 500 :height 500})
           :sigma (plot2d (qa/current-applet) {:width 500 :height 500})
           :b0 (plot2d (qa/current-applet) {:width 500 :height 500})
           :b1 (plot2d (qa/current-applet) {:width 500 :height 500})
           :b2 (plot2d (qa/current-applet) {:width 500 :height 500})}))

#_(defn draw []
  (when-not (= @all-data (:data @state))
    (swap! state assoc :data @all-data)
    (let [data @all-data]
      (q/background 0)
      (q/image (show (render-histogram (:nu @state) data 0)) 0 0)
      (q/image (show (render-histogram (:sigma @state) data 1)) 520 0)
      (q/image (show (render-histogram (:b0 @state) data 2)) 0 520)
      (q/image (show (render-histogram (:b1 @state) data 3)) 0 1040)
      (q/image (show (render-histogram (:b1 @state) data 4)) 520 1040))))

#_(defn display-sketch []
  (q/defsketch diagrams
    :renderer :p2d
    :size :fullscreen
    :display 2
    :setup setup
    :draw draw
    :middleware [pause-on-error]))
