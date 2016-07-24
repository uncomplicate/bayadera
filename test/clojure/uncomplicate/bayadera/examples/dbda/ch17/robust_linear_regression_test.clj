(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.examples.dbda.ch16.robust-linear-regression-test
  (:require [midje.sweet :refer :all]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares.pause-on-error :refer [pause-on-error]]
            [uncomplicate.commons.core :refer [with-release let-release wrap-float]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.neanderthal
             [core :refer [dim]]
             [real :refer [entry entry!]]
             [math :refer [sqrt]]
             [native :refer [sv sge]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [core :refer :all]
             [util :refer [bin-mapper hdi]]
             [opencl :refer [with-default-bayadera]]
             [mcmc :refer [mix!]]]
            [uncomplicate.bayadera.opencl.models
             :refer [cl-distribution-model cl-likelihood-model]]
            [uncomplicate.bayadera.toolbox
             [processing :refer :all]
             [plots :refer [render-sample render-histogram]]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]))

(def all-data (atom {}))
(def state (atom nil))

(let [in-file (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch17/ht-wt-data-30.csv"))]
  (let [data (loop [c 0 data (drop 1 (csv/read-csv in-file)) hw (transient [])]
               (if data
                 (let [[_ h w] (first data)]
                   (recur (inc c) (next data)
                          (-> hw
                              (conj! (double (read-string h)))
                              (conj! (double (read-string w))))))
                 (op [c] (persistent! hw))))]
    (def params (sv data))))

(def rlr-prior
  (cl-distribution-model [(slurp (io/resource "uncomplicate/bayadera/opencl/distributions/gaussian.h"))
                          (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/uniform.h"))
                          (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/exponential.h"))
                          (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/t.h"))
                          (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch17/robust-linear-regression.h"))]
                         :name "rlr" :mcmc-logpdf "rlr_mcmc_logpdf" :params-size 7 :dimension 4))

(defn rlr-likelihood [n]
  (cl-likelihood-model (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch17/robust-linear-regression.h"))
                       :name "rlr" :params-size n))

(defn analysis []
  (with-default-bayadera
    (with-release [prior (distribution rlr-prior)
                   prior-dist (prior (sv 5 -200 100 2 4 0.001 1000))
                   prior-sampler (sampler prior-dist {:limits (sge 2 4 [1 10 -400 100 -10 10 0.001 1000])})
                   post (posterior "rlr_30" (rlr-likelihood (dim params)) prior-dist)
                   post-dist (post params)
                   post-sampler (sampler post-dist {:limits (sge 2 4 [1 10 -400 100 -10 10 0.001 1000])})]
      (println (time (mix! prior-sampler {:step 256})))
      (println (time (mix! post-sampler {:step 256})))
      (histogram! post-sampler 100))))

(defn setup []
  (reset! state
          {:data @all-data
           :nu (plot2d (qa/current-applet) {:width 500 :height 500})
           :b0 (plot2d (qa/current-applet) {:width 500 :height 500})
           :b1 (plot2d (qa/current-applet) {:width 500 :height 500})
           :sigma (plot2d (qa/current-applet) {:width 500 :height 500})}))

(defn draw []
  (when-not (= @all-data (:data @state))
    (swap! state assoc :data @all-data)
    (q/background 0)
    (q/image (show (render-histogram (:nu @state) @all-data 0)) 0 0)
    (q/image (show (render-histogram (:b0 @state) @all-data 1)) 0 520)
    (q/image (show (render-histogram (:b1 @state) @all-data 2)) 0 1040)
    (q/image (show (render-histogram (:sigma @state) @all-data 3)) 520 0)))

(defn display-sketch []
  (q/defsketch diagrams
    :renderer :p2d
    :size :fullscreen
    :display 2
    :setup setup
    :draw draw
    :middleware [pause-on-error]))
