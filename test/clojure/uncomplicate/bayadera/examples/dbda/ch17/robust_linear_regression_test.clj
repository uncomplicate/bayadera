(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.examples.dbda.ch17.robust-linear-regression-test
  (:require [midje.sweet :refer :all]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares.pause-on-error :refer [pause-on-error]]
            [uncomplicate.commons.core :refer [with-release let-release wrap-float]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.clojurecl.core :refer [finish!]]
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
             [mcmc :refer [mix! burn-in! pow-n acc-rate!]]]
            [uncomplicate.bayadera.opencl.models
             :refer [source-library cl-distribution-model cl-likelihood-model]]
            [uncomplicate.bayadera.toolbox
             [processing :refer :all]
             [plots :refer [render-sample render-histogram]]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]))

(def all-data (atom {}))
(def state (atom nil))

(defn read-data [in-file]
  (loop [c 0 data (drop 1 (csv/read-csv in-file)) hw (transient [])]
    (if data
      (let [[_ h w] (first data)]
        (recur (inc c) (next data)
               (-> hw
                   (conj! (double (read-string h)))
                   (conj! (double (read-string w))))))
      (op [c] (persistent! hw)))))

(def params-30 (sv (read-data (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch17/ht-wt-data-30.csv")))))
(def params-300 (sv (read-data (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch17/ht-wt-data-300.csv")))))

(def rlr-prior
  (cl-distribution-model [(:gaussian source-library)
                          (:uniform source-library)
                          (:exponential source-library)
                          (:t source-library)
                          (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch17/robust-linear-regression.h"))]
                         :name "rlr" :mcmc-logpdf "rlr_mcmc_logpdf" :params-size 7 :dimension 4))

(defn rlr-likelihood [n]
  (cl-likelihood-model (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch17/robust-linear-regression.h"))
                       :name "rlr" :params-size n))

(defn analysis []
  (with-default-bayadera
    (with-release [prior (distribution rlr-prior)
                   prior-dist (prior (sv 10 -100 100 5 10 0.001 1000))
                   prior-sampler (sampler prior-dist {:walkers 22528 :limits (sge 2 4 [1 20 -400 100 0 20 0.01 100])})
                   post-30 (posterior "rlr_30" (rlr-likelihood (dim params-30)) prior-dist)
                   post-30-dist (post-30 params-30)
                   post-30-sampler (sampler post-30-dist {:limits (sge 2 4 [1 20 -400 100 0 20 0.01 100])})
                   post-300 (posterior "rlr_300" (rlr-likelihood (dim params-300)) prior-dist)
                   post-300-dist (post-300 params-300)
                   post-300-sampler (sampler post-300-dist {:limits (sge 2 4 [1 10 -400 100 0 20 0.01 100])})]
      (println (time (mix! post-30-sampler {:step 128})))
      (println (time (mix! post-300-sampler {:step 384})))
      (println (uncomplicate.bayadera.mcmc/info post-300-sampler))
      [(histogram! post-30-sampler 1000)
       (histogram! post-300-sampler 1000)])))

(defn setup []
  (reset! state
          {:data @all-data
           :nu-30 (plot2d (qa/current-applet) {:width 350 :height 350})
           :b0-30 (plot2d (qa/current-applet) {:width 350 :height 350})
           :b1-30 (plot2d (qa/current-applet) {:width 350 :height 350})
           :sigma-30 (plot2d (qa/current-applet) {:width 350 :height 350})
           :nu-300 (plot2d (qa/current-applet) {:width 350 :height 350})
           :b0-300 (plot2d (qa/current-applet) {:width 350 :height 350})
           :b1-300 (plot2d (qa/current-applet) {:width 350 :height 350})
           :sigma-300 (plot2d (qa/current-applet) {:width 350 :height 350})}))

(defn draw []
  (when-not (= @all-data (:data @state))
    (swap! state assoc :data @all-data)
    (q/background 0)
    (q/image (show (render-histogram (:nu-30 @state) (@all-data 0) 0)) 0 0)
    (q/image (show (render-histogram (:b0-30 @state) (@all-data 0) 1)) 0 370)
    (q/image (show (render-histogram (:b1-30 @state) (@all-data 0) 2)) 0 740)
    (q/image (show (render-histogram (:sigma-30 @state) (@all-data 0) 3)) 0 1110)
    (q/image (show (render-histogram (:nu-300 @state) (@all-data 1) 0)) 370 0)
    (q/image (show (render-histogram (:b0-300 @state) (@all-data 1) 1)) 370 370)
    (q/image (show (render-histogram (:b1-300 @state) (@all-data 1) 2)) 370 740)
    (q/image (show (render-histogram (:sigma-300 @state) (@all-data 1) 3)) 370 1110)))

(defn display-sketch []
  (q/defsketch diagrams
    :renderer :p2d
    :size :fullscreen
    :display 2
    :setup setup
    :draw draw
    :middleware [pause-on-error]))
