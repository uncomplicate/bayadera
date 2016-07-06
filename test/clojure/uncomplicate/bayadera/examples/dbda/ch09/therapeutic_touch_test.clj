(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.examples.dbda.ch09.therapeutic-touch-test
  (:require [midje.sweet :refer :all]
            [quil.core :as q]
            [quil.applet :as qa]
            [quil.middlewares.pause-on-error :refer [pause-on-error]]
            [uncomplicate.commons.core :refer [with-release let-release wrap-float]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.neanderthal
             [core :refer [row native dot imax imin scal! col submatrix transfer]]
             [real :refer [entry entry!]]
             [native :refer [sv sge]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [core :refer :all]
             [util :refer [bin-mapper hdi]]
             [opencl :refer [with-default-bayadera]]
             [mcmc :refer [mix! info anneal! burn-in! acc-rate! run-sampler!]]]
            [uncomplicate.bayadera.opencl.models
             :refer [cl-likelihood-model cl-distribution-model]]
            [uncomplicate.bayadera.toolbox
             [processing :refer :all]
             [plots :refer [render-sample render-histogram]]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]))

(def all-data (atom {}))
(def state (atom nil))

(def subjects 28)

(def touch-prior
  (cl-distribution-model [(slurp (io/resource "uncomplicate/bayadera/opencl/distributions/beta.h"))
                          (slurp (io/resource "uncomplicate/bayadera/opencl/distributions/gamma.h"))
                          (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch09/therapeutic-touch.h"))]
                         :name "touch" :params-size 4 :dimension (+ subjects 2)
                         :limits (sge 2 (+ subjects 2)
                                      (op (take (+ 2 (* subjects 2))
                                                (interleave (repeat 0) (repeat 1)))
                                          [0 50]))))

(def touch-likelihood
  (cl-likelihood-model [(slurp (io/resource "uncomplicate/bayadera/opencl/distributions/binomial.h"))
                        (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch09/therapeutic-touch-lik.h"))]
                       :name "touch" :params-size (* subjects 2)))

(let [in-file (slurp (io/resource "uncomplicate/bayadera/examples/dbda/ch09/therapeutic-touch-data.csv"))]
  (def params (sv (seq (reduce (fn [^ints acc [b c]]
                                 (let [c (* 2 (dec (int (bigint (subs c 1)))))]
                                   (aset acc c (inc (aget acc c) ))
                                   (aset acc (inc c) (+ (aget acc (inc c)) (int (read-string b))))
                                   acc))
                               (int-array 56)
                               (drop 1 (csv/read-csv in-file)))))))

(defn analysis []
  (with-default-bayadera
    (let [walker-count (* 256 44 2)]
      (with-release [prior (distribution touch-prior)
                     prior-dist (prior (sv 1 1 1.105125 1.105125))
                     post (posterior "touch" touch-likelihood prior-dist)
                     post-dist (post (sv (take (* subjects 2) params)))
                     post-sampler (sampler post-dist {:walkers walker-count})]
        ;;(println (time (mix! post-sampler {:step 256 :max-iterations 10000})))
        (time (do (anneal! post-sampler 5000 1.3) (uncomplicate.clojurecl.core/finish!)))
        (println (acc-rate! post-sampler 1.4))
        (time (do (burn-in! post-sampler 1000 1.3) (uncomplicate.clojurecl.core/finish!)) )
        (println (acc-rate! post-sampler 1.3))
        (time (do (anneal! post-sampler 1000 1.2) (uncomplicate.clojurecl.core/finish!)))
        (println (acc-rate! post-sampler 1.2))
        ;;(time (do (anneal! post-sampler 100 1.05) (uncomplicate.clojurecl.core/finish!)))
        (time (do (burn-in! post-sampler 200 1.2) (uncomplicate.clojurecl.core/finish!)))
        ;;(println (time (acc-rate! post-sampler 1.2)))
        (println (time (run-sampler! post-sampler 64 1.2)))
        (println (info post-sampler))
        (histogram! post-sampler 320)))))

(defn setup []
  (reset! state
          {:data @all-data
           :omega (plot2d (qa/current-applet) {:width 300 :height 300})
           :kappa-2 (plot2d (qa/current-applet) {:width 300 :height 300})
           :thetas (vec (repeatedly 28 (partial plot2d (qa/current-applet)
                                                {:width 180 :height 180})))}))

(defn draw []
  (when-not (= @all-data (:data @state))
    (swap! state assoc :data @all-data)
    (let [data @all-data]
      (q/background 0)
      (q/image (show (render-histogram (:omega @state) data subjects)) 0 0)
      (q/image (show (render-histogram (:kappa-2 @state) data (inc subjects))) 350 0)
      (dotimes [i 6]
        (dotimes [j 5]
          (let [index (+ (* i 5) j)]
            (when (< index 28)
              (q/image (show (render-histogram ((:thetas @state) index) data index))
                       (* j 200) (+ 320 (* i 200))))))))))

(defn display-sketch []
  (q/defsketch diagrams
    :renderer :p2d
    :size :fullscreen
    :display 2
    :setup setup
    :draw draw
    :middleware [pause-on-error]))
