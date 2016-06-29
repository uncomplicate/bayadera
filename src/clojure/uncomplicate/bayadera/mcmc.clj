(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.mcmc
  (:require [uncomplicate.neanderthal
             [core :refer [iamax]]
             [real :refer [entry]]]
            [uncomplicate.bayadera
             [protocols :as p]
             [util :refer [srand-int]]]))

(defn init-position!
  ([samp position]
   (p/init-position! samp position))
  ([samp]
   (p/init-position! samp (srand-int))))

(defn acc-rate!
  (^double [samp a]
   (double (p/acc-rate! samp a)))
  (^double [samp]
   (double (p/acc-rate! samp 2.0))))

(defn burn-in!
  ([samp ^long steps ^double a]
   (p/burn-in! samp steps a))
  ([samp ^long steps]
   (p/burn-in! samp steps 2.0))
  ([samp]
   (p/burn-in! samp (* 256 (long (p/dimension (p/model samp)))) 2.0)))

(defn run-sampler!
  ([samp ^long steps ^double a]
   (p/run-sampler! samp steps a))
  ([samp ^long steps]
   (p/run-sampler! samp steps 2.0))
  ([samp]
   (p/run-sampler! samp (* 64 (long (p/dimension (p/model samp)))) 2.0)))

(defn ^:private sqrt-n [^double temp]
  (fn ^double [^long i]
    (Math/sqrt (- temp i))))

(defn ^:private minus-n [^double temp]
  (fn ^double [^long i]
    (- temp i)))

(defn anneal!
  ([samp schedule ^long steps ^double a]
   (p/anneal! samp (schedule steps) steps a))
  ([samp ^long steps ^double a]
   (p/anneal! samp (minus-n steps) steps a))
  ([samp ^long steps]
   (anneal! samp steps 2.0))
  ([samp]
   (anneal! samp (* 256 (long (p/dimension (p/model samp)))) 2.0)))

(defn mix!
  ([samp options]
   (let [{step :step
          run-step :run-step
          max-iterations :max-iterations
          position :position
          schedule :cooling-schedule
          a :a
          min-acc :min-acc-rate
          max-acc :max-acc-rate
          acor-mult :acor-mult
          :or {step 64
               run-step 64
               max-iterations 1048576
               schedule minus-n
               position (srand-int)
               a 2.0
               min-acc 0.3
               max-acc 0.5
               acor-mult 10.0}} options
         a (double a)
         min-acc (double min-acc)
         max-acc (double max-acc)
         max-iterations (long max-iterations)
         dimension (long (p/dimension (p/model samp)))
         step (* (long step) dimension)
         run-step (* (long run-step) dimension)]
     (anneal! samp schedule step a)
     (burn-in! samp step a)
     (let [a (loop [i 0 acc-rate (acc-rate! samp a) a a]
               (if (and (< i max-iterations) (not (< min-acc acc-rate max-acc)))
                 (let [new-a (/ (* a acc-rate) (if (< acc-rate min-acc) min-acc max-acc))]
                   (anneal! samp schedule step new-a)
                   (burn-in! samp step new-a)
                   (recur (+ i step) (acc-rate! samp a) new-a))
                 a))]
       (burn-in! samp step a)
       (loop [i 0 step run-step ac (run-sampler! samp step a)]
         (let [tau (:tau (:autocorrelation ac))
               tau-fraction (/ (double step) (double acor-mult)
                               (entry tau (iamax tau)))]
           (if (and (< i max-iterations) (< tau-fraction 1.0))
             (recur (+ i step) (inc (long (/ step tau-fraction)))
                    (run-sampler! samp step a))
             ac))))))
  ([samp]
   (mix! samp nil)))

(defn info [samp]
  (p/info samp))
