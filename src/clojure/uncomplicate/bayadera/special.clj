(ns uncomplicate.bayadera.special
  (:require [uncomplicate.neanderthal
             [core :refer [asum freduce]]
             [math :refer :all]]))

(let [magicn 5.2421875
      cs (double-array [57.156235665862923517
                        -59.597960355475491248
                        14.136097974741747174
                        -0.49191381609762019978
                        0.33994649984811888699e-4
                        0.46523628927048575665e-4
                        -0.98374475304879564677e-4
                        0.15808870322491248884e-3
                        -0.21026444172410488319e-3
                        0.21743961811521264320e-3
                        -0.16431810653676389022e-3
                        0.84418223983852743293e-4
                        -0.26190838401581408670e-4
                        0.36899182659531622704e-5])
      comp-cs (fn ^double [^double d]
                (loop [i (dec (alength cs)) acc 0.99999999999999709182]
                  (if (< i 0)
                    acc
                    (recur (dec i) (+ acc (/ (aget cs i) (+ d (inc i))))))))]
  (defn lngamma
    "Computes the logarithm of the gammma function for a positive real number r.
     See: http://en.wikipedia.org/wiki/Gamma_function

     Uses the Lanczos approximation with g=4.7421875, n=15, the parameters
     described in http://mrob.com/pub/ries/lanczos-gamma.html"
    ^double [^double x]
    (if (< x 0)
      (throw (IllegalArgumentException.
              (format "x have to be positive, but is %f." x)))
      (+ (- (* (+ x 0.5)
               (log (+ x magicn)))
            (+ x magicn))
         (log (/ (* 2.5066282746310005
                    (double (comp-cs x)))
                 x))))))

(defn gamma
  "Gamma function: http://en.wikipedia.org/wiki/Gamma_function"
  ^double [^double x]
  (exp (lngamma x)))

(let [table-size (double 1000)
      lnfactorial-table (double-array
                         (map (fn [^double x]
                                (lngamma (inc x)))
                              (range 0 (inc table-size))))]
  (defn lnfactorial
    "Natural logarithm of a factorial of a positive real x."
    ^double [^double x]
    (if (< x 0)
      (throw (IllegalArgumentException.
              (format "x have to be positive, but is %f." x)))
      (if (and (< x table-size) (round? x))
        (aget lnfactorial-table x)
        (lngamma (inc x))))))

(let [factorial-table (double-array
                       (reduce (fn [acc ^double x]
                                 (conj acc (* (double (peek acc)) x)))
                               [1.0]
                               (range 1 171)))]
  (defn factorial
  "Factorial function:
  Computes a product of all natural numbers from 0 to x.
  If x is a real number, computes the exponent of lnfactorial."
  ^double [^double x]
  (if (and (<= 0 x 170) (round? x))
    (aget factorial-table x)
    (exp (lnfactorial x)))))

(defn lnbeta
  "Natural logarithm of beta function."
  ^double [^double a ^double b]
  (- (+ (lngamma a) (lngamma b))
       (lngamma (+ a b))))

(defn beta
  "Beta function of a and b."
  ^double [^double a ^double b]
  (exp (lnbeta a b)))

(defn binco
  "Binomial coefficient of n and k:
  Computes the number of ways to choose k items out of n items."
  ^double [^double n ^double k]
  (if (and (<= 0 n 170) (round? n))
      (/ (factorial n)
         (factorial k) (factorial (- n k)))
      (exp (- (lnfactorial n)
              (lnfactorial k) (lnfactorial (- n k))))))

(defn multico
  "Multinomial coefficient of a RealVector xks:
  Computes the number of ways of partitioning N object
  into k groups of size x1, x2, ... , xk;
  where k = (dim xks) and N = (sum xks)"
  ^double [xks]
  (exp (freduce (fn ^double [^double acc ^double xk]
                  (- acc (lnfactorial xk)))
                (lnfactorial (asum xks))
                xks)))

(let [cheb-coef (double-array [-1.3026537197817094
                               6.4196979235649026E-1
                               1.9476473204185836E-2
                               -9.561514786808631E-3
                               -9.46595344482036E-4
                               3.66839497852761E-4
                               4.2523324806907E-5
                               -2.0278578112534E-5
                               -1.624290004647E-6
                               1.303655835580E-6
                               1.5626441722E-8
                               -8.5238095915E-8
                               6.529054439E-9
                               5.059343495E-9
                               -9.91364156E-10
                               -2.27365122E-10
                               9.6467911E-11
                               2.394038E-12
                               -6.886027E-12
                               8.94487E-13
                               3.13092E-13
                               -1.12708E-13
                               3.81E-16
                               7.106E-15
                               -1.523E-15
                               -9.4E-17
                               1.21E-16
                               -2.8E-17])
      erfc* (fn ^double [^double x]
              (let [t (/ 2.0 (+ 2.0 x))
                    ty (- (* 4.0 t) 2.0)]
                (loop [i 27 d 0.0 dd 0.0]
                  (if (= 0 i)
                    (* t (exp (+ (- (* x x))
                                 (* 0.5 (- (* ty d) 1.3026537197817094))
                                 (- dd))))
                    (recur (dec i) (+ (aget cheb-coef i) (* ty d) (- dd)) d)))))]

  (defn erfc
    "The complementary error function: erfc(x) = 1 - erf(x)"
    ^double [^double x]
    (if (<= 0.0 x)
      (erfc* x)
      (- 2.0 (double (erfc* (- x))))))

  (defn erf
  "Error function: erf(x) = 2/√π 0∫x e-t2dt"
    ^double [^double x]
    (- 1.0 (erfc x))))
