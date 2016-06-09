(ns ^{:author "Dragan Djuric"}
    uncomplicate.bayadera.toolbox.theme)

;; ============= Color mapping functions =================================

(definterface Colormap
  (^float r [^float x])
  (^float g [^float x])
  (^float b [^float x]))

(defmacro ^:private cube-helix-color [p0 p1 gamma s r h x]
  `(let [xg# (Math/pow ~x ~gamma)
         a# (* ~h xg# (- 1 xg#) 0.5)
         phi# (* 2 Math/PI (+ (/ ~s 3) (* ~r ~x)))]
     (* (float 255.0) (+ xg# (* a# (+ (* ~p0 (Math/cos phi#)) (* ~p1 (Math/sin phi#))))))))

(deftype CubeHelix [^float gamma ^float start-color ^float rotations ^float hue]
  Colormap
  (r [_ x]
    (cube-helix-color (float -0.14861) (float 1.78277)
                      gamma start-color rotations hue x))
  (g [_ x]
    (cube-helix-color (float -0.29227) (float -0.90649)
                      gamma start-color rotations hue x))
  (b [_ x]
    (cube-helix-color (float 1.97294) (float 0.0)
                      gamma start-color rotations hue x)))

(defn cube-helix
  ([^double gamma ^double start-color ^double rotations ^double hue]
   (CubeHelix. gamma start-color rotations hue))
  ([]
   (cube-helix 1.0 0.5 -1.5 1.0)))

(definterface RGBColor
  (^float r [])
  (^float g [])
  (^float b []))

(deftype ConstantColor [^float red ^float green ^float blue]
  RGBColor
  (r [_] red)
  (g [_] green)
  (b [_] blue)
  Colormap
  (r [_ x] red)
  (g [_ x] green)
  (b [_ x] blue))

(defn rgb-color [^double red ^double green ^double blue]
  (ConstantColor. red green blue))

(defn hsb-color [^double h ^double s ^double b]
  (let [color (java.awt.Color/getHSBColor (/ h 360.0) (/ s 100.0) (/ b 100.0))]
    (ConstantColor. (.getRed color) (.getGreen color) (.getBlue color))))

(defn red
  ([^RGBColor c]
   (.r c))
  ([^Colormap cm ^double x]
   (.r cm x)))

(defn green
  ([^RGBColor c]
   (.g c))
  ([^Colormap cm ^double x]
   (.g cm x)))

(defn blue
  ([^RGBColor c]
   (.b c))
  ([^Colormap cm ^double x]
   (.b cm x)))

;; ============= Styles and themes ========================================

(defrecord Style [^RGBColor color ^float weight])
(defrecord Theme [^Style frame ^Style ticks ^Style labels ^Style grid
                  ^Style data ^Colormap colormap])

(let [frame-style (->Style (hsb-color 200 50 60) 1)
      data-style (->Style (hsb-color 320 100 100) 2)
      grid-style (->Style (hsb-color 60 30 10) 1)
      label-style (->Style (hsb-color 180 40 100) 10)]
  (def cyberpunk-theme
    (->Theme frame-style frame-style label-style grid-style data-style (cube-helix))))
