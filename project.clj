;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject uncomplicate/bayadera "0.2.0-SNAPSHOT"
  :description "Bayesian Inference and Probabilistic Machine Learning Library for Clojure"
  :author "Dragan Djuric"
  :url "http://github.com/uncomplicate/bayadera"
  :scm {:name "git"
        :url "https://github.com/uncomplicate/bayadera"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [uncomplicate/commons "0.4.0-SNAPSHOT"]
                 [uncomplicate/fluokitten "0.6.1-SNAPSHOT"]
                 [uncomplicate/clojurecl "0.8.0-SNAPSHOT"]
                 [uncomplicate/clojurecuda "0.3.0-SNAPSHOT"]
                 [uncomplicate/neanderthal "0.18.0-SNAPSHOT"]
                 [org.apache.commons/commons-math3 "3.6.1"]
                 [quil "2.6.0"]]

  :codox {:src-dir-uri "http://github.com/uncomplicate/bayadera/blob/master"
          :src-linenum-anchor-prefix "L"
          :output-dir "docs/codox"}

  :profiles {:dev {:dependencies [[midje "1.9.1"]
                                  [org.clojure/data.csv "0.1.4"]]
                   :plugins [[lein-midje "3.2.1"]
                             [codox "0.10.3"]]
                   :global-vars {*warn-on-reflection* true
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 16}
                   :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                        "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"
                                        #_"--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"]}}

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]
  :source-paths ["src/clojure" "src/device"]
  :resource-paths ["src/device"]
  :test-paths ["test" "test/clojure"])
