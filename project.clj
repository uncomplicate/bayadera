;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject uncomplicate/bayadera "0.1.0-SNAPSHOT"
  :description "Bayesian Inference and Probabilistic Machine Learning Library for Clojure"
  :author "Dragan Djuric"
  :url "http://github.com/uncomplicate/bayadera"
  :scm {:name "git"
        :url "https://github.com/uncomplicate/bayadera"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [uncomplicate/commons "0.3.0"]
                 [uncomplicate/fluokitten "0.6.0"]
                 [uncomplicate/clojurecl "0.7.1"]
                 [uncomplicate/neanderthal "0.9.1-SNAPSHOT"]
                 [org.apache.commons/commons-math3 "3.6.1"]
                 [me.raynes/fs "1.4.6"]
                 [quil "2.5.0"]]

  :aot [;;uncomplicate.bayadera.protocols
                                        ;uncomplicate.bayadera.impl
                                        ;uncomplicate.bayadera.core
                                        ;uncomplicate.bayadera.opencl.amd-gcn
                                        ;uncomplicate.bayadera.mcmc.opencl.amd-gcn-stretch
        uncomplicate.bayadera.toolbox.theme
      ]

  :codox {:src-dir-uri "http://github.com/uncomplicate/bayadera/blob/master"
          :src-linenum-anchor-prefix "L"
          :output-dir "docs/codox"}

  :profiles {:dev {:dependencies [[midje "1.8.3"]
                                  [org.clojure/data.csv "0.1.3"]]
                   :plugins [[lein-midje "3.2"]
                             [codox "0.10.1"]]
                   :global-vars {*warn-on-reflection* true
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                        "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"]}}

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]
  :source-paths ["src/clojure" "src/opencl"]
  :java-source-paths ["src/java"]
  :resource-paths ["src/opencl"]
  :test-paths ["test" "test/clojure"])
