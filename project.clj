(defproject uncomplicate/bayadera "0.1.0-SNAPSHOT"
  :description "Bayesian Inference and Probabilistic Machine Learning Library for Clojure"
  :url "http://github.com/blueberry/bayadera"
  :scm {:name "git"
        :url "https://github.com/blueberry/bayadera"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [uncomplicate/commons "0.2.0"]
                 [uncomplicate/fluokitten "0.5.0"]
                 [uncomplicate/clojurecl "0.6.4"]
                 [uncomplicate/neanderthal "0.6.3-SNAPSHOT"]
                 [org.apache.commons/commons-math3 "3.6.1"]
                 [me.raynes/fs "1.4.6"]
                 [quil "2.4.0"]]

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
                                  [criterium "0.4.4"]]
                   :plugins [[lein-midje "3.1.3"]
                             [codox "0.9.4"]]
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
