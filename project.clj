(let [nar-classifier (str (System/getProperty "os.arch") "-"
                          (System/getProperty "os.name")
                          "-gpp-jni")]
  (defproject uncomplicate/bayadera "0.1.0-SNAPSHOT"
    :description "FIXME: write description"
    :url "http://github.com/blueberry/bayadera"
    :license {:name "Eclipse Public License"
              :url "http://www.eclipse.org/legal/epl-v10.html"}
    :dependencies [[org.clojure/clojure "1.8.0-alpha4"]
                   [uncomplicate/clojurecl "0.3.0-SNAPSHOT"]
                   [uncomplicate/neanderthal "0.4.0-SNAPSHOT"]
                   [org.apache.commons/commons-math3 "3.3"]
                   [me.raynes/fs "1.4.6"]]

    :global-vars {*warn-on-reflection* true
                  *unchecked-math* :warn-on-boxed}

    :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                         "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"]

    :aot [uncomplicate.bayadera.mcmc.opencl.amd-gcn-stretch]

    :codox {:src-dir-uri "http://github.com/uncomplicate/bayadera/blob/master"
            :src-linenum-anchor-prefix "L"
            :output-dir "docs/codox"}

    :profiles {:dev {:dependencies [[midje "1.7.0"]
                                    [criterium "0.4.3"]
                                    [uncomplicate/neanderthal-atlas "0.1.0"
                                     :classifier ~nar-classifier]
                                    [incanter/incanter-core "1.9.0"]
                                    [incanter/incanter-io "1.9.0"]]
                     :plugins [[lein-midje "3.1.3"]
                               [codox "0.8.13"]]}}

    :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]
    :source-paths ["src/clojure" "src/opencl"]
    :java-source-paths ["src/java"]
    :resource-paths ["src/opencl"]
    :test-paths ["test" "test/clojure"]))
