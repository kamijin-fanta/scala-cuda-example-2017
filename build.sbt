name := "scala-cuda"

version := "1.0"

scalaVersion := "2.12.2"

libraryDependencies += "org.jcuda" % "jcuda" % "0.8.0"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"
libraryDependencies += "com.github.jai-imageio" % "jai-imageio-core" % "1.3.0"

fork := true
