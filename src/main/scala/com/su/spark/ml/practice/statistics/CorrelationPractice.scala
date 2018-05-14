package com.su.spark.ml.practice.statistics

import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
//import org.apache.spark.ml.stat

class CorrelationPractice {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
        .builder()
        .appName("correlation-practice")
        .getOrCreate()

    val data = Seq(
      Vectors.sparse(4, Seq((0, 1.0) ,(3, -1.0))),
      Vectors.dense(1.0, 2.0, 3.0, 4.0),
      Vectors.dense(5.0, 6.0, 7.0, 8.0),
      Vectors.sparse(4, Seq((0,1.0), (3, -1.0)))
    )

//    val df = data.map(Tuple1.apply).toDF("features")
//    val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
//    println(s"Pearson correlation matrix:\n $coeff1")

  }
}
