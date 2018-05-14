package com.su.spark.ml.practice.classification

import org.apache.spark.sql.{DataFrame, SparkSession}
import com.su.spark.ml.utils.DataUtils.read_tsv

object DecisionTreeClassifier {

  val data_path = "resources/data/classification.txt"
  def main(args: Array[String]): Unit = {

    val spark:SparkSession = SparkSession
                          .builder()
                          .master("local")
                          .appName("decision-tree-classifer")
                          .getOrCreate()
//    val data:DataFrame = read_tsv(spark, data_path)
//                .selectExpr("_c0 as label", "_c1 as X", "_c2 as Y")

//    data.show()

    println("shihab uddin")
  }
}
