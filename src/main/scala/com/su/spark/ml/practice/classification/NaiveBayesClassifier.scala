package com.su.spark.ml.practice.classification

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.scalatest.TestData

object NaiveBayesClassifier {

  val data_path = "src/main/resources/data/decision-tree.csv"
  val MODEL_PATH = "src/main/resources/models/DTClassifier/"

  val SCHEMA = StructType(Array(
    StructField("temperature", StringType, true),
    StructField("outlook", StringType, true),
    StructField("humidity", StringType, true),
    StructField("windy", StringType, true),
    StructField("play", StringType, true)
  ))

  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession
        .builder()
        .appName("naive bayes classifier")
        .master("local")
        .getOrCreate()

    val data = spark
        .read
        .format("csv")
        .option("header", true)
        .option("sep", ",")
        .schema(SCHEMA)
        .csv(data_path)

    val columns = data.columns
    val totalCol = columns.length

    var featureCols: Array[String] = Array.empty[String]
    var featureStringIndexer: Array[StringIndexer] = Array.empty[StringIndexer]

    for(i <- 0 until(totalCol -1)) {
      val col = columns(i)
      val feature = col + "_index"
      val indexer = new StringIndexer()
          .setInputCol(col)
          .setOutputCol(feature)

      featureCols = featureCols :+ feature
      featureStringIndexer = featureStringIndexer :+ indexer
    }

    val Array(trainData, testData) = new Pipeline()
        .setStages(featureStringIndexer)
        .fit(data)
        .transform(data)
        .randomSplit(Array(0.7, 0.3))

    val va = new VectorAssembler()
        .setInputCols(featureCols)
        .setOutputCol("features")

    val labelIndexer = new StringIndexer()
        .setInputCol(columns(totalCol - 1))
        .setOutputCol("label")
        .fit(data)

    val labelConverter = new IndexToString()
        .setInputCol("label")
        .setOutputCol("outputLabel")
        .setLabels(labelIndexer.labels)

    val nb = new NaiveBayes()
        .setFeaturesCol("features")
        .setLabelCol("label")

    val model = new Pipeline()
        .setStages(Array(labelIndexer, va, nb, labelConverter))
        .fit(trainData)

    evaluate(model, testData)

  }

  def evaluate(model : PipelineModel, testData: DataFrame): Unit = {

    val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")

    val result = model.transform(testData)
    val accuracy = evaluator.evaluate(result)

    println("accuracy : " + accuracy)

  }
}
