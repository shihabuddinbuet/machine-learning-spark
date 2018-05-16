package com.su.spark.ml.practice.classification

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

object GradientBoostClassifier {

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
        .master("local")
        .appName("gradient boost classifier")
        .getOrCreate()

    val data = spark
        .read
        .format("csv")
        .option("sep", ",")
        .option("header", true)
        .schema(SCHEMA)
        .csv(data_path)

    val columns = data.columns
    val totalCol = columns.length

    var featureStringIndexer: Array[StringIndexer] = Array.empty[StringIndexer]
    var featureColumns: Array[String] = Array.empty[String]

    for(i <- 0 until(totalCol - 1)) {
      var column = columns(i)
      val featureCol = column + "_index"
      var indexer = new StringIndexer()
          .setInputCol(column)
          .setOutputCol(featureCol)
      featureStringIndexer = featureStringIndexer :+ indexer
      featureColumns = featureColumns :+ featureCol
    }

    val Array(trainData, testData) = new Pipeline()
        .setStages(featureStringIndexer)
        .fit(data)
        .transform(data)
        .randomSplit(Array(0.7, 0.3))

    val va = new VectorAssembler()
        .setInputCols(featureColumns)
        .setOutputCol("features")

    val labelIndexer = new StringIndexer()
        .setInputCol(columns(totalCol - 1))
        .setOutputCol("label")
        .fit(data)

    val labelConverter = new IndexToString()
        .setInputCol("label")
        .setOutputCol("outputLabel")
        .setLabels(labelIndexer.labels)

    val gbc = new GBTClassifier()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setMaxIter(5)

    val model = new Pipeline()
        .setStages(Array(labelIndexer, va, gbc, labelConverter))
        .fit(trainData)

    evaluate(model, testData)

  }

  def evaluate(model: PipelineModel, testData: DataFrame): Unit = {
    val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")

    val result = model.transform(testData)
    val accuracy = evaluator.evaluate(result)
    println("accuracy : " + accuracy)

  }
}
