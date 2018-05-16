package com.su.spark.ml.practice.classification

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

object RandomForestClassifier {
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
        .appName("random-forest-classifier")
        .master("local")
        .getOrCreate()

    val data = spark
        .read
        .format("csv")
        .option("sep", ",")
        .option("header", true)
        .schema(SCHEMA)
        .csv(data_path)

    val columns = data.columns
    val colLen = columns.length

    var featureStringIndexers: Array[StringIndexer] = Array.empty[StringIndexer]
    var featureCols: Array[String] = Array.empty[String]

    for(i <- 0 until(colLen-1)) {
      var inputCol = columns(i)
      var outputCol = columns(i) + "_index"

      var indexer = new StringIndexer()
          .setInputCol(inputCol)
          .setOutputCol(outputCol)
      featureStringIndexers = featureStringIndexers :+ indexer
      featureCols = featureCols :+ outputCol
    }

    val labelIndexer = new StringIndexer()
        .setInputCol(columns(colLen - 1))
        .setOutputCol("label")
        .fit(data)

    val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("outputLabel")
        .setLabels(labelIndexer.labels)

    val va = new VectorAssembler()
        .setInputCols(featureCols)
        .setOutputCol("features")

    val Array(trainData, testData) = new Pipeline()
        .setStages(featureStringIndexers)
        .fit(data)
        .transform(data)
        .randomSplit(Array(0.7, 0.3))

    val rfc = new RandomForestClassifier()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setNumTrees(10)

    val model = new Pipeline()
        .setStages(Array(labelIndexer, va, rfc, labelConverter))
        .fit(trainData)

    evaluate(model, testData)
  }

  def evaluate(model: PipelineModel, testData: DataFrame): Unit = {

    val evaluator = new MulticlassClassificationEvaluator()
        .setPredictionCol("prediction")
        .setLabelCol("label")
        .setMetricName("accuracy")

    val predictions = model.transform(testData)
    val accuracey = evaluator.evaluate(predictions)
    println("accuracy : " + accuracey)
  }
}
