package com.su.spark.ml.practice.classification

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import com.su.spark.ml.utils.DataUtils.read_tsv
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.types.{StringType, StructField, StructType}

object DecisionTreeClassifier {

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

    val spark:SparkSession = SparkSession.
                          builder().
                          master("local").
                          appName("decision-tree-classifer").
                          getOrCreate()

    val rootData = spark
        .read
        .format("csv")
        .schema(SCHEMA)
        .option("sep", ",")
        .option("header", true)
        .csv(data_path)

    rootData.show()
    val columns = rootData.columns
    var stringIndexers: Array[StringIndexer] = Array.empty[StringIndexer]
    var indexedCols:Array[String] = Array.empty[String]

    for(col <- columns) {
      var indexCol = col + "_index"
      var indexer = new StringIndexer()
          .setInputCol(col)
          .setOutputCol(indexCol)
      stringIndexers = stringIndexers :+ indexer
      indexedCols = indexedCols :+ indexCol
    }

    val Array(trainData, testData) = new Pipeline()
        .setStages(stringIndexers)
        .fit(rootData)
        .transform(rootData)
        .randomSplit(Array(0.7, 0.3))

    val va = new VectorAssembler()
        .setInputCols(indexedCols.slice(0, indexedCols.length-1))
        .setOutputCol("features_index")

    val labelIndexer = new StringIndexer()
        .setInputCol("play")
        .setOutputCol("label_index")
        .fit(rootData)

    val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("outputLabel")
        .setLabels(labelIndexer.labels)

    val dt = new DecisionTreeClassifier()
        .setLabelCol("label_index")
        .setFeaturesCol("features_index")

    val pipeline = new Pipeline()
        .setStages(Array(labelIndexer, va, dt, labelConverter))

    val model = pipeline.fit(trainData)
    evaluate(model, testData)

  }

  def evaluate(model: PipelineModel, testData: DataFrame): Unit = {

    val predictions = model.transform(testData)
    predictions.selectExpr("label_index", "prediction").show()

    val eval = new MulticlassClassificationEvaluator()
        .setLabelCol("label_index")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")

    val acc = eval.evaluate(predictions)
    println("accuracy : " + acc)

  }

  def saveModel(path: String, model: PipelineModel): Unit = {
    model.write.save(path)
  }

}
