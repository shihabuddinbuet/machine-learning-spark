package com.su.spark.ml.practice.regression

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

object LinearRegression {

  val SCHEMA = StructType(
    Array(
      StructField("salary", DoubleType, true),
      StructField("age", DoubleType, true),
      StructField("label", DoubleType, true))
  )

  val DATA_PATH = "src/main/resources/data/binomial-lr.csv"

  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession
        .builder()
        .master("local")
        .appName("linear-regression")
        .getOrCreate()

    val data = spark
        .read
        .format("csv")
        .option("sep", ",")
        .option("header", true)
        .schema(SCHEMA)
        .csv(DATA_PATH)

    val tData = data.rdd.
        map {case Row(salary:Double, age:Double, label:Double)
                            => (label, new DenseVector(Array(salary, age)))}

    import spark.implicits._

    val Array(trainData, testData) = tData.toDF("label", "features")
        .randomSplit(Array(0.7, 0.3))

    val lr = new LinearRegression()
        .setMaxIter(5)
        .setRegParam(0.3)
        .setElasticNetParam(0.8)

    val model = new Pipeline()
        .setStages(Array(lr))
        .fit(trainData)

    evaluate(model, testData)

  }

  def evaluate(model:PipelineModel, testData: DataFrame): Unit = {

    val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("logloss")

    val result = model.transform(testData)
    val ll = evaluator.evaluate(result)
    println("accuracy : " + ll)
  }
}
