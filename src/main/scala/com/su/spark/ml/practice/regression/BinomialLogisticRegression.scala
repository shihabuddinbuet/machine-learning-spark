package com.su.spark.ml.practice.regression

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object BinomialLogisticRegression {

  val SCHEMA = StructType(
    Array(
      StructField("salary", DoubleType, true),
      StructField("age", DoubleType, true),
      StructField("label", DoubleType, true))
  )

  val DATA_PATH = "src/main/resources/data/binomial-lr.csv"

  def main(args: Array[String]): Unit = {

    val spark:SparkSession = SparkSession
        .builder()
        .master("local")
        .appName("binomial-logistic-regression")
        .getOrCreate()

    val rootData:DataFrame = spark
        .read
        .format("csv")
        .schema(SCHEMA)
        .option("sep", ",")
        .option("header", true)
        .csv(DATA_PATH)

    val trainRDD = rootData.rdd.map{
      case Row(salary:Double, age:Double, label:Double) => (label, new DenseVector(Array(salary, age)))
    }

    import spark.implicits._
    val Array(trainData, testData) = trainRDD.toDF("label", "features")
        .randomSplit(Array(0.7, 0.3))

    val lr = new LogisticRegression()
        .setMaxIter(15)
        .setRegParam(0.3)
        .setElasticNetParam(0.7)

    val model = lr.fit(trainData)
    val prediction = model.transform(testData)
    prediction.show()

  }
}
