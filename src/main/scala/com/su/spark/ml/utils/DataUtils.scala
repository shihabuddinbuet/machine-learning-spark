package com.su.spark.ml.utils

import org.apache.spark.sql.catalyst.ScalaReflection.Schema
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, SparkSession}

object DataUtils {

  def read_tsv(spark: SparkSession, location:String, schema:StructType = null): DataFrame = {
    if(schema != null) spark.read.option("delimiter", "\t").schema(schema).load(location)
    else spark.read.option("delimiter", "\t").load(location)
  }

  def read_parquet(spark: SparkSession, location:String): DataFrame = {
      return spark.read.parquet(location)
  }
}
