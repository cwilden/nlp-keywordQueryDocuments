// Databricks notebook source
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
// COMMAND ----------

//import corpus - movie plot summaries
val plots = sc.textFile("/FileStore/tables/plot_summaries.txt")

// COMMAND ----------

import org.apache.spark.sql.functions.split 
val plotDF = plots.toDF("text")

// COMMAND ----------

import org.apache.spark.sql.types.IntegerType

val idPlotDF = plotDF.withColumn("label", (split($"text", "\t").getItem(0)).cast(IntegerType))
      .withColumn("text", split($"text", "\t").getItem(1))

// COMMAND ----------

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

//file with keywords to query on documents
val searchTerms = sc.textFile("/FileStore/tables/searchterms.txt")

// COMMAND ----------

searchTerms.collect()

// COMMAND ----------

import org.apache.spark.ml.feature.{Tokenizer}
val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("raw") 

// COMMAND ----------

val plotWords = tokenizer.transform(idPlotDF)

// COMMAND ----------

//remove Stop Words
import org.apache.spark.ml.feature.StopWordsRemover
val remover = new StopWordsRemover()
  .setInputCol("raw")
  .setOutputCol("filtered")

// COMMAND ----------

val filteredData = remover.transform(plotWords)

// COMMAND ----------

// filteredData.show()

// COMMAND ----------

import org.apache.spark.ml.feature.{HashingTF}
val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("rawFeatures")

// COMMAND ----------

val tf = hashingTF.transform(filteredData)

// COMMAND ----------

// tf.show()

// COMMAND ----------

import org.apache.spark.ml.feature.{IDF}

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features").fit(tf)

// COMMAND ----------

val tfidf = idf.transform(tf)

// COMMAND ----------

val searchTermsDF = searchTerms.toDF("keywords")

// COMMAND ----------

val splitTerms = searchTermsDF.withColumn("filtered", split(col("keywords"), " "))

// COMMAND ----------

val sizeSplitTerms = splitTerms.withColumn("wordCount", size(col("filtered")))

// COMMAND ----------

val singleWordsDF = sizeSplitTerms.filter($"wordCount" === 1)

// COMMAND ----------

val multipleWordsDF = sizeSplitTerms.filter($"wordCount" > 1)

// COMMAND ----------

import org.apache.spark.sql.types._

val schema = StructType(Seq(StructField(
  "features", 
  StructType(Seq(
    StructField("indices", ArrayType(LongType, true), true), 
    StructField("size", LongType, true),
    StructField("type", ShortType, true), 
    StructField("values", ArrayType(DoubleType, true), true)
)), true)))

val featureVector = tfidf.select(from_json(
to_json(struct($"features")), schema).getItem("features").alias("data")
                   )

// featureVector.show()

// COMMAND ----------

val featureVectorValues = featureVector.select($"data".getItem("values")).toDF("weights")
// featureVectorValues.show()

// COMMAND ----------

val featureVectorRDD = sqlContext.createDataFrame(
  featureVectorValues.rdd.zipWithIndex.map {
    case (row, index) => Row.fromSeq(row.toSeq :+ index)
  },
  // Create schema for index column
  StructType(featureVectorValues.schema.fields :+ StructField("index", LongType, false))
)

// COMMAND ----------

//import and join movie information for plots
val movies = spark.read.option("header","false").option("delimiter","\t"). option("inferSchema","true").csv("FileStore/tables/movie_metadata-ab497.tsv").toDF("label", "freebaseID", "movie", "release date", "revenue", "runtime", "languages", "countries", "genres")

// COMMAND ----------

// movies.show()

// COMMAND ----------

val moviesJoinedDF = tfidf.join(movies, Seq("label"))

// COMMAND ----------

// moviesJoinedDF.show()

// COMMAND ----------

val moviesRDD = spark.sqlContext.createDataFrame(
  moviesJoinedDF.rdd.zipWithIndex.map {
    case (row, index) => Row.fromSeq(row.toSeq :+ index)
  },
  // Create schema for index column
  StructType(moviesJoinedDF.schema.fields :+ StructField("index", LongType, false))
)

// COMMAND ----------

val moviesWithWeights = moviesRDD.join(featureVectorRDD, Seq("index")).drop("index","freebaseID", "raw", "rawFeatures", "release date", "revenue", "runtime", "languages", "countries", "genres")

// COMMAND ----------

val firstWord = singleWordsDF.select($"keywords").take(1)(0).mkString(" ")

// COMMAND ----------

val keywordTF = hashingTF.transform(singleWordsDF.select($"filtered"))
val keywordTFIDF = idf.transform(keywordTF)

// COMMAND ----------

val schema = StructType(Seq(StructField(
  "features", 
  StructType(Seq(
    StructField("indices", ArrayType(LongType, true), true), 
    StructField("size", LongType, true),
    StructField("type", ShortType, true), 
    StructField("values", ArrayType(DoubleType, true), true)
)), true)))

val keyWordVector = keywordTFIDF.select(from_json(
to_json(struct($"features")), schema).getItem("features").alias("keywordWeight")
                   )

// keyWordVector.show()

// COMMAND ----------

val keywordVectorValues = keyWordVector.select($"keywordWeight".getItem("values")).toDF("keywordWeight")

// COMMAND ----------

// keywordVectorValues.show()

// COMMAND ----------

val keywordVector = spark.sqlContext.createDataFrame(
  keywordVectorValues.rdd.zipWithIndex.map {
    case (row, index) => Row.fromSeq(row.toSeq :+ index)
  },
  // Create schema for index column
  StructType(keywordVectorValues.schema.fields :+ StructField("index", LongType, false))
)

// COMMAND ----------

val keywordTFIDFRDD = spark.sqlContext.createDataFrame(
  keywordTFIDF.rdd.zipWithIndex.map {
    case (row, index) => Row.fromSeq(row.toSeq :+ index)
  },
  // Create schema for index column
  StructType(keywordTFIDF.schema.fields :+ StructField("index", LongType, false))
)

// COMMAND ----------

val keywordWithWeights = keywordVector.join(keywordTFIDFRDD, Seq("index")).drop("index")

// COMMAND ----------

// keywordWithWeights.show()

// COMMAND ----------

val keywordsDF = keywordWithWeights.withColumn("words", concat_ws(" ", $"filtered"))

// COMMAND ----------

val actualWeightsKeywordsDF = keywordsDF.withColumn("weight", $"keywordWeight".getItem(0))

// COMMAND ----------

val sortedAllWeights = actualWeightsKeywordsDF.select($"weight").collect().map(r => r.getDouble(0)).sorted

// COMMAND ----------

import org.apache.spark

val sortedWeightsBroadcast = sc.broadcast(sortedAllWeights)

// COMMAND ----------

val moviesMatch = moviesWithWeights.withColumn("exploCol", explode($"weights"))

// COMMAND ----------

// moviesMatch.select($"label", $"movie", $"weights",$"exploCol").show()

// COMMAND ----------

def findClosest(element: Double, values: Array[Double]): Double = {
  var left = 0
  var right = values.length - 1
  var closest = Double.NaN
  var min = Double.MaxValue
  while(left <= right) {
    val mid = (left + right) / 2
    val current = values(mid)
    if(current == element) {
      closest = element
      left = right + 1
    }
    else {
      if(current < element) {
        left = mid + 1
      }
      else {
        right = mid - 1
      }
      val distance = (current - element).abs
      if(distance < min) {
        min = distance
        closest = current
      }
    }
  }
  closest
}

// COMMAND ----------

val findClosestUdf = udf((element: Double) => findClosest(element, sortedWeightsBroadcast.value))

// COMMAND ----------

val moviesWithClosest = moviesMatch.withColumn("weight", findClosestUdf(moviesMatch("exploCol")))

// COMMAND ----------

// moviesWithClosest.select($"label", $"movie", $"text", $"exploCol", $"weight").show()

// COMMAND ----------

val moviesWithDifference = moviesWithClosest.withColumn("difference", abs(moviesWithClosest("exploCol") - moviesWithClosest("weight")))

// COMMAND ----------

val moviesNoDups = moviesWithDifference.dropDuplicates("label")

// COMMAND ----------

import org.apache.spark.sql.functions._

val topMatches = moviesWithDifference.select("*").filter($"difference" <= 0.001).drop("filtered", "exploCol", "weights", "features")

// COMMAND ----------

import org.apache.spark.sql.expressions.Window

val window = Window.partitionBy("weight").orderBy("difference")

// COMMAND ----------

import org.apache.spark.sql.functions.row_number

val rankByScore = row_number().over(window)

// COMMAND ----------

val top10 = topMatches.select('*, rankByScore as 'rank).filter(col("rank") <= 10).drop("rank")

// COMMAND ----------

val moviesWithKeyword = top10.join(actualWeightsKeywordsDF, Seq("weight")).drop("weight", "keywordWeight", "filtered", "difference", "rawFeatures", "features")

// COMMAND ----------

display(moviesWithKeyword)
