package com.hh.offline

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import com.hh.offline.OfflineRecommender.MONGODB_RATING_COLLECTION
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
/**
  * author: hehuan
  * date: 2019/8/28 21:54
  */
object ALSTrainer {
  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores"->"local[*]",
      "mongo.uri"->"mongodb://192.168.8.130:27017/recommender",
      "mongo.db"->"recommender"
    )

    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("OfflineRecommender")

    //创建一个sparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._
    implicit val mongoConfig = MongoConfig(config("mongo.uri"),config("mongo.db"))

    //加载评分数据
    val ratingRDD = spark.read
      .option("uri",mongoConfig.uri)
      .option("collection",MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .rdd
      .map(rating => Rating(rating.uid,rating.mid,rating.score))
      .cache()

    //随机切分数据集，生成训练集和测试集
    val splits = ratingRDD.randomSplit(Array(0.8,0.2))
    val trainingRDD = splits(0)
    val testRDD  = splits(1)

    //模型参数选择，输出最优的参数
    adjustALSParam(trainingRDD,testRDD)
    spark.close()
  }

  def adjustALSParam(trainingData: RDD[Rating], testData: RDD[Rating]):Unit={

  }
}
