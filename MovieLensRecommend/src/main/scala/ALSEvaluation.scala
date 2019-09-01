import Recommend._
import java.io.File

import breeze.numerics.sqrt
import scala.io.Source
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.joda.time.format._
import org.joda.time._
import org.joda.time.Duration
import org.jfree.data.category.DefaultCategoryDataset
import org.apache.spark.mllib.regression.LabeledPoint
/**
  * author: hehuan
  * date: 2019/8/31 17:25
  */
object ALSEvaluation {
  def main(args: Array[String]): Unit = {
    SetLogger
    println("=========数据准备阶段========")
    val (trainData,validationData,testData)=PrepareData()
    //为增加执行效率，使用persist命令持久化在内存中
    trainData.persist();validationData.persist();testData.persist()

    println("=========训练验证阶段=========")
    val bestModel = trainValidation(trainData,validationData)

    println("=========测试阶段=========")
    val testRmse = computeRMSE(bestModel,testData)

    println("测试结果 RMSE= "+testRmse)

    trainData.unpersist();validationData.unpersist();testData.unpersist()
  }
  //数据准备
  def PrepareData():(RDD[Rating],RDD[Rating],RDD[Rating]) = {
    //1,创建用户评分数据
    val sc = new SparkContext(new SparkConf().setAppName("Recommend").setMaster("local[*]"))
    println("开始读取用户评分数据中。。。")
    val rawUserData = sc.textFile("/Users/hehuan/Documents/MoviesRecommendSystem/recommender/MovieLensRecommend/src/main/resources/ml-100k/u.data")
    val rawRatings = rawUserData.map(_.split("\t").take(3))
    val ratingRDD = rawRatings.map{
      case Array(user,movie,rating)=>
        Rating(user.toInt,movie.toInt,rating.toDouble)
    }
    println("共计："+ratingRDD.count().toString+"条评分")
    //2，创建电影id与名称对照表
    println("开始读取电影数据中。。。")
    val DataDir = "/Users/hehuan/Documents/MoviesRecommendSystem/recommender/MovieLensRecommend/src/main/resources/ml-100k"
    val itemRDD = sc.textFile(new File(DataDir,"u.item").toString)
    val movieTitle = itemRDD.map(line=>line.split("\\|").take(2))
      .map(array=>(array(0).toInt,array(1)))
      .collect()
      .toMap
    //3，显示数据记录数
    val numRatings = ratingRDD.count()
    val numUsers = ratingRDD.map(_.user).distinct().count()
    val numMovies = ratingRDD.map(_.product).distinct().count()
    println("共计: 评分："+numRatings+"条\t用户："+numUsers+"位\t电影："+numMovies)

    //4，以随机方式将数据分为3个部分并且返回
    println("将数据分为：")
    //将训练数据，验证数据、测试数据的量随机分成8：1；1的比例
    val Array(trainData,validationData,testData) = ratingRDD.randomSplit(Array(0.8,0.1,0.1))
    println(" trainData:"+trainData.count()+" validationData:"+validationData.count()+" testData:"+testData.count())
    return (trainData,validationData,testData)
  }
  //训练数据评估
  def trainValidation(trainData: RDD[Rating], validationData: RDD[Rating]):MatrixFactorizationModel={
    println("-----------评估rank参数使用-----------")
    evaluateParameter(trainData,validationData,"rank",Array(5,10,15,20,50,100),Array(10),Array(0.1))
    println("-----------评估numIterations参数使用-----------")
    evaluateParameter(trainData,validationData,"numIterations",Array(10),Array(5,10,15,20,25),Array(0.1))
    println("-----------评估lambda参数使用-----------")
    evaluateParameter(trainData,validationData,"lambda",Array(10),Array(10),Array(0.05,0.1,1,5,10.0))
    println("-----------所有参数交叉评估找出最好的参数组合-------------")
    val bestModel = evaluateAllParameter(trainData,validationData,Array(5,10,15,20,25),Array(5,10,15,20,25),Array(0.05,0.1,1,5,10.0))
    return (bestModel)
  }
  //评估单个参数，判断哪个参数有较低的误差，并绘制图形
  def evaluateParameter(trainData: RDD[Rating], validationData: RDD[Rating], evaluateParameter: String, rankArray: Array[Int], numIterationsArray: Array[Int], lambdaArray: Array[Double])={
    var dataBarChart = new DefaultCategoryDataset()
    var dataLineChart  = new DefaultCategoryDataset()
    for(rank<-rankArray;numIterations<-numIterationsArray;lambda<-lambdaArray){
      val (rmse,time) = trainModel(trainData,validationData,rank,numIterations,lambda)
      val parameterData =
        evaluateParameter match {
          case "rank" =>rank
          case "numIterations" =>numIterations
          case "lambda" => lambda
        }
      dataBarChart.addValue(rmse,evaluateParameter,parameterData.toString())
      dataLineChart.addValue(time,"Time",parameterData.toString())
    }
    Chart.plotBarLineChart("ALS evaluations "+evaluateParameter,evaluateParameter,"RMSE",0.58,5,"Time",dataBarChart,dataLineChart)
  }


  //训练模型
  def trainModel(trainData: RDD[Rating], validationData: RDD[Rating], rank: Int, iterations: Int, lambda: Double):(Double,Double)={
    val startTime = new DateTime()
    val model = ALS.train(trainData,rank,iterations,lambda)
    val endTime = new DateTime()
    val Rmse = computeRMSE(model,validationData)
    val duration = new Duration(startTime,endTime)
    println("训练参数：rank："+rank + "，iterations："+iterations+",lambda:"+lambda+",结果 RMSE="+Rmse+",训练需要时间："+duration.getMillis+"毫秒！")
   // println(f"训练参数：rank:$rank%3d,iterations:$iterations%.2f,lambda=$lambda%.2d,结果 Rmse = $Rmse%.2f"+"训练需要时间："+duration.getMillis+"毫秒！")
    (Rmse,duration.getStandardSeconds)
  }

  //计算RMSE
  def computeRMSE(model: MatrixFactorizationModel, RatingRDD: RDD[Rating]): Double = {
    val num = RatingRDD.count()
    // 计算预测评分
    val predictedRDD = model.predict(RatingRDD.map(r => (r.user, r.product)))

    // 以...,..作为外键，inner join实际观测值和预测值
    val predict = predictedRDD.map( p => ( (p.user, p.product), p.rating ) )
    val observed = RatingRDD.map( r => ( (r.user, r.product), r.rating ) )
    val predictedAndRatings = predict.join(observed).values
    // 内连接得到(uid, mid),(actual, predict)
    math.sqrt(
      predictedAndRatings.map(x=>(x._1-x._2)*(x._1-x._2)).reduce(_+_) / num)
  }

//评估多个参数，交叉验证评估出最优组合参数
  def evaluateAllParameter(trainData: RDD[Rating], validationData: RDD[Rating], rankArray: Array[Int], numIterationsArray: Array[Int], lambdaArray: Array[Double]):MatrixFactorizationModel={
    val evaluations =
      for (rank<-rankArray;numIterations<-numIterationsArray;lambda<-lambdaArray) yield{
        val (rmse,time) = trainModel(trainData,validationData,rank,numIterations,lambda)
        (rank,numIterations,lambda,rmse)
      }
    val Eval = (evaluations.sortBy(_._4))
    val BestEval = Eval(0)
    println("最佳model参数：rank："+BestEval._1+",iterations:"+BestEval._2+",lambda:"+BestEval._3+",结果RMSE="+BestEval._4)
    val bestModel = ALS.train(trainData,BestEval._1,BestEval._2,BestEval._3)
    (bestModel)
  }

}
