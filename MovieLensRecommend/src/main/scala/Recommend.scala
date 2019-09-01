import java.io.File
import scala.io.Source
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS,Rating,MatrixFactorizationModel}
import scala.collection.immutable.Map

/**
  * author: hehuan
  * date: 2019/8/31 15:51
  */
object Recommend {
  //主程序入口
  def main(args: Array[String]): Unit = {
    //不显示日志信息
    SetLogger
    //一、数据准备
    val (ratings,movieTitle) = prepareData()
    println("模型训练开始。。。")
    //二、训练模型
    //参数使用ALSEvaluation.scala中训练出的最优参数
    val model = ALS.train(ratings,15,20,0.1)
    //一定要在下面写一条打印语句，要不然会报错，不知道怎么回事。。。
    println("模型训练完成。。。")
    //三、进行不同的推荐
    recommend(model,movieTitle)

  }

  // 推荐类型选择
  def recommend(model: MatrixFactorizationModel, movieTitle: Map[Int, String]): Unit = {
    var choose = ""
    while (choose != "3") {
      print("请选择要推荐的类型 1.针对用户推荐电影 2.针对电影推荐给感兴趣的用户 3.离开")
      choose = readLine()
      if (choose == "1") {
        print("请输入用户ID？")
        val inputUserID = readLine()
        RecommendMovies(model, movieTitle, inputUserID.toInt)
      } else if (choose == "2") {
        print("请输入电影的ID？")
        val inputMoveID = readLine()
        RecommendUsers(model, movieTitle, inputMoveID.toInt)
      }
    }
  }

  // 一、数据准备阶段
  def prepareData():(RDD[Rating],Map[Int,String])={
    //----------1.创建用户评分数据------------
    //创建SparkContext  local[*]表示尽可能多的使用本地线程
    val sc = new SparkContext(new SparkConf().setAppName("Recommend").setMaster("local[*]"))
    println("开始读取用户评分数据中。。。")

    //SparkContext读取用户评分数据
    val rawUserData = sc.textFile("/Users/hehuan/Documents/MoviesRecommendSystem/recommender/MovieLensRecommend/src/main/resources/ml-100k/u.data")

    //只需要前三个字段：用户id，电影id，评分
    val rawRatings = rawUserData.map(_.split("\t").take(3))

    //rawRatings.map使用map转换，创建Rating数据类型（userID，movieID，rating）
    val ratingRDD = rawRatings.map{
      case Array(user,movie,rating)=>
        Rating(user.toInt,movie.toInt,rating.toDouble)
    }
    println("共计："+ratingRDD.count().toString+"条评分")

    //----------2.创建电影id与名称对照表-------
    println("开始读取电影数据中。。。")
    val DataDir = "/Users/hehuan/Documents/MoviesRecommendSystem/recommender/MovieLensRecommend/src/main/resources/ml-100k"
    val itemRDD = sc.textFile(new File(DataDir,"u.item").toString)

    //数据格式为：1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/
    //只需要前两个字段：电影id，电影名称
    val movieTitle = itemRDD.map(line=>line.split("\\|").take(2))
      .map(array=>(array(0).toInt,array(1)))
      .collect()
      .toMap

    //----------3.显示数据记录数--------------
    val numRatings = ratingRDD.count()
    val numUsers = ratingRDD.map(_.user).distinct().count()
    val numMovies = ratingRDD.map(_.product).distinct().count()
    println("共计: 评分："+numRatings+"条\t用户："+numUsers+"位\t电影："+numMovies)

    //返回
    return (ratingRDD,movieTitle)
  }

  // 三、推荐阶段
 //针对用户推荐电影
def RecommendMovies(model: MatrixFactorizationModel, movieTitle: Map[Int, String], inputUserID: Int)={
  //获取针对inputUserID推荐前10部电影，并存入RecommendMovie
   val RecommendMovie = model.recommendProducts(inputUserID,10)
  var i = 1
  println("针对用户id"+inputUserID+"推荐以下电影：")

   //使用foreach读取RecommendMovie的每一条数据，并显示在界面上
   RecommendMovie.foreach{ r=>
    println(i.toString()+"."+movieTitle(r.product)+" 评分："+r.rating.toString())
    i += 1
  }
}
//针对电影推荐给用户
  def RecommendUsers(model: MatrixFactorizationModel, movieTitle: Map[Int, String], inputMovieID: Int): Unit ={
    //获取针对inputMovieID推荐前10部电影，并存入RecommendUser
    val RecommendUser = model.recommendUsers(inputMovieID,10)
    var i = 1
    println("针对电影id"+inputMovieID+"电影名："+movieTitle(inputMovieID.toInt)+"推荐下列用户id：")
    RecommendUser.foreach{r=>
      println(i.toString+"用户id："+r.user+" 评分："+r.rating)
      i += 1
    }
  }
  // 设置不显示log信息
  def SetLogger = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("Spark.ui.showConsoleProgress","false")
    Logger.getRootLogger().setLevel(Level.OFF)
  }
}
