from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, FloatType
from collections import defaultdict


SAVE_LIMIT = 100
PATH = "../output/"


if __name__ == '__main__':
    spark = (
        SparkSession.builder.appName("BD Project")
        .master("local[*]")
        .config("spark.driver.memory", "3g")
        .config("spark.sql.catalogImplementation", "hive")
        .config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083")
        .config("spark.sql.avro.compression.codec", "snappy")
        .config(
            "spark.jars",
            "/usr/hdp/current/hive-client/lib/hive-metastore-1.2.1000.2.6.5.0-292.jar,/usr/hdp/current/hive-client/lib/hive-exec-1.2.1000.2.6.5.0-292.jar",
        )
        .config("spark.jars.packages", "org.apache.spark:spark-avro_2.11:2.4.4")
        .enableHiveSupport()
        .getOrCreate()
    )

    sc = spark.sparkContext
    sc.setLogLevel("OFF")

    tracks = spark.read.format("avro").table("projectdb.tracks_part")
    tracks.createOrReplaceTempView("tracks")

    artists = spark.read.format("avro").table("projectdb.artists_part")
    artists.createOrReplaceTempView("artists")

    df_tracks = spark.sql("select * from tracks limit 10000")
    df_artists = spark.sql("select * from artists limit 10000")

    artist_popularity, artist_followers = defaultdict(int), defaultdict(float)
    for artist_id, followers, popularity in df_artists.select("artist_id", "followers", "popularity").collect():
        artist_followers[artist_id] = followers
        artist_popularity[artist_id] = popularity

    custom_func = F.udf(lambda x: sum([artist_popularity[y] for y in eval(x)]), IntegerType())
    df_tracks = df_tracks.withColumn('artists_popularity', custom_func(F.col("id_artists")))

    custom_func = F.udf(lambda x: sum([artist_followers[y] for y in eval(x)]), FloatType())
    df_tracks = df_tracks.withColumn('artists_followers', custom_func(F.col("id_artists")))

    df_tracks = df_tracks.withColumn("release_year", F.year("release_date"))

    df_tracks_enc = df_tracks

    feature_columns_gbt = [
        "explicit",
        "danceability",
        "loudness",
        "instrumentalness",
        "release_year",
        "artists_followers",
        "artists_popularity"
    ]
    vector_assembler = VectorAssembler(
        inputCols=feature_columns_gbt, outputCol="features_unscaled"
    )
    scaler = MinMaxScaler(inputCol="features_unscaled", outputCol="features")
    pipeline = Pipeline(stages=[vector_assembler, scaler])
    features_pipeline_model = pipeline.fit(df_tracks_enc)
    df_tracks_enc = features_pipeline_model.transform(df_tracks_enc)

    gbt_features = [(c,) for c in feature_columns_gbt]
    gbt_features_df = spark.createDataFrame(data=gbt_features, schema=["feature"])
    gbt_features_df.show()
    gbt_features_df.coalesce(1).write.mode("overwrite").format("csv").option(
        "sep", ","
    ).option("header", "true").csv(PATH + "pda/gbt_features")

    gbt_data = df_tracks_enc

    rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="popularity")

    ## Cross-validation
    gbt_train_data, gbt_test_data = gbt_data.randomSplit([0.7, 0.3], seed=42)

    gbt_model = GBTRegressor(labelCol="popularity", seed=42)
    gbt_params = (
        ParamGridBuilder()
        .addGrid(gbt_model.maxIter, [10, 50])
        .addGrid(gbt_model.maxDepth, [3, 4, 5])
        .build()
    )
    cv_gbt = CrossValidator(
        estimator=gbt_model,
        estimatorParamMaps=gbt_params,
        evaluator=rmse_evaluator,
        parallelism=2,
        numFolds=2,
        seed=42,
    )
    cv_gbt_model = cv_gbt.fit(gbt_train_data)

    gbt_params_mapped = [
        dict([(y_gbt[0].name, y_gbt[1]) for y_gbt in x_gbt.items()]) for x_gbt in gbt_params
    ]
    gbt_param_names = list(gbt_params_mapped[0].keys())
    cv_gbt_config = [[float(x[name]) for name in gbt_param_names] for x in gbt_params_mapped]
    cv_gbt_config_df = spark.createDataFrame(data=cv_gbt_config, schema=gbt_param_names)
    cv_gbt_config_df.show()
    cv_gbt_config_df.coalesce(1).write.mode("overwrite").format("csv").option(
        "sep", ","
    ).option("header", "true").csv(PATH + "pda/cv_gbt_config")

    best_gbt_maxIter = cv_gbt_model.bestModel._java_obj.parent().getMaxIter()
    best_gbt_maxDepth = cv_gbt_model.bestModel._java_obj.parent().getMaxDepth()
    print("Best GBT model maxIter = ", best_gbt_maxIter)
    print("Best GBT model maxDepth = ", best_gbt_maxDepth)
    best_gbt_params = [
        ("maxIter", float(best_gbt_maxIter)),
        ("maxDepth", float(best_gbt_maxDepth)),
    ]
    best_gbt_params_df = spark.createDataFrame(
        data=best_gbt_params, schema=["parameter", "value"]
    )
    best_gbt_params_df.show()
    best_gbt_params_df.coalesce(1).write.mode("overwrite").format("csv").option(
        "sep", ","
    ).option("header", "true").csv(PATH + "pda/best_gbt_params")

    final_gbt = cv_gbt_model.bestModel
    final_gbt.write().overwrite().save(PATH + "models/gbt")

    ## Testing

    gbt_predictions = (
        final_gbt.transform(gbt_test_data)
    )

    gbt_rmse_score = RegressionEvaluator(labelCol="popularity", predictionCol="prediction", metricName="rmse")
    gbt_rmse_score = gbt_rmse_score.evaluate(gbt_predictions)

    gbt_r2_score = RegressionEvaluator(labelCol="popularity", predictionCol="prediction", metricName="r2")
    gbt_r2_score = gbt_r2_score.evaluate(gbt_predictions)

    best_gbt_scores = [("RMSE", float(gbt_rmse_score)), ("R2", float(gbt_r2_score))]
    best_gbt_scores_df = spark.createDataFrame(
        data=best_gbt_scores, schema=["metric", "value"]
    )
    best_gbt_scores_df.show()
    best_gbt_scores_df.coalesce(1).write.mode("overwrite").format("csv").option(
        "sep", ","
    ).option("header", "true").csv(PATH + "pda/best_gbt_scores")

    gbt_predictions.select("prediction").limit(SAVE_LIMIT).coalesce(1).write.mode("overwrite").format(
        "json"
    ).json(PATH + "pda/gbt_popularity")
