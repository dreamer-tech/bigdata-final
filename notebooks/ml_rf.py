import datetime

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator,
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType
from collections import defaultdict

SAVE_LIMIT = 100
PATH = "../output/"


rmse_evaluator = RegressionEvaluator(metricName="rmse", predictionCol='', labelCol="popularity")
r2_evaluator = RegressionEvaluator(metricName="r2", predictionCol='', labelCol="popularity")


def get_preference_data(prediction_df):
    initial_recs = get_preferences(
        prediction_df,
        in_column="is_recommended_enc",
        out_column="initial_recommendations",
    )
    predicted_recs = get_preferences(
        prediction_df,
        in_column="prediction",
        out_column="prediction_recommendations",
        custom_udf=fivePreferencesListUDF,
    )

    transformed_data = initial_recs.join(predicted_recs, "user_id", "inner")
    return transformed_data


def evaluate_recommendations(prediction_df, model_name=""):
    transformed_data = get_preference_data(prediction_df)

    print("Start evaluating {} RMSE...".format(model_name))
    rmse_score = rmse_evaluator.evaluate(transformed_data)
    print("Finish with {} RMSE".format(model_name))

    print("Start evaluating {} R2...".format(model_name))
    r2_score = r2_evaluator.evaluate(transformed_data)
    print("Finish with {} R2".format(model_name))

    print("{} RMSE score = {:.3f}".format(model_name, rmse_score))
    print("{} R2 score = {:.3f}".format(model_name, r2_score))

    return transformed_data, rmse_score, r2_score


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

    df_tracks = spark.sql("select * from tracks limit 100000")
    df_artists = spark.sql("select * from artists limit 100000")

    artist_popularity, artist_followers = defaultdict(int), defaultdict(float)
    for artist_id, followers, popularity in df_artists.select("artist_id", "followers", "popularity").collect():
        artist_followers[artist_id] = followers
        artist_popularity[artist_id] = popularity

    custom_func = F.udf(lambda x: sum([artist_popularity[y] for y in eval(x)]), IntegerType())
    df_tracks = df_tracks.withColumn('artists_popularity', custom_func(F.col("id_artists")))

    custom_func = F.udf(lambda x: sum([artist_followers[y] for y in eval(x)]), FloatType())
    df_tracks = df_tracks.withColumn('artists_followers', custom_func(F.col("id_artists")))

    # tracks_df['id_artists'] = [i[2:-2] for i in tracks_df['id_artists']]
    # custom_func = F.udf(lambda x: x[2:-2], StringType())
    # df_tracks = df_tracks.withColumn("id_artists", custom_func(F.col("id_artists")))

    # tracks_df['release_year'] = [int(i.split('-')[0]) for i in tracks_df['release_date']]
    df_tracks = df_tracks.withColumn("release_year", F.year("release_date"))

    # artists_df.rename(columns={'id': 'id_artists', 'popularity': 'artists_popularity'}, inplace=True)
    # df_artists = df_artists.withColumn("id_artists", F.col("artist_id"))
    # df_artists = df_artists.withColumn("artists_popularity", F.col("popularity"))

    # artists_df.drop(['genres', 'name'], axis=1, inplace=True)
    # df_artists = df_artists.drop("artist_id", "popularity", "genres", "artist_name")

    # tracks_df = tracks_df.merge(artists_df, on='id_artists')
    # df_tracks = df_tracks.join(df_artists, on='id_artists')

    # df_games = df_games.drop("title", "steam_deck", "price_final", "date_release")
    # df_games = (
    #     df_games.withColumn("linux", F.col("linux").cast("double"))
    #     .withColumn("mac", F.col("mac").cast("double"))
    #     .withColumn("win", F.col("win").cast("double"))
    # )

    # rating_dict = {
    #     "Overwhelmingly Positive": 8,
    #     "Very Positive": 7,
    #     "Positive": 6,
    #     "Mostly Positive": 5,
    #     "Mixed": 4,
    #     "Mostly Negative": 3,
    #     "Negative": 2,
    #     "Very Negative": 1,
    #     "Overwhelmingly Negative": 0,
    # }
    # encode_rating = F.udf(lambda x: rating_dict[x], IntegerType())
    # df_games = df_games.withColumn("rating", encode_rating(F.col("rating")))
    # df_games_rec = df_games.join(
    #     rec_enc.select("is_recommended_enc", "app_id", "user_id"), "app_id", "inner"
    # )
    # df_games_rec = df_games_rec.withColumn(
    #     "is_recommended_enc", F.col("is_recommended_enc").cast("double") - 1.0
    # )

    df_tracks_enc = df_tracks

    feature_columns_rf = [
        "explicit",
        "danceability",
        "loudness",
        "instrumentalness",
        "release_year",
        "artists_followers",
        "artists_popularity",
        "popularity"
    ]
    vector_assembler = VectorAssembler(
        inputCols=feature_columns_rf, outputCol="features_unscaled"
    )
    scaler = MinMaxScaler(inputCol="features_unscaled", outputCol="features")
    pipeline = Pipeline(stages=[vector_assembler, scaler])
    features_pipeline_model_svc = pipeline.fit(df_tracks_enc)
    df_tracks_enc = features_pipeline_model_svc.transform(df_tracks_enc)
    # df_tracks_enc.show()

    rf_features = [(c,) for c in feature_columns_rf]
    rf_features_df = spark.createDataFrame(data=rf_features, schema=["feature"])
    rf_features_df.show()
    rf_features_df.coalesce(1).write.mode("overwrite").format("csv").option(
        "sep", ","
    ).option("header", "true").csv(PATH + "pda/rf_features")

    rf_data = df_tracks_enc
    # rf_data.show()

    ## Cross-validation
    rf_train_data, rf_test_data = rf_data.randomSplit([0.7, 0.3], seed=42)

    rf_model = RandomForestClassifier(labelCol="popularity", seed=42)
    rf_params = (
        ParamGridBuilder()
        .addGrid(rf_model.numTrees, [5, 7, 10])
        .addGrid(rf_model.maxDepth, [3, 4, 5])
        .build()
    )
    cv_rf = CrossValidator(
        estimator=rf_model,
        estimatorParamMaps=rf_params,
        evaluator=rmse_evaluator,
        parallelism=2,
        numFolds=2,
        seed=42,
    )
    cv_rf_model = cv_rf.fit(rf_train_data)

    rf_params_mapped = [
        dict([(y_rf[0].name, y_rf[1]) for y_rf in x_rf.items()]) for x_rf in rf_params
    ]
    rf_param_names = list(rf_params_mapped[0].keys())
    cv_rf_config = [[float(x[name]) for name in rf_param_names] for x in rf_params_mapped]
    cv_rf_config_df = spark.createDataFrame(data=cv_rf_config, schema=rf_param_names)
    cv_rf_config_df.show()
    cv_rf_config_df.coalesce(1).write.mode("overwrite").format("csv").option(
        "sep", ","
    ).option("header", "true").csv(PATH + "pda/cv_rf_config")

    best_rf_numTrees = cv_rf_model.bestModel._java_obj.parent().getNumTrees()
    best_rf_maxDepth = cv_rf_model.bestModel._java_obj.parent().getMaxDepth()
    print("Best RF model numTrees = ", best_rf_numTrees)
    print("Best RF model maxDepth = ", best_rf_maxDepth)
    best_rf_params = [
        ("numTrees", float(best_rf_numTrees)),
        ("maxDepth", float(best_rf_maxDepth)),
    ]
    best_rf_params_df = spark.createDataFrame(
        data=best_rf_params, schema=["parameter", "value"]
    )
    best_rf_params_df.show()
    best_rf_params_df.coalesce(1).write.mode("overwrite").format("csv").option(
        "sep", ","
    ).option("header", "true").csv(PATH + "pda/best_rf_params")

    final_rf = cv_rf_model.bestModel
    final_rf.write().overwrite().save(PATH + "models/rf")

    ## Testing

    rf_predictions = (
        final_rf.transform(rf_test_data)
        # .select("user_id", "app_id", "is_recommended_enc", "prediction")
        # .withColumn("is_recommended_enc", F.col("is_recommended_enc") + 1)
        # .withColumn("prediction", F.col("prediction") + 1)
    )
    rf_predictions.show()
    exit(0)

    rf_recommendations, rf_rmse_score, rf_r2_score = evaluate_recommendations(
        rf_predictions.withColumn(
            "is_recommended_enc", F.col("is_recommended_enc").cast("double")
        ),
        "RF",
    )

    best_rf_scores = [("RMSE", float(rf_rmse_score)), ("R2", float(rf_r2_score))]
    best_rf_scores_df = spark.createDataFrame(
        data=best_rf_scores, schema=["metric", "value"]
    )
    best_rf_scores_df.show()
    best_rf_scores_df.coalesce(1).write.mode("overwrite").format("csv").option(
        "sep", ","
    ).option("header", "true").csv(PATH + "pda/best_rf_scores")

    rf_recommendations.limit(SAVE_LIMIT).coalesce(1).write.mode("overwrite").format(
        "json"
    ).json(PATH + "pda/rf_recommendations")
