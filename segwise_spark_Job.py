## Make sure pyspark is installed: pip install pyspark
import os
from pyspark.sql import SparkSession
from functools import reduce
from pyspark.sql.functions import min, max,min as min_, max as max_, to_date, year, concat_ws,substring, col, concat, lit
from pyspark.ml.feature import Bucketizer
from itertools import combinations
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from multiprocessing import Pool

def run_spark_job(spark):
    current_directory = os.getcwd()
    print("Current Directory:", current_directory)

    file_name = "playstore.csv"
    file_path = os.path.join(current_directory, file_name)
    print("File Path:", file_path)
    print(os.path.exists(file_path))

    df = spark.read.csv(file_path, header=True, inferSchema=True)

    columns = ["free", "genre", "minInstalls", "price", "ratings", "adSupported", "containsAds", "reviews",
               'releasedDayYear', 'sale', 'score', 'dateUpdated']
    df = df.select(*columns)

    df = df.withColumn("releasedYear", substring(col("releasedDayYear"), -4, 4).cast("int"))
    df = df.filter((col("releasedYear") >= 1900) & (col("releasedYear") <= 2024))

    df = df.withColumn("yearUpdated", substring(col("dateUpdated"), 1, 4).cast("int"))
    df = df.filter((col("yearUpdated") >= 1900) & (col("yearUpdated") <= 2024))

    df = df.withColumn("minInstalls", df["minInstalls"].cast("long"))
    df = df.filter(col("minInstalls") >= 0)

    df = df.withColumn("price", df["price"].cast("float"))
    df = df.filter(col("price") >= 0)

    df = df.withColumn("ratings", df["ratings"].cast("long"))
    df = df.filter(col("ratings") >= 0)

    df = df.withColumn("sale", df["sale"].cast("long"))
    df = df.filter(col("sale") >= 0)

    df = df.withColumn("reviews", df["reviews"].cast("long"))
    df = df.filter(col("reviews") >= 0)

    df = df.withColumn("score", df["score"].cast("float"))
    df = df.filter(col("score") >= 0)

    df = df.withColumn('free', df["free"].cast("boolean"))
    df = df.withColumn('adSupported', df["adSupported"].cast("boolean"))
    df = df.withColumn('containsAds', df["containsAds"].cast("boolean"))

    min_max_values = df.agg(
        min_("ratings").alias("min_ratings"),
        max_("ratings").alias("max_ratings"),
        min_("score").alias("min_score"),
        max_("score").alias("max_score"),
        min_("yearUpdated").alias("min_year_updated"),
        max_("yearUpdated").alias("max_year_updated"),
        min_("releasedYear").alias("min_released_year"),
        max_("releasedYear").alias("max_released_year"),
        min_("price").alias("min_price"),
        max_("price").alias("max_price"),
        min_("minInstalls").alias("min_installs"),
        max_("minInstalls").alias("max_installs")
        ).collect()[0]
    
    min_ratings = min_max_values['min_ratings']
    max_ratings = min_max_values['max_ratings']
    min_score = min_max_values['min_score']
    max_score = min_max_values['max_score']
    min_released_year = min_max_values['min_released_year']
    max_released_year = min_max_values['max_released_year']
    min_year_updated = min_max_values['min_year_updated']
    max_year_updated = min_max_values['max_year_updated']
    min_price = min_max_values['min_price']
    max_price = min_max_values['max_price']
    min_installs = min_max_values['min_installs']
    max_installs = min_max_values['max_installs']

    year_interval = 2
    year_ranges = [x for x in range(min_released_year, max_released_year + 3, year_interval)]
    print(year_ranges)
    # Bucketizing 'releasedYear' column
    df = df.filter(~col("releasedYear").isNull())
    bucketizer_year = Bucketizer(splits=year_ranges, inputCol="releasedYear", outputCol="year_bucket")
    df = bucketizer_year.transform(df)

    updated_year_interval = 2
    updated_year_ranges = [x for x in range(min_year_updated, max_year_updated + 3, updated_year_interval)]
    print(updated_year_ranges)
    # Bucketizing 'updatedYear' column
    df = df.filter(~col("yearUpdated").isNull())
    bucketizer_updated_year = Bucketizer(splits=updated_year_ranges, inputCol="yearUpdated", outputCol="updated_year_bucket")
    df = bucketizer_updated_year.transform(df)

    install_interval = 1000000
    install_ranges = [x for x in range(min_installs, max_installs + install_interval + 1, install_interval)]
    print(install_ranges)
    # Bucketizing 'minInstalls' column
    df = df.filter(~col("minInstalls").isNull())
    bucketizer_installs = Bucketizer(splits=install_ranges, inputCol="minInstalls", outputCol="install_bucket")
    df = bucketizer_installs.transform(df)

    price_interval = 5
    price_ranges = [x for x in range(int(min_price), int(max_price) + 6, price_interval)]
    print(price_ranges)
    # Bucketizing 'price' column
    df = df.filter(~col("price").isNull())
    bucketizer_price = Bucketizer(splits=price_ranges, inputCol="price", outputCol="price_bucket")
    df = bucketizer_price.transform(df)

    ratings_interval = 1000000
    ratings_ranges = [x for x in range(int(min_ratings), int(max_ratings) + ratings_interval + 1, ratings_interval)]
    print(ratings_ranges)
    # Bucketizing 'ratings' column
    df = df.filter(~col("ratings").isNull())
    bucketizer_ratings = Bucketizer(splits=ratings_ranges, inputCol="ratings", outputCol="ratings_bucket")
    df = bucketizer_ratings.transform(df)

    score_interval = 1
    score_ranges = [x for x in range(int(min_score), int(max_score) + score_interval, score_interval)]
    print(score_ranges)
    # Bucketizing 'score' column
    df = df.filter(~col("score").isNull())
    bucketizer_score = Bucketizer(splits=score_ranges, inputCol="score", outputCol="score_bucket")
    df = bucketizer_score.transform(df)

    df = df.withColumn("updated_bucket_range",concat(lit("["), (col("updated_year_bucket")*updated_year_interval+min_year_updated).cast("int"), lit("-"),(col("updated_year_bucket")*updated_year_interval+min_year_updated).cast("int")+updated_year_interval-1, lit("]")))
    df = df.withColumn("year_bucket_range",concat(lit("["), (col("year_bucket")*year_interval+min_released_year).cast("int"), lit("-"),(col("year_bucket")*year_interval+min_released_year).cast("int")+year_interval-1, lit("]")))
    df = df.withColumn("price_bucket_range",concat(lit("["), (col("price_bucket")*price_interval+min_price).cast("int"), lit("-"),(col("price_bucket")*price_interval+min_price).cast("int")+price_interval, lit("]")))
    df = df.withColumn("install_bucket_range",concat(lit("["), (col("install_bucket")*install_interval+min_installs).cast("int"), lit("-"),(col("install_bucket")*install_interval+min_installs).cast("int")+install_interval, lit("]")))
    df = df.withColumn("score_bucket_range",concat(lit("["), (col("score_bucket")*score_interval+min_score).cast("int"), lit("-"),(col("score_bucket")*score_interval+min_score).cast("int")+score_interval, lit("]")))
    df = df.withColumn("rating_bucket_range",concat(lit("["), (col("ratings_bucket")*ratings_interval+min_ratings).cast("int"), lit("-"),(col("ratings_bucket")*ratings_interval+min_ratings).cast("int")+ratings_interval, lit("]")))
    df.select("dateUpdated","updated_bucket_range","score", "score_bucket_range", "price", "price_bucket_range", "releasedYear", "year_bucket_range","minInstalls", "install_bucket_range","ratings","rating_bucket_range").show()

    selected_columns = ["free", "genre", "adSupported", "containsAds", "updated_bucket_range",
                        "year_bucket_range", "install_bucket_range", "price_bucket_range", "rating_bucket_range",
                        "score_bucket_range"]

    min_comb_size = 1
    max_comb_size = 3

    schema = StructType([
        StructField("property", StringType(), True),
        StructField("count", IntegerType(), True)
    ])

    df.cache()
    output_file_name = "output_all_insights.csv"
    output_file_path = os.path.join(current_directory, output_file_name)

    if os.path.exists(output_file_path):
        print(f"Existing Output File removed: {output_file_path}")
        with open(output_file_path, 'w'):
            pass
    print(f"Output CSV will be generated at: {current_directory} \n with file name {output_file_name}")

    for comb_size in range(min_comb_size, max_comb_size + 1):
        combinations_without_duplicates = list(combinations(selected_columns, comb_size))
        for combination in combinations_without_duplicates:
            combination_list = list(combination)
            print(combination_list)

            properties = [concat(concat_ws("=", lit(column), col(column)), lit(";")).alias(column) for column in
                          combination_list]
            combined_properties = concat(*properties).alias("property")

            aggregation = df.groupBy(*combination_list).count().withColumnRenamed("count", "count")

            result_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)
            result_df.cache()

            result_df = aggregation.select(combined_properties, col("count"))
            final_df = result_df.toPandas()
            final_df.to_csv(output_file_path, mode='a', header=False, index=False)
            result_df.show(truncate=False)

    df.unpersist()
    result_df.unpersist()


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("PlaystoreAppInsights") \
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
        .getOrCreate()

    run_spark_job(spark)

    spark.stop()
