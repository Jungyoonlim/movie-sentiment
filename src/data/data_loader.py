from pyspark.sql import SparkSession
from pyspark.sql.functions import lit 

def load_data(file_path):
    spark = SparkSession.builder \
        .appName("TextClassification") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    pos_data = spark.read.text(f"{file_path}/pos").withColumn("sentiment", lit(1))
    neg_data = spark.read.text(f"{file_path}/neg").withColumn("sentiment", lit(0))

    data = pos_data.union(neg_data)
    data = data.repartition(100)

    return data 

