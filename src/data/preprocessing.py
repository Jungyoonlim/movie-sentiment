from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF
from pyspark.sql.functions import udf, regexp_replace, lower, split
from pyspark.sql.types import ArrayType, StringType

def preprocess_text(data):
    # Tokenization and stop-word removal
    data = data.withColumn("tokens", split(regexp_replace(lower(data["value"]), "[^a-zA-Z\\s]", ""), "\\s+"))
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    data = remover.transform(data)

    # TF-IDF
    hashingTF = HashingTF(inputCol="filtered", outputCol="tf")
    data = hashingTF.transform(data)
    idf = IDF(inputCol="tf", outputCol="tfidf")
    data = idf.fit(data).transform(data)
    
    # Data Quality Checks
    data = data.filter(data["text"].isNotNull())
    data = data.filter(data["sentiment"].isin([0, 1]))

    # Data Mixture 
    pos_count = data.filter(data["sentiment"] == 1).count()
    neg_count = data.filter(data["sentiment"] == 0).count()
    print (f"Positive count: {pos_count}, Negative count: {neg_count}")

    return data


