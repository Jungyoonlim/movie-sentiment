from pyspark.sql.functions import udf, regexp_replace, lower, split
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF

def preprocess_text(data):
    # Tokenization and stop-word removal
    tokenizer = udf(lambda x: regexp_replace(x, "[^a-zA-Z\\s]", "").lower().split())
    data = data.withColumn("tokens", tokenizer(data["review_text"]))
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    data = remover.transform(data)

    # TF-IDF
    hashingTF = HashingTF(inputCol="filtered", outputCol="tf")
    data = hashingTF.transform(data)
    idf = IDF(inputCol="tf", outputCol="tfidf")
    data = idf.fit(data).transform(data)
    return data

