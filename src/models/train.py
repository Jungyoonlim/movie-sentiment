from pyspark.ml.classification import LogisticRegression 

def train_model(train_data):
    lr = LogisticRegression(featuresCol="features", labelCol="sentiment", maxIter=10)
    lr_model = lr.fit(train_data)
    return lr_model