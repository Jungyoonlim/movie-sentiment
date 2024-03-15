from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def train_model(train_data):
    # Define the models
    lr = LogisticRegression(labelCol="sentiment", featuresCol="tfidf")
    rf = RandomForestClassifier(labelCol="sentiment", featuresCol="tfidf")
    gbt = GBTClassifier(labelCol="sentiment", featuresCol="tfidf")

    # Define the parameter grid for hyperparameter tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .addGrid(rf.numTrees, [50, 100]) \
        .addGrid(gbt.maxDepth, [3, 5]) \
        .build()

    # Create the cross-validator
    cv = CrossValidator(estimator=lr,
                        estimatorParamMaps=paramGrid,
                        evaluator=BinaryClassificationEvaluator(labelCol="sentiment"),
                        numFolds=3)

    # Train and evaluate models
    models = [("Logistic Regression", lr), ("Random Forest", rf), ("Gradient Boosted Trees", gbt)]
    best_model = None
    best_accuracy = 0.0

    for name, model in models:
        cv.setEstimator(model)
        cv_model = cv.fit(train_data)
        model = cv_model.bestModel
        predictions = model.transform(train_data)
        evaluator = BinaryClassificationEvaluator(labelCol="sentiment")
        accuracy = evaluator.evaluate(predictions)
        print(f"{name} - Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy

    return best_model