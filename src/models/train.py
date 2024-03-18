from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def train_model(train_data):
    # Define the models
    lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.1)
    rf = RandomForestClassifier(numTrees=200, maxDepth=10, featureSubsetStrategy="auto")
    gbt = GBTClassifier(maxDepth=10, stepSize=0.01, subsamplingRate=0.8)


    # Define the parameter grid for hyperparameter tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .addGrid(rf.numTrees, [50, 100, 200]) \
        .addGrid(gbt.maxDepth, [5, 10, 20]) \
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