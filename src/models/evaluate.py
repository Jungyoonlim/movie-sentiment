from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def evaluate_model(model, test_data):
    predictions = model.transform(test_data)
    binary_evaluator = BinaryClassificationEvaluator(labelCol="sentiment")
    accuracy = binary_evaluator.evaluate(predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="sentiment", predictionCol="prediction")
    precision = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
    recall = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedRecall"})
    f1_score = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "f1"})
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1_score:.4f}")