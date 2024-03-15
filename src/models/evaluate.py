from pyspark.ml.evaluation import BinaryClassificationEvaluator

def evaluate_model(model, test_data):
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="sentiment")
    accuracy = evaluator.evaluate(predictions)
    print("Test Accuracy: ", accuracy)
    