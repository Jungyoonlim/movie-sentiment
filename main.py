from src.data.data_loader import load_data
from src.data.preprocessing import preprocess_text
from src.models.train import train_model
from src.models.evaluate import evaluate_model 

def main():
    train_data = load_data("data/raw/imdb/train")
    test_data = load_data("data/raw/imdb/test")

    print("Train Data:")
    train_data.show(5)

    print("Test Data:")
    test_data.show(5)

    # load data 
    raw_data = load_data()

    # preprocess data
    preprocessed_train_data = preprocess_text(train_data)
    preprocessed_test_data = preprocess_text(raw_data)

    print("Processed Train Data:")
    preprocessed_train_data.show(5)

    print("Processed Test Data:")
    preprocessed_test_data.show(5)

    # Train model
    model = train_model(preprocessed_train_data)

    # Evaluate model
    evaluate_model(model, preprocessed_test_data)

    # Save model
    model.save("models/sentiment_classifier")

if __name__ == "__main__":
    main()