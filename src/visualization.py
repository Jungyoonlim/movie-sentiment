import matplotlib.pyplot as pyplot

def plot_sentiment_dist(data):
    # Count the number of pos and neg sentiments
    pos_count = data.filter(data["sentiment"] == 1).count()
    neg_count = data.filter(data["sentiment"] == 0).count()

    # Plot the sentiment distribution
    pyplot.bar(["Positive", "Negative"], [pos_count, neg_count])
    pyplot.title("Sentiment Distribution")
    pyplot.figure(figsize=(8,6))
    pyplot.xlabel("Sentiment")
    pyplot.ylabel("Count")
    pyplot.title('Sentiment Distribution')
    pyplot.show()

def plot_accuracy(accuracy):
    # Create a bar plot for accuracy
    pyplot.figure(figsize=(6, 4))
    pyplot.bar(['Accuracy'], [accuracy])
    pyplot.ylim([0, 1])
    pyplot.xlabel('Metric')
    pyplot.ylabel('Score')
    pyplot.title('Model Accuracy')
    pyplot.show()