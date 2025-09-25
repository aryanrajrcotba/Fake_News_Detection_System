import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

def train_and_evaluate_model():
    """
    This function trains a fake news detection model, evaluates it with
    detailed metrics, and saves the model and evaluation plots.
    """
    # 1. Load the dataset
    df = pd.read_csv('fake_news_dataset.csv')
    df['text'].fillna('', inplace=True)
    
    # 2. Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # 3. Create and fit the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = vectorizer.fit_transform(x_train)
    tfidf_test = vectorizer.transform(x_test)

    # 4. Initialize and train the PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    # 5. Make predictions on the test set
    y_pred = pac.predict(tfidf_test)
    
    # --- MODEL EVALUATION ---
    
    # 5a. Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(accuracy * 100, 2)}%')
    print("-" * 30)

    # 5b. Classification Report (Precision, Recall, F1-Score)
    print("Classification Report:")
    # target_names are based on your labels: 0 for 'Real', 1 for 'Fake'
    print(classification_report(y_test, y_pred, target_names=['Real News (0)', 'Fake News (1)']))
    print("-" * 30)

    # 5c. Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png') # Save the plot as an image
    print("Confusion Matrix plot saved as 'confusion_matrix.png'")
    print("-" * 30)

    # 5d. Credibility Score (Decision Function) & Predicted Levels
    # This score shows the model's confidence. Negative = Real, Positive = Fake.
    decision_scores = pac.decision_function(tfidf_test)
    print("Sample Predictions with Credibility Scores:")
    for i in range(5):
        text_sample = x_test.iloc[i][:70] + "..." # Show a snippet of the text
        true_label = "Real" if y_test.iloc[i] == 0 else "Fake"
        pred_label = "Real" if y_pred[i] == 0 else "Fake"
        score = decision_scores[i]
        print(f'Text: "{text_sample}"')
        print(f'--> True: {true_label}, Predicted: {pred_label}, Credibility Score: {score:.2f}\n')
    print("-" * 30)

    # 5e. ROC Curve and AUC Score
    # The decision function provides the scores needed to plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, decision_scores)
    auc_score = roc_auc_score(y_test, decision_scores)
    print(f"AUC (Area Under Curve) Score: {auc_score:.2f}")

    # Plotting the ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Dashed line for random guess
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png') # Save the plot as an image
    print("ROC Curve plot saved as 'roc_curve.png'")
    print("-" * 30)
    
    # 6. Save the trained model and vectorizer for the app
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(pac, model_file)
    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Model and vectorizer saved successfully!")


if __name__ == '__main__':
    train_and_evaluate_model()