import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_and_evaluate_model():
    """
    Trains and evaluates a generalizable model using a large dataset.
    """
    
    print("Loading datasets...")
    df_fake = pd.read_csv('Fake.csv')
    df_true = pd.read_csv('True.csv')

    df_fake['label'] = 1
    df_true['label'] = 0

    df = pd.concat([df_fake, df_true], ignore_index=True)
    print("Datasets loaded and combined successfully.")
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df['text'] = df['title'] + " " + df['text']
    df.drop(columns=['title', 'subject', 'date'], inplace=True)
    
    print("Dataset prepared for training.")
    print("-" * 30)

    # 2. Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # 3. Use an Improved TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
    
    print("Vectorizing text data...")
    tfidf_train = vectorizer.fit_transform(x_train)
    tfidf_test = vectorizer.transform(x_test)
    print("Vectorization complete.")
    print("-" * 30)
    
    # 4. Train the PassiveAggressiveClassifier Model
    print("Training PassiveAggressiveClassifier model...")
    model = PassiveAggressiveClassifier(max_iter=50) 
    model.fit(tfidf_train, y_train)
    print("Model training complete.")
    print("-" * 30)
    
    # 5. Make predictions and evaluate
    print("Evaluating model...")
    
    y_pred = model.predict(tfidf_test) 
    
    print("MODEL EVALUATION RESULTS")
    print("-" * 30)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(accuracy * 100, 2)}%')
    print("-" * 30)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    print("-" * 30)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix plot saved as 'confusion_matrix.png'")
    print("-" * 30)
    
    # 6. Save the new model and vectorizer
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    print("New model and vectorizer saved successfully!")

if __name__ == '__main__':
    train_and_evaluate_model()