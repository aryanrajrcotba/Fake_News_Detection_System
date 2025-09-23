import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_and_save_model():
    """
    This function trains a fake news detection model and saves it.
    """
    # 1. Load the dataset
    df = pd.read_csv('fake_news_dataset.csv')
    
    # Handle potential missing values if any
    df['text'].fillna('', inplace=True)
    
    # 2. Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # 3. Create a TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = vectorizer.fit_transform(x_train)
    tfidf_test = vectorizer.transform(x_test)

    # 4. Initialize and train the PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    # 5. Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {round(score * 100, 2)}%')

    # 6. Save the model and the vectorizer
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(pac, model_file)
        
    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Model and vectorizer saved successfully!")

if __name__ == '__main__':
    train_and_save_model()