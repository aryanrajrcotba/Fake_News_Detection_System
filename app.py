from flask import Flask, render_template, request
import pickle
import requests

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model or vectorizer files not found.")
    print("Please run model.py first to train and save the model.")
    exit()

# NewsAPI configuration
API_KEY = "96db3e7b7b8a452c981a5465f0f64d50"  # <-- IMPORTANT: Replace with your actual NewsAPI key
NEWS_API_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"

def fetch_news():
    """Fetches top headlines from the NewsAPI."""
    try:
        response = requests.get(NEWS_API_URL)
        response.raise_for_status()  # Raise an exception for bad status codes
        articles = response.json().get('articles', [])
        # Filter out articles with no content or description
        return [
            article for article in articles 
            if article.get('content') or article.get('description')
        ]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

@app.route('/')
def home():
    """Renders the home page with live news articles."""
    articles = fetch_news()
    # Use description as fallback if content is missing
    for article in articles:
        article['content_to_check'] = article.get('content') or article.get('description', '')
    return render_template('index.html', articles=articles)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    if request.method == 'POST':
        news_text = request.form['news_text']
        
        # Prepare the text for prediction
        vectorized_text = vectorizer.transform([news_text])
        
        # Make a prediction
        prediction = model.predict(vectorized_text)
        
        # Map prediction to a human-readable result
        result = "Real News" if prediction[0] == 0 else "Fake News"
        
        return render_template('results.html', prediction=result, news_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)