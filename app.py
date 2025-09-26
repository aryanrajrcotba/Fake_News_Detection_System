import os
import requests
import re
import time
from flask import Flask, render_template, request
from transformers import pipeline
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('Agg')


print("Loading AI model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("AI model loaded successfully.")


app = Flask(__name__)


API_KEY = os.environ.get("API_KEY", "96db3e7b7b8a452c981a5465f0f64d50")
NEWS_API_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"


SOURCE_CREDIBILITY = { "reuters": {"score": "Very High", "value": 5}, "associated press": {"score": "Very High", "value": 5}, "bbc news": {"score": "Very High", "value": 5}, "the guardian": {"score": "High", "value": 4}, "the new york times": {"score": "High", "value": 4}, "cnn": {"score": "Moderate", "value": 3}, "fox news": {"score": "Moderate", "value": 3} }

def get_source_credibility(source_name):
    if not source_name: return {"score": "Unknown", "value": 0}
    source_name_lower = source_name.lower()
    for key, cred in SOURCE_CREDIBILITY.items():
        if key in source_name_lower: return cred
    return {"score": "Not Rated", "value": 0}

def fetch_news():
    try:
        response = requests.get(NEWS_API_URL)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        for article in articles:
            article['credibility'] = get_source_credibility(article.get('source', {}).get('name'))
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

@app.route('/')
def home():
    articles = fetch_news()
    
    # --- CHANGE 1: Updated dashboard stats to only have two categories ---
    stats = {"Real News": 0, "Fake News": 0}
    candidate_labels = ["reliable news", "fabricated news", "opinion piece", "satire"]

    for article in articles:
        text = (article.get('title', '') or '') + ' ' + (article.get('description', '') or '')
        if not text.strip():
            continue
        
        try:
            prediction = classifier(text, candidate_labels)
            top_label = prediction['labels'][0]

            if top_label == "reliable news":
                stats["Real News"] += 1
            else: 
                stats["Fake News"] += 1
        except Exception as e:
            print(f"Error classifying article for dashboard: {e}")

    
    labels = stats.keys()
    sizes = stats.values()
    colors = ['#2ecc71', '#e74c3c'] # Green for Real, Red for Fake
    
    chart_image = None
    if sum(sizes) > 0:
        plt.figure(figsize=(5, 5))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title('Classification of Fetched Articles', pad=20)
        plt.axis('equal')
        
        timestamp = int(time.time())
        chart_image = f'chart_{timestamp}.png'
        plt.savefig(f'static/{chart_image}')
        plt.close()

    return render_template('index.html', articles=articles, chart_image=chart_image)

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']
    cleaned_text = re.sub(r'\[\+\d+ chars\]', '', news_text).strip()
    
    candidate_labels = ["reliable news", "fabricated news", "opinion piece", "satire"]
    prediction_result = classifier(cleaned_text, candidate_labels)
    
    top_label = prediction_result['labels'][0]
    confidence_score = prediction_result['scores'][0]
    
    # --- CHANGE 2: Updated prediction logic to only have two outcomes ---
    if top_label == "reliable news":
        result = "Real News"
    else: # Everything else (fabricated, opinion, satire) is considered Fake News
        result = "Fake News"
        
    return render_template(
        'results.html', 
        prediction=result, 
        original_text=news_text,
        confidence=f"{confidence_score:.0%}"
    )

if __name__ == '__main__':
    app.run(debug=True)