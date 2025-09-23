# Fake_News_Detection_System

A web application that fetches live news articles from the NewsAPI and uses a Machine Learning model to classify them as "Real News" or "Fake News".

🚀 Live Demo
You can view the live deployed application here:

https://fake-news-detection-system-29xr.onrender.com/



📖 Overview
This project combines a Python Flask backend with a simple HTML/CSS frontend to create a functional Fake News Detection tool. The backend serves a pre-trained PassiveAggressiveClassifier model from Scikit-learn, which predicts the authenticity of news text. The frontend fetches and displays top headlines from the USA via the NewsAPI, allowing users to check any article with a single click.

✨ Features
Live News Feed: Fetches and displays real-time news headlines using the NewsAPI.

AI-Powered Classification: Uses a TfidfVectorizer and a PassiveAggressiveClassifier to predict if news content is real or fake.

Simple & Clean UI: A user-friendly interface for browsing news and viewing results.

Deployment Ready: Configured with Gunicorn and Git LFS for seamless deployment on cloud platforms like Render.

Secure API Key Handling: Uses environment variables to keep API keys safe.

🛠️ Tech Stack
Backend: Python, Flask, Gunicorn

Machine Learning: Scikit-learn, Pandas

Frontend: HTML5, CSS3

API: NewsAPI.org

Deployment: Render, Git, Git LFS

⚙️ Setup and Local Installation
To run this project on your local machine, follow these steps:

Clone the Repository

Bash

git clone https://github.com/aryanrajrcotba/Fake_News_Detection_System.git
cd your-repo-name
Create a Virtual Environment

Bash

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install Dependencies
The project uses Git LFS for large model files. Make sure you have it installed from git-lfs.github.com.

Bash

# Pull the large model files
git lfs pull

# Install Python packages
pip install -r requirements.txt
Set Up Environment Variables
You need a free API key from NewsAPI.org.

Bash

# For Windows (in Command Prompt)
set API_KEY="YOUR_NEWS_API_KEY"

# For macOS/Linux
export API_KEY="YOUR_NEWS_API_KEY"
Note: This variable is only set for the current terminal session. For a more permanent solution, consider using a .env file with python-dotenv.

Run the Application

Bash

flask run
The application will be available at http://127.0.0.1:5000.

🚀 How to Use
Navigate to the home page to see the latest news articles.

Find an article you want to analyze.

Click the "Check Authenticity" button on the article's card.

The system will process the text and redirect you to a results page showing the classification: "Real News" or "Fake News".