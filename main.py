import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Fetch news from an example API
def fetch_news(api_url):
    response = requests.get(api_url)
    data = response.json()
    articles = data['articles']
    return articles

# Preprocess and categorize news articles
def categorize_news(articles):
    df = pd.DataFrame(articles)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['title'])
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    df['category'] = kmeans.labels_
    return df

# Display personalized news feed
def display_news(df, user_preferences):
    filtered_df = df[df['category'].isin(user_preferences)]
    for index, row in filtered_df.iterrows():
        print(f"Title: {row['title']}\nDescription: {row['description']}\n")

# Example usage
if __name__ == "__main__":
    API_URL = "https://newsapi.org/v2/top-headlines?country=us&apiKey=YOUR_API_KEY"
    articles = fetch_news(API_URL)
    categorized_news = categorize_news(articles)
    user_preferences = [0, 2]  # Example preferences
    display_news(categorized_news, user_preferences)
