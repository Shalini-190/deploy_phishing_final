import pandas as pd
import requests
from app.model import AdvancedPhishingDetector
from sklearn.model_selection import train_test_split
import numpy as np
from app.features import extract_url_features

def load_openphish():
    print("Loading OpenPhish...")
    try:
        df = pd.read_csv("https://openphish.com/feed.txt", header=None)
        urls = df[0].dropna().tolist()
        print(f"{len(urls)} from OpenPhish")
        return urls
    except:
        print("OpenPhish failed")
        return []

def load_urlhaus():
    print("Loading URLHaus...")
    try:
        url = "https://urlhaus.abuse.ch/downloads/text/"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        urls = []
        for line in resp.text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"): continue
            if line.startswith("http"): urls.append(line)
            else: urls.append("http://" + line)
        print(f"{len(urls)} from URLHaus")
        return urls
    except Exception as e:
        print(f"URLHaus failed: {e}")
        return []

def load_legit():
    print("Loading Legitimate URLs (Expanded)...")
    try:
        # Expanded list of legitimate domains
        domains = [
            "google.com", "facebook.com", "youtube.com", "twitter.com", "instagram.com",
            "linkedin.com", "wikipedia.org", "yahoo.com", "reddit.com", "amazon.com",
            "netflix.com", "microsoft.com", "apple.com", "twitch.tv", "stackoverflow.com",
            "github.com", "gitlab.com", "bitbucket.org", "medium.com", "quora.com",
            "paypal.com", "chase.com", "bankofamerica.com", "wellsfargo.com", "nytimes.com",
            "cnn.com", "bbc.co.uk", "theguardian.com", "forbes.com", "bloomberg.com",
            "adobe.com", "dropbox.com", "salesforce.com", "zoom.us", "slack.com",
            "spotify.com", "hulu.com", "disneyplus.com", "whatsapp.com", "telegram.org",
            "weather.com", "accuweather.com", "espn.com", "nba.com", "nfl.com",
            "imdb.com", "rottentomatoes.com", "yelp.com", "tripadvisor.com", "airbnb.com",
            "uber.com", "lyft.com", "booking.com", "expedia.com", "kayak.com",
            "craigslist.org", "ebay.com", "etsy.com", "walmart.com", "target.com",
            "bestbuy.com", "homedepot.com", "ikea.com", "nike.com", "adidas.com",
            "cnn.com", "foxnews.com", "nbcnews.com", "washingtonpost.com", "usatoday.com"
        ]
        
        paths = [
            "", "/login", "/signin", "/account", "/profile", "/settings", "/dashboard",
            "/search?q=test", "/watch?v=12345", "/user/profile", "/item/12345",
            "/category/electronics", "/articles/2023/news", "/help/contact",
            "/about-us", "/terms", "/privacy", "/download", "/upload", "/images"
        ]
        
        urls = []
        # Generate combinations
        for domain in domains:
            for path in paths:
                urls.append(f"https://www.{domain}{path}")
                urls.append(f"http://{domain}{path}")
        
        # Add specific GitHub URLs to ensure it learns them
        urls.append("https://github.com/microsoft/vscode")
        urls.append("https://github.com/tensorflow/tensorflow")
        urls.append("https://github.com/pytorch/pytorch")
        
        print(f"{len(urls)} legitimate URLs (generated)")
        return urls
    except:
        return []

def main():
    print("Starting Training Pipeline")
    
    # 1. Load Data
    phish_urls = list(set(load_openphish() + load_urlhaus()))[:2000] # Limit for speed
    legit_urls = load_legit()
    
    # Balance dataset
    min_len = min(len(phish_urls), len(legit_urls))
    phish_urls = phish_urls[:min_len]
    legit_urls = legit_urls[:min_len]
    
    print(f"Dataset: {len(phish_urls)} Phishing, {len(legit_urls)} Legitimate")
    
    urls = phish_urls + legit_urls
    labels = [1]*len(phish_urls) + [0]*len(legit_urls)
    
    # 2. Extract Features
    print("Extracting features...")
    detector = AdvancedPhishingDetector(use_slm=True) # Initialize to get scaler ready
    
    X = []
    for u in urls:
        feats = extract_url_features(u)
        X.append(list(feats.values()))
    
    X = np.array(X)
    y = np.array(labels)
    
    # 3. Train
    # Fit scaler first
    X_scaled = detector.scaler.fit_transform(X)
    detector.train_model(X_scaled, y)
    
    # 4. Save
    detector.save("phishing_model")
    print("Model saved to phishing_model.pkl")

if __name__ == "__main__":
    main()
