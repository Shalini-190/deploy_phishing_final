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
    print("Loading Cisco Umbrella Top 1M (Subset)...")
    try:
        # Using a smaller subset for quick training in this environment
        # In production, download the full zip
        urls = ["https://www.google.com", "https://www.facebook.com", "https://www.youtube.com", 
                "https://www.amazon.com", "https://www.wikipedia.org", "https://www.twitter.com",
                "https://www.instagram.com", "https://www.linkedin.com", "https://www.reddit.com",
                "https://www.netflix.com", "https://www.microsoft.com", "https://www.apple.com"] * 100
        print(f"{len(urls)} legitimate URLs (simulated)")
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
