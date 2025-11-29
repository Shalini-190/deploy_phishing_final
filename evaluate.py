import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from app.model import AdvancedPhishingDetector
from app.features import extract_url_features

# Reuse data loading functions
def load_openphish():
    try:
        df = pd.read_csv("https://openphish.com/feed.txt", header=None)
        return df[0].dropna().tolist()
    except: return []

def load_urlhaus():
    try:
        url = "https://urlhaus.abuse.ch/downloads/text/"
        resp = requests.get(url, timeout=20)
        urls = []
        for line in resp.text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"): continue
            if line.startswith("http"): urls.append(line)
            else: urls.append("http://" + line)
        return urls
    except: return []

def load_legit():
    # Simulated legit URLs for demo
    base_urls = ["https://www.google.com", "https://www.facebook.com", "https://www.youtube.com", 
            "https://www.amazon.com", "https://www.wikipedia.org", "https://www.twitter.com",
            "https://www.instagram.com", "https://www.linkedin.com", "https://www.reddit.com",
            "https://www.netflix.com", "https://www.microsoft.com", "https://www.apple.com"]
    return base_urls * 100

def main():
    print("Loading data for evaluation...")
    phish_urls = list(set(load_openphish() + load_urlhaus()))[:2000]
    legit_urls = load_legit()
    
    min_len = min(len(phish_urls), len(legit_urls))
    phish_urls = phish_urls[:min_len]
    legit_urls = legit_urls[:min_len]
    
    urls = phish_urls + legit_urls
    labels = [1]*len(phish_urls) + [0]*len(legit_urls)
    
    print(f"Data loaded: {len(urls)} URLs ({len(phish_urls)} Phishing, {len(legit_urls)} Legitimate)")
    
    print("Extracting features...")
    X = []
    for u in urls:
        feats = extract_url_features(u)
        X.append(list(feats.values()))
    X = np.array(X)
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a new model for evaluation purposes
    print("Training evaluation model...")
    detector = AdvancedPhishingDetector(use_slm=False) # Disable SLM for speed/metrics focus
    X_train_scaled = detector.scaler.fit_transform(X_train)
    detector.train_model(X_train_scaled, y_train)
    
    # Evaluate
    print("Evaluating...")
    X_test_scaled = detector.scaler.transform(X_test)
    y_pred = detector.model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "="*30)
    print("PERFORMANCE METRICS")
    print("="*30)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    print("="*30)

if __name__ == "__main__":
    main()
