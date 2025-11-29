import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from .features import extract_url_features, extract_email_features

class LightweightSLM:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            print(f"Loading SLM: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.enabled = True
        except Exception as e:
            print(f"SLM load failed: {e}")
            self.enabled = False
            return

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2)
        ).to(self.device)
        
        print("SLM loaded")

    def analyze(self, text):
        if not self.enabled:
            return {'slm_enabled': False}

        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(emb)
            probs = torch.softmax(logits, dim=1)[0]

        phishing_p = probs[1].item()
        legit_p = probs[0].item()

        return {
            'slm_enabled': True,
            'slm_prediction': int(phishing_p > 0.5),
            'slm_phishing_probability': phishing_p,
            'slm_legitimate_probability': legit_p,
            'slm_confidence': max(phishing_p, legit_p)
        }

    def save(self, path):
        if self.enabled:
            torch.save({'classifier': self.classifier.state_dict()}, path)

    def load(self, path):
        if self.enabled:
            try:
                ckpt = torch.load(path, map_location=self.device)
                self.classifier.load_state_dict(ckpt['classifier'])
                print(f"SLM weights loaded from {path}")
            except FileNotFoundError:
                print(f"SLM weights not found at {path}, using random init")

class AdvancedPhishingDetector:
    def __init__(self, use_slm=True):
        self.use_slm = use_slm
        self.scaler = StandardScaler()
        self.model = None
        self.slm = LightweightSLM() if use_slm else None

    def train_model(self, X, y):
        print("Training ensemble models...")
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        gb = GradientBoostingClassifier()
        xgb = XGBClassifier(n_estimators=200, eval_metric='logloss')
        lr = LogisticRegression(max_iter=1000)
        nn_clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=400)

        self.model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb), ('gb', gb), ('lr', lr), ('nn', nn_clf)],
            voting='soft',
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        print("Training complete")

    def predict_url(self, url):
        # 1. ML Prediction
        features = extract_url_features(url)
        feature_values = list(features.values())
        
        try:
            X = self.scaler.transform([feature_values])
        except:
            X = np.array([feature_values])

        if self.model:
            ml_pred = self.model.predict(X)[0]
            ml_prob = self.model.predict_proba(X)[0][1]
        else:
            ml_pred = 0
            ml_prob = 0.0

        result = {
            "url": url,
            "ml_prediction": int(ml_pred),
            "ml_phishing_probability": float(ml_prob),
            "features": features
        }

        # 2. SLM Prediction
        if self.use_slm and self.slm:
            slm_text = f"URL Analysis: {url}"
            slm_res = self.slm.analyze(slm_text)
            result.update(slm_res)

            if slm_res.get('slm_enabled'):
                p = ml_prob * 0.6 + slm_res['slm_phishing_probability'] * 0.4
                result['ensemble_prediction'] = int(p > 0.5)
                result['ensemble_probability'] = p
                result['final_verdict'] = "Phishing" if p > 0.5 else "Legitimate"
            else:
                result['ensemble_probability'] = ml_prob
                result['final_verdict'] = "Phishing" if ml_pred == 1 else "Legitimate"
        else:
            result['ensemble_probability'] = ml_prob
            result['final_verdict'] = "Phishing" if ml_pred == 1 else "Legitimate"
        
        return result

    def predict_email(self, subject, body):
        features = extract_email_features(subject, body)
        
        score = 0
        if features['num_urls'] > 0: score += 0.2
        if features['has_urgent_words']: score += 0.3
        if features['has_financial_words']: score += 0.2
        
        slm_res = {'slm_enabled': False}
        if self.use_slm and self.slm:
            text = f"Subject: {subject} | Body: {body[:512]}"
            slm_res = self.slm.analyze(text)
        
        result = {
            "type": "email",
            "subject": subject,
            "heuristic_score": score,
            "features": features
        }
        result.update(slm_res)
        
        if slm_res.get('slm_enabled'):
            final_prob = (score + slm_res['slm_phishing_probability']) / 2
            final_prob = min(final_prob, 1.0)
        else:
            final_prob = score
            
        result['ensemble_probability'] = final_prob
        result['final_verdict'] = "Phishing" if final_prob > 0.5 else "Legitimate"
        
        return result

    def save(self, path_prefix="model"):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, f"{path_prefix}.pkl")
        if self.use_slm and self.slm:
            self.slm.save(f"{path_prefix}_slm.pt")

    def load(self, path_prefix="model"):
        try:
            data = joblib.load(f"{path_prefix}.pkl")
            self.model = data['model']
            self.scaler = data['scaler']
            print(f"ML model loaded from {path_prefix}.pkl")
        except FileNotFoundError:
            print(f"ML model not found at {path_prefix}.pkl")
            
        if self.use_slm and self.slm:
            self.slm.load(f"{path_prefix}_slm.pt")
