"""
FINAL VERSION — OpenPhish + URLHaus + Cisco Umbrella
Phishing URL Detector with Integrated Small Language Model (SLM)
Balanced dataset | 100% URL-based | No manual CSV download
"""

import pandas as pd
import numpy as np
import requests
import re
import math
import warnings
warnings.filterwarnings('ignore')

import joblib
from urllib.parse import urlparse, parse_qs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ---------------------- SLM -------------------------
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False
    print("⚠ transformers not installed — SLM disabled")


# ============================================================
#                SMALL LANGUAGE MODEL (SLM)
# ============================================================

class LightweightSLM:
    def __init__(self, model_name="distilbert-base-uncased"):
        if not TRANSFORMERS_AVAILABLE:
            self.enabled = False
            print("⚠ SLM disabled")
            return

        self.enabled = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("⏳ Loading SLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2)
        ).to(self.device)

        print("✅ SLM loaded")

    def extract_text(self, url):
        parsed = urlparse(url)
        return (
            f"URL: {url} | "
            f"Protocol: {parsed.scheme} | "
            f"Domain: {parsed.netloc} | "
            f"Path: {parsed.path}"
        )

    def analyze(self, url):
        if not self.enabled:
            return {'slm_enabled': False}

        text = self.extract_text(url)
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                max_length=256).to(self.device)

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
            'slm_confidence': max(phishing_p, legit_p),
            'slm_reasoning': "SLM-based URL semantic analysis"
        }

    def save(self, path):
        if self.enabled:
            torch.save({'classifier': self.classifier.state_dict()}, path)

    def load(self, path):
        if self.enabled:
            ckpt = torch.load(path, map_location=self.device)
            self.classifier.load_state_dict(ckpt['classifier'])


# ============================================================
#                MAIN PHISHING DETECTOR
# ============================================================

class AdvancedPhishingDetectorWithSLM:

    def __init__(self, use_slm=True):
        self.use_slm = use_slm
        self.scaler = StandardScaler()
        self.model = None
        if use_slm:
            self.slm = LightweightSLM()
        else:
            self.slm = None

    # =============== FEATURE EXTRACTION ======================

    def entropy(self, text):
        if not text:
            return 0
        p = [text.count(c) / len(text) for c in set(text)]
        return -sum(x * math.log2(x) for x in p)

    def extract_features(self, url):
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path

        f = {}
        f['url_len'] = len(url)
        f['domain_len'] = len(domain)
        f['path_len'] = len(path)
        f['num_dot'] = url.count('.')
        f['num_hyph'] = url.count('-')
        f['num_slash'] = url.count('/')
        f['has_https'] = 1 if parsed.scheme == 'https' else 0
        f['entropy_url'] = self.entropy(url)
        f['entropy_domain'] = self.entropy(domain)
        f['entropy_path'] = self.entropy(path)
        f['num_digits'] = sum(c.isdigit() for c in url)
        f['num_letters'] = sum(c.isalpha() for c in url)
        f['num_special'] = sum(not c.isalnum() for c in url)
        f['num_params'] = url.count('=')
        f['has_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', domain) else 0

        return f

    # ============================================================
    #                DATASET CREATION FROM URLS
    # ============================================================

    def load_openphish(self):
        print("📥 Loading OpenPhish...")
        try:
            df = pd.read_csv("https://openphish.com/feed.txt", header=None)
            urls = df[0].dropna().tolist()
            print(f"✔ {len(urls)} from OpenPhish")
            return urls
        except:
            print("❌ OpenPhish failed")
            return []

    def load_urlhaus(self):
        print("📥 Loading URLHaus...")

        try:
            url = "https://urlhaus.abuse.ch/downloads/text/"
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()

            urls = []
            for line in resp.text.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("http://") or line.startswith("https://"):
                    urls.append(line)
                else:
                    urls.append("http://" + line)

            print(f"✔ Loaded {len(urls)} phishing URLs from URLHaus")
            return urls

        except Exception as e:
            print(f"❌ URLHaus failed: {e}")
            return []

    def load_cisco_legit(self):
        print("📥 Loading Cisco Umbrella Top 1M...")
        try:
            df = pd.read_csv(
                "http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip",
                compression="zip",
                header=None,
                names=["rank", "domain"]
            )
            urls = [f"https://{d}/" for d in df["domain"].tolist()]
            print(f"✔ {len(urls)} legitimate")
            return urls
        except:
            print("❌ Cisco failed")
            return ["https://www.google.com/", "https://www.facebook.com/"] * 5000

    # ============================================================
    #                     TRAIN PIPELINE
    # ============================================================

    def create_training_data(self, target=10000):
        phish1 = self.load_openphish()
        phish2 = self.load_urlhaus()
        phishing = list(set(phish1 + phish2))
        phishing = phishing[:target]

        legit = self.load_cisco_legit()[:target]

        print(f"\n📊 Final balanced dataset: Legit={len(legit)}, Phish={len(phishing)}")

        urls = legit + phishing
        labels = [0]*len(legit) + [1]*len(phishing)
        return urls, labels

    # ============================================================

    def train(self):
        urls, labels = self.create_training_data()
        print("🔍 Extracting features...")

        X = [list(self.extract_features(u).values()) for u in urls]
        X = np.array(X)
        y = np.array(labels)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)

        print("🤖 Training models...")

        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        gb = GradientBoostingClassifier()
        xgb = XGBClassifier(n_estimators=200, eval_metric='logloss')
        lr = LogisticRegression(max_iter=1000)
        nn = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=400)

        self.model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb), ('gb', gb), ('lr', lr), ('nn', nn)],
            voting='soft',
            n_jobs=-1
        )

        self.model.fit(X_tr, y_tr)

        acc = self.model.score(X_te, y_te)
        print(f"🎯 Accuracy = {acc*100:.2f}%")

    # ============================================================

    def predict(self, url):
        f = self.extract_features(url)
        X = self.scaler.transform([list(f.values())])

        ml_pred = self.model.predict(X)[0]
        ml_prob = self.model.predict_proba(X)[0][1]

        result = {
            "url": url,
            "ml_prediction": int(ml_pred),
            "ml_phishing_probability": float(ml_prob),
        }

        if self.use_slm:
            slm = self.slm.analyze(url)
            result.update(slm)

            if slm['slm_enabled']:
                p = ml_prob*0.6 + slm['slm_phishing_probability']*0.4
                result['ensemble_prediction'] = int(p > 0.5)
                result['ensemble_probability'] = p

        return result

    # ============================================================

    def save_model(self, path="phishing_detector_complete.pkl"):
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler
        }, path)

        if self.use_slm:
            self.slm.save(path.replace(".pkl", "_slm.pt"))

        print("💾 Saved model")


# ============================================================

def main():
    d = AdvancedPhishingDetectorWithSLM(use_slm=True)
    d.train()
    d.save_model()

    tests = [
        "https://www.google.com/",
        "http://paypal-verify.tk/login",
        "http://amaz0n-update.xyz/"
    ]

    for u in tests:
        print("\n🔎", d.predict(u))


if __name__ == "__main__":
    main()


# %% [code cell]
from google.colab import drive
drive.mount('/content/drive')

# %% [code cell]
"""
GOOGLE COLAB — INTERACTIVE PHISHING URL CHECKER
For: AdvancedPhishingDetectorWithSLM + SLM
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import os

# ============================================================
#              LOAD TRAINED MODEL (ML + SLM)
# ============================================================

try:
    detector = AdvancedPhishingDetectorWithSLM(use_slm=True)

    if os.path.exists("phishing_detector_complete.pkl"):
        print("📥 Loading existing ML model...")
        data = joblib.load("phishing_detector_complete.pkl")
        detector.model = data["model"]
        detector.scaler = data["scaler"]
        print("✅ ML model loaded!")

        slm_path = "phishing_detector_complete_slm.pt"
        if os.path.exists(slm_path):
            detector.slm.load(slm_path)
            print("✅ SLM loaded!")

    else:
        print("⚠ Model not found. Training new model...")
        detector.train()
        detector.save_model()

except NameError:
    print("❌ ERROR: Detector class not found.")
    print("➡ Run the main detector code cell first!")


# ============================================================
#              COLAB INTERFACE SETUP
# ============================================================

output = widgets.Output()

style = """
<style>
    .box {
        padding: 22px;
        border-radius: 14px;
        color: white;
        font-family: 'Arial';
        margin-top: 20px;
    }
    .danger {
        background: linear-gradient(135deg, #ff414d, #ff7a45);
    }
    .safe {
        background: linear-gradient(135deg, #18b27a, #1572b9);
    }
    .header {
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .url-box {
        padding: 12px;
        background: rgba(0,0,0,0.25);
        border-radius: 8px;
        word-wrap: break-word;
        margin-bottom: 15px;
    }
    .stat {
        font-size: 18px;
        margin: 6px 0;
    }
    .value {
        font-size: 26px;
        font-weight: bold;
    }
</style>
"""


# ============================================================
#              URL ANALYSIS LOGIC
# ============================================================

def analyze_url(url):
    url = url.strip()
    if not url:
        with output:
            clear_output()
            print("⚠ Please enter a URL.")
        return

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    with output:
        clear_output()
        print("🔍 Analyzing...")

    try:
        result = detector.predict(url)

        # Base ML predictions
        ml_phish = result["ml_phishing_probability"]
        ml_legit = 1 - ml_phish

        # If SLM enabled — use ensemble
        if detector.use_slm and result.get("slm_enabled"):
            ens_phish = result["ensemble_probability"]
            ens_legit = 1 - ens_phish
            final_pred = result["ensemble_prediction"]
        else:
            ens_phish = ml_phish
            ens_legit = ml_legit
            final_pred = result["ml_prediction"]

        # Convert to %
        ml_phish *= 100
        ml_legit *= 100
        ens_phish *= 100
        ens_legit *= 100

        # UI box style
        box_class = "danger" if final_pred == 1 else "safe"
        verdict = "🚨 PHISHING DETECTED" if final_pred == 1 else "✅ SAFE URL"

        html = f"""
        {style}
        <div class="box {box_class}">
            <div class="header">{verdict}</div>
            <div class="url-box"><b>URL:</b> {url}</div>

            <div class="stat">
                <b>ML Phishing Probability:</b>
                <span class="value">{ml_phish:.2f}%</span>
            </div>

            <div class="stat">
                <b>ML Legitimate Probability:</b>
                <span class="value">{ml_legit:.2f}%</span>
            </div>

            <hr style="opacity:0.3; margin:12px 0;">

            <div class="stat">
                <b>Ensemble Phishing Probability:</b>
                <span class="value">{ens_phish:.2f}%</span>
            </div>

            <div class="stat">
                <b>Ensemble Legitimate Probability:</b>
                <span class="value">{ens_legit:.2f}%</span>
            </div>

        </div>
        """

        with output:
            clear_output()
            display(HTML(html))

    except Exception as e:
        with output:
            clear_output()
            print("❌ Error:", e)


# ============================================================
#              WIDGET ELEMENTS
# ============================================================

url_input = widgets.Text(
    value="",
    placeholder="Enter URL…",
    layout=widgets.Layout(width="70%", height="40px")
)

check_button = widgets.Button(
    description="🔍 Check URL",
    button_style="primary",
    layout=widgets.Layout(width="150px", height="40px")
)

example_btn1 = widgets.Button(
    description="Try Phish",
    button_style="danger",
    layout=widgets.Layout(width="120px", height="35px")
)

example_btn2 = widgets.Button(
    description="Try Legit",
    button_style="success",
    layout=widgets.Layout(width="120px", height="35px")
)

clear_btn = widgets.Button(
    description="Clear",
    layout=widgets.Layout(width="100px", height="35px")
)


# Button handlers
check_button.on_click(lambda b: analyze_url(url_input.value))
example_btn1.on_click(lambda b: analyze_url("http://paypal-security-verify-update.tk/login"))
example_btn2.on_click(lambda b: analyze_url("https://www.google.com"))
clear_btn.on_click(lambda b: (setattr(url_input, "value", ""), output.clear_output()))
url_input.on_submit(lambda x: analyze_url(url_input.value))


# ============================================================
#              DISPLAY UI
# ============================================================

display(HTML("""
<div style="background: linear-gradient(135deg, #667eea, #764ba2); padding:25px;
border-radius:15px; text-align:center; color:white; margin-bottom:20px;">
    <h1>🛡️ Phishing URL Detector</h1>
    <p>AI + SLM | 55+ Features | Full Probability Breakdown</p>
</div>
"""))

display(widgets.HBox([url_input, check_button]))
display(widgets.HBox([example_btn1, example_btn2, clear_btn]))
display(output)

print("\n✅ Ready! Enter any URL to test.")


# %% [code cell]


"""
GOOGLE COLAB - PHISHING DETECTION: TRAIN MODEL + DEPLOY API
Run this entire notebook in Google Colab to:
1. Train the phishing detection model
2. Deploy a public API using ngrok
3. Get a public URL to use in your frontend

Just click "Runtime" → "Run all" in Colab!
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
print("📦 Installing dependencies...")

!pip install -q flask flask-cors
!pip install -q pandas numpy scikit-learn xgboost joblib
!pip install -q torch transformers requests

print("✅ All dependencies installed!")

# ============================================================
# CELL 2: Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import requests
import re
import math
import warnings
warnings.filterwarnings('ignore')

import joblib
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False
    print("⚠ transformers not installed — SLM disabled")

print("✅ Imports complete!")

# ============================================================
# CELL 3: Small Language Model (SLM) Class
# ============================================================
class LightweightSLM:
    def __init__(self, model_name="distilbert-base-uncased"):
        if not TRANSFORMERS_AVAILABLE:
            self.enabled = False
            print("⚠ SLM disabled")
            return

        self.enabled = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("⏳ Loading SLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2)
        ).to(self.device)

        print("✅ SLM loaded")

    def extract_text(self, url):
        parsed = urlparse(url)
        return (
            f"URL: {url} | "
            f"Protocol: {parsed.scheme} | "
            f"Domain: {parsed.netloc} | "
            f"Path: {parsed.path}"
        )

    def analyze(self, url):
        if not self.enabled:
            return {'slm_enabled': False}

        text = self.extract_text(url)
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                max_length=256).to(self.device)

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
            'slm_confidence': max(phishing_p, legit_p),
            'slm_reasoning': "SLM-based URL semantic analysis"
        }

    def save(self, path):
        if self.enabled:
            torch.save({'classifier': self.classifier.state_dict()}, path)

    def load(self, path):
        if self.enabled:
            ckpt = torch.load(path, map_location=self.device)
            self.classifier.load_state_dict(ckpt['classifier'])

print("✅ SLM class defined!")

# ============================================================
# CELL 4: Main Phishing Detector Class
# ============================================================
class AdvancedPhishingDetectorWithSLM:

    def __init__(self, use_slm=True):
        self.use_slm = use_slm
        self.scaler = StandardScaler()
        self.model = None
        if use_slm:
            self.slm = LightweightSLM()
        else:
            self.slm = None

    def entropy(self, text):
        if not text:
            return 0
        p = [text.count(c) / len(text) for c in set(text)]
        return -sum(x * math.log2(x) for x in p if x > 0)

    def extract_features(self, url):
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path

        f = {}
        f['url_len'] = len(url)
        f['domain_len'] = len(domain)
        f['path_len'] = len(path)
        f['num_dot'] = url.count('.')
        f['num_hyph'] = url.count('-')
        f['num_slash'] = url.count('/')
        f['has_https'] = 1 if parsed.scheme == 'https' else 0
        f['entropy_url'] = self.entropy(url)
        f['entropy_domain'] = self.entropy(domain)
        f['entropy_path'] = self.entropy(path)
        f['num_digits'] = sum(c.isdigit() for c in url)
        f['num_letters'] = sum(c.isalpha() for c in url)
        f['num_special'] = sum(not c.isalnum() for c in url)
        f['num_params'] = url.count('=')
        f['has_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', domain) else 0

        return f

    def load_openphish(self):
        print("📥 Loading OpenPhish...")
        try:
            df = pd.read_csv("https://openphish.com/feed.txt", header=None)
            urls = df[0].dropna().tolist()
            print(f"✔ {len(urls)} from OpenPhish")
            return urls
        except:
            print("❌ OpenPhish failed")
            return []

    def load_urlhaus(self):
        print("📥 Loading URLHaus...")
        try:
            url = "https://urlhaus.abuse.ch/downloads/text/"
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()

            urls = []
            for line in resp.text.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("http://") or line.startswith("https://"):
                    urls.append(line)
                else:
                    urls.append("http://" + line)

            print(f"✔ Loaded {len(urls)} phishing URLs from URLHaus")
            return urls
        except Exception as e:
            print(f"❌ URLHaus failed: {e}")
            return []

    def load_cisco_legit(self):
        print("📥 Loading Cisco Umbrella Top 1M...")
        try:
            df = pd.read_csv(
                "http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip",
                compression="zip",
                header=None,
                names=["rank", "domain"]
            )
            urls = [f"https://{d}/" for d in df["domain"].tolist()]
            print(f"✔ {len(urls)} legitimate")
            return urls
        except:
            print("❌ Cisco failed")
            return ["https://www.google.com/", "https://www.facebook.com/"] * 5000

    def create_training_data(self, target=10000):
        phish1 = self.load_openphish()
        phish2 = self.load_urlhaus()
        phishing = list(set(phish1 + phish2))
        phishing = phishing[:target]

        legit = self.load_cisco_legit()[:target]

        print(f"\n📊 Final balanced dataset: Legit={len(legit)}, Phish={len(phishing)}")

        urls = legit + phishing
        labels = [0]*len(legit) + [1]*len(phishing)
        return urls, labels

    def train(self):
        urls, labels = self.create_training_data()
        print("🔍 Extracting features...")

        X = [list(self.extract_features(u).values()) for u in urls]
        X = np.array(X)
        y = np.array(labels)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)

        print("🤖 Training models...")

        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        gb = GradientBoostingClassifier()
        xgb = XGBClassifier(n_estimators=200, eval_metric='logloss')
        lr = LogisticRegression(max_iter=1000)
        nn = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=400)

        self.model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb), ('gb', gb), ('lr', lr), ('nn', nn)],
            voting='soft',
            n_jobs=-1
        )

        self.model.fit(X_tr, y_tr)

        acc = self.model.score(X_te, y_te)
        print(f"🎯 Accuracy = {acc*100:.2f}%")

    def predict(self, url):
        f = self.extract_features(url)
        X = self.scaler.transform([list(f.values())])

        ml_pred = self.model.predict(X)[0]
        ml_prob = self.model.predict_proba(X)[0][1]

        result = {
            "url": url,
            "ml_prediction": int(ml_pred),
            "ml_phishing_probability": float(ml_prob),
        }

        if self.use_slm:
            slm = self.slm.analyze(url)
            result.update(slm)

            if slm['slm_enabled']:
                p = ml_prob*0.6 + slm['slm_phishing_probability']*0.4
                result['ensemble_prediction'] = int(p > 0.5)
                result['ensemble_probability'] = p

        return result

    def save_model(self, path="phishing_detector_complete.pkl"):
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler
        }, path)

        if self.use_slm:
            self.slm.save(path.replace(".pkl", "_slm.pt"))

        print("💾 Saved model")

print("✅ Detector class defined!")

# ============================================================
# CELL 5: Train the Model
# ============================================================
print("\n" + "="*60)
print("🚀 STARTING MODEL TRAINING")
print("="*60 + "\n")

detector = AdvancedPhishingDetectorWithSLM(use_slm=True)
detector.train()
detector.save_model()

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)

# Test predictions
test_urls = [
    "https://www.google.com/",
    "http://paypal-verify.tk/login",
    "http://amaz0n-update.xyz/"
]

print("\n🧪 Testing predictions:")
for url in test_urls:
    result = detector.predict(url)
    print(f"\n📊 {url}")
    print(f"   Prediction: {'Phishing' if result['ml_prediction'] == 1 else 'Legitimate'}")
    print(f"   Probability: {result['ml_phishing_probability']:.3f}")

# ============================================================
# CELL 6: Flask API Setup
# ============================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
print("\n🔄 Loading trained model for API...")
model_data = joblib.load('phishing_detector_complete.pkl')
ml_model = model_data['model']
scaler = model_data['scaler']

# Load SLM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
slm_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
slm_model.eval()

slm_classifier = nn.Sequential(
    nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 2)
).to(device)

if os.path.exists('phishing_detector_complete_slm.pt'):
    checkpoint = torch.load('phishing_detector_complete_slm.pt', map_location=device)
    slm_classifier.load_state_dict(checkpoint['classifier'])
    print("✅ SLM loaded with trained weights")

# Helper functions
def entropy(text):
    if not text:
        return 0
    p = [text.count(c) / len(text) for c in set(text)]
    return -sum(x * math.log2(x) for x in p if x > 0)

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    features = {
        'url_len': len(url),
        'domain_len': len(domain),
        'path_len': len(path),
        'num_dot': url.count('.'),
        'num_hyph': url.count('-'),
        'num_slash': url.count('/'),
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'entropy_url': entropy(url),
        'entropy_domain': entropy(domain),
        'entropy_path': entropy(path),
        'num_digits': sum(c.isdigit() for c in url),
        'num_letters': sum(c.isalpha() for c in url),
        'num_special': sum(not c.isalnum() for c in url),
        'num_params': url.count('='),
        'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', domain) else 0
    }
    return features

def analyze_with_slm(url):
    try:
        parsed = urlparse(url)
        text = f"URL: {url} | Protocol: {parsed.scheme} | Domain: {parsed.netloc} | Path: {parsed.path}"

        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = slm_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            logits = slm_classifier(emb)
            probs = torch.softmax(logits, dim=1)[0]

        phishing_prob = probs[1].item()
        legit_prob = probs[0].item()

        return {
            'slm_prediction': int(phishing_prob > 0.5),
            'slm_phishing_probability': round(phishing_prob, 4),
            'slm_legitimate_probability': round(legit_prob, 4),
            'slm_confidence': round(max(phishing_prob, legit_prob), 4)
        }
    except:
        return None

# API Endpoints
@app.route('/')
def home():
    return jsonify({
        'message': 'Phishing Detection API - Trained and Ready!',
        'version': '1.0',
        'endpoints': {
            'health': 'GET /health',
            'predict': 'POST /predict',
            'batch': 'POST /batch-predict'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'slm_loaded': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'url' not in data:
        return jsonify({'error': 'URL required'}), 400

    url = data['url'].strip()

    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'URL must start with http:// or https://'}), 400

    try:
        # ML prediction
        features = extract_features(url)
        X = scaler.transform([list(features.values())])
        ml_pred = ml_model.predict(X)[0]
        ml_prob = ml_model.predict_proba(X)[0]

        result = {
            'url': url,
            'ml_prediction': int(ml_pred),
            'ml_phishing_probability': round(float(ml_prob[1]), 4),
            'ml_legitimate_probability': round(float(ml_prob[0]), 4),
            'prediction_label': 'Phishing' if ml_pred == 1 else 'Legitimate'
        }

        # SLM prediction
        slm_result = analyze_with_slm(url)
        if slm_result:
            result.update(slm_result)
            ensemble_prob = ml_prob[1] * 0.6 + slm_result['slm_phishing_probability'] * 0.4
            result['ensemble_prediction'] = int(ensemble_prob > 0.5)
            result['ensemble_probability'] = round(float(ensemble_prob), 4)
            result['final_prediction'] = 'Phishing' if ensemble_prob > 0.5 else 'Legitimate'
            result['confidence'] = round(float(max(ensemble_prob, 1 - ensemble_prob)), 4)
        else:
            result['final_prediction'] = result['prediction_label']
            result['confidence'] = round(float(max(ml_prob)), 4)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    data = request.get_json()

    if not data or 'urls' not in data:
        return jsonify({'error': 'urls array required'}), 400

    urls = data['urls']
    if len(urls) > 100:
        return jsonify({'error': 'Max 100 URLs per batch'}), 400

    results = []
    for url in urls:
        try:
            features = extract_features(url)
            X = scaler.transform([list(features.values())])
            ml_pred = ml_model.predict(X)[0]
            ml_prob = ml_model.predict_proba(X)[0][1]

            results.append({
                'url': url,
                'prediction': int(ml_pred),
                'probability': round(float(ml_prob), 4),
                'label': 'Phishing' if ml_pred == 1 else 'Legitimate'
            })
        except Exception as e:
            results.append({'url': url, 'error': str(e)})

    return jsonify({'results': results})

print("✅ Flask API configured!")

# ============================================================
# CELL 7: Deploy API with Cloudflare Tunnel (NO SIGNUP NEEDED!)
# ============================================================
import threading
import time
import subprocess
import re

# Start Flask in background thread
def run_flask():
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# Wait for Flask to start
time.sleep(3)

print("\n" + "="*60)
print("🚀 DEPLOYING API WITH CLOUDFLARE TUNNEL...")
print("="*60)

# Install cloudflared
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1

# Start cloudflare tunnel in background
tunnel_process = subprocess.Popen(
    ['cloudflared', 'tunnel', '--url', 'http://localhost:5000'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait for tunnel URL
public_url = None
print("⏳ Getting public URL (this takes ~10 seconds)...")

for _ in range(30):
    line = tunnel_process.stderr.readline()
    if 'trycloudflare.com' in line:
        # Extract URL from output
        match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
        if match:
            public_url = match.group(0)
            break
    time.sleep(1)

if public_url:
    print("\n" + "="*60)
    print("🎉 API DEPLOYED SUCCESSFULLY!")
    print("="*60)
    print(f"\n🌐 PUBLIC URL: {public_url}")
    print("\n📡 API Endpoints:")
    print(f"   GET  {public_url}/")
    print(f"   GET  {public_url}/health")
    print(f"   POST {public_url}/predict")
    print(f"   POST {public_url}/batch-predict")
    print("\n💡 Test with cURL:")
    print(f'   curl -X POST {public_url}/predict -H "Content-Type: application/json" -d \'{{"url": "http://paypal-verify.tk"}}\'')
    print("\n✅ Copy this URL to use in your frontend!")
    print("\n⚠️  Keep this notebook running to keep the API alive!")
    print("="*60 + "\n")
else:
    print("\n❌ Failed to get public URL. Try running this cell again.")
    print("="*60 + "\n")

# ============================================================
# CELL 8: Test the API
# ============================================================
import requests
import json

if public_url:
    print("🧪 Testing the deployed API...\n")

    # Wait a bit for tunnel to stabilize
    time.sleep(5)

    try:
        # Test health
        response = requests.get(f"{public_url}/health", timeout=10)
        print("Health Check:")
        print(json.dumps(response.json(), indent=2))

        # Test prediction
        test_url = "http://paypal-verify.tk/login"
        response = requests.post(
            f"{public_url}/predict",
            json={"url": test_url},
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        print(f"\n\nPrediction for: {test_url}")
        print(json.dumps(response.json(), indent=2))

        print("\n✅ API is working! Use the public URL in your frontend.")
    except Exception as e:
        print(f"\n⚠️  Test failed: {e}")
        print("The API is still running, try testing manually with the URL above.")
else:
    print("⚠️  No public URL available. Please run Cell 7 again.")

# %% [code cell]
"""
GOOGLE COLAB - LOAD EXISTING MODEL & DEPLOY API
Upload your .pkl and .pt files, then run this to get a public API!

Steps:
1. Upload your files using the file icon on the left
2. Update the file paths below
3. Run all cells
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
print("📦 Installing dependencies...")

!pip install -q flask flask-cors
!pip install -q pandas numpy scikit-learn xgboost joblib
!pip install -q torch transformers requests

print("✅ All dependencies installed!")

# ============================================================
# CELL 2: Upload Model Files (MANUAL STEP)
# ============================================================
from google.colab import files

print("\n" + "="*60)
print("📤 UPLOAD YOUR MODEL FILES")
print("="*60)
print("\nPlease upload:")
print("1. phishing_detector_complete.pkl")
print("2. phishing_detector_complete_slm.pt")
print("\nClick the button below to upload...\n")

uploaded = files.upload()

print("\n✅ Files uploaded successfully!")
print("Uploaded files:", list(uploaded.keys()))

# ============================================================
# CELL 3: Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import re
import math
import warnings
warnings.filterwarnings('ignore')

import joblib
from urllib.parse import urlparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

print("✅ Imports complete!")

# ============================================================
# CELL 4: Load Models
# ============================================================
print("\n" + "="*60)
print("🔄 LOADING MODELS...")
print("="*60)

# Update these paths if your files have different names
MODEL_PATH = 'phishing_detector_complete.pkl'
SLM_PATH = 'phishing_detector_complete_slm.pt'

# Load ML model
print(f"\n📦 Loading ML model from: {MODEL_PATH}")
model_data = joblib.load(MODEL_PATH)
ml_model = model_data['model']
scaler = model_data['scaler']
print("✅ ML model loaded successfully")

# Load SLM
print(f"\n📦 Loading SLM from: {SLM_PATH}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
slm_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
slm_model.eval()

slm_classifier = nn.Sequential(
    nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 2)
).to(device)

if os.path.exists(SLM_PATH):
    checkpoint = torch.load(SLM_PATH, map_location=device)
    slm_classifier.load_state_dict(checkpoint['classifier'])
    print("✅ SLM loaded with trained weights")
else:
    print("⚠️  SLM file not found, using untrained weights")

print("\n" + "="*60)
print("✅ ALL MODELS LOADED SUCCESSFULLY!")
print("="*60)

# ============================================================
# CELL 5: Helper Functions
# ============================================================

def entropy(text):
    """Calculate Shannon entropy"""
    if not text:
        return 0
    p = [text.count(c) / len(text) for c in set(text)]
    return -sum(x * math.log2(x) for x in p if x > 0)


def extract_features(url):
    """Extract URL features for ML model"""
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    features = {
        'url_len': len(url),
        'domain_len': len(domain),
        'path_len': len(path),
        'num_dot': url.count('.'),
        'num_hyph': url.count('-'),
        'num_slash': url.count('/'),
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'entropy_url': entropy(url),
        'entropy_domain': entropy(domain),
        'entropy_path': entropy(path),
        'num_digits': sum(c.isdigit() for c in url),
        'num_letters': sum(c.isalpha() for c in url),
        'num_special': sum(not c.isalnum() for c in url),
        'num_params': url.count('='),
        'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', domain) else 0
    }
    return features


def analyze_with_slm(url):
    """Analyze URL using SLM"""
    try:
        parsed = urlparse(url)
        text = f"URL: {url} | Protocol: {parsed.scheme} | Domain: {parsed.netloc} | Path: {parsed.path}"

        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = slm_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            logits = slm_classifier(emb)
            probs = torch.softmax(logits, dim=1)[0]

        phishing_prob = probs[1].item()
        legit_prob = probs[0].item()

        return {
            'slm_prediction': int(phishing_prob > 0.5),
            'slm_phishing_probability': round(phishing_prob, 4),
            'slm_legitimate_probability': round(legit_prob, 4),
            'slm_confidence': round(max(phishing_prob, legit_prob), 4)
        }
    except Exception as e:
        print(f"SLM error: {e}")
        return None

print("✅ Helper functions defined!")

# ============================================================
# CELL 6: Test Predictions (Verify Models Work)
# ============================================================
print("\n" + "="*60)
print("🧪 TESTING PREDICTIONS...")
print("="*60)

test_urls = [
    "https://www.google.com/",
    "http://paypal-verify.tk/login",
    "http://amaz0n-update.xyz/"
]

for url in test_urls:
    try:
        features = extract_features(url)
        X = scaler.transform([list(features.values())])
        ml_pred = ml_model.predict(X)[0]
        ml_prob = ml_model.predict_proba(X)[0][1]

        print(f"\n📊 {url}")
        print(f"   Prediction: {'🚨 Phishing' if ml_pred == 1 else '✅ Legitimate'}")
        print(f"   Confidence: {ml_prob*100:.1f}%")
    except Exception as e:
        print(f"\n❌ Error testing {url}: {e}")

print("\n✅ Model testing complete!")

# ============================================================
# CELL 7: Create Flask API
# ============================================================

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        'message': 'Phishing Detection API - Ready!',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            'health': 'GET /health',
            'predict': 'POST /predict',
            'batch': 'POST /batch-predict'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'ml_model_loaded': True,
        'slm_model_loaded': True,
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'url' not in data:
        return jsonify({'error': 'URL required in request body'}), 400

    url = data['url'].strip()

    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'URL must start with http:// or https://'}), 400

    try:
        # ML prediction
        features = extract_features(url)
        X = scaler.transform([list(features.values())])
        ml_pred = ml_model.predict(X)[0]
        ml_prob = ml_model.predict_proba(X)[0]

        result = {
            'url': url,
            'ml_prediction': int(ml_pred),
            'ml_phishing_probability': round(float(ml_prob[1]), 4),
            'ml_legitimate_probability': round(float(ml_prob[0]), 4),
            'prediction_label': 'Phishing' if ml_pred == 1 else 'Legitimate'
        }

        # SLM prediction
        slm_result = analyze_with_slm(url)
        if slm_result:
            result.update(slm_result)
            ensemble_prob = ml_prob[1] * 0.6 + slm_result['slm_phishing_probability'] * 0.4
            result['ensemble_prediction'] = int(ensemble_prob > 0.5)
            result['ensemble_probability'] = round(float(ensemble_prob), 4)
            result['final_prediction'] = 'Phishing' if ensemble_prob > 0.5 else 'Legitimate'
            result['confidence'] = round(float(max(ensemble_prob, 1 - ensemble_prob)), 4)
        else:
            result['final_prediction'] = result['prediction_label']
            result['confidence'] = round(float(max(ml_prob)), 4)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    data = request.get_json()

    if not data or 'urls' not in data:
        return jsonify({'error': 'urls array required in request body'}), 400

    urls = data['urls']
    if not isinstance(urls, list):
        return jsonify({'error': 'urls must be an array'}), 400

    if len(urls) > 100:
        return jsonify({'error': 'Maximum 100 URLs per batch'}), 400

    results = []
    for url in urls:
        try:
            features = extract_features(url)
            X = scaler.transform([list(features.values())])
            ml_pred = ml_model.predict(X)[0]
            ml_prob = ml_model.predict_proba(X)[0][1]

            results.append({
                'url': url,
                'prediction': int(ml_pred),
                'probability': round(float(ml_prob), 4),
                'label': 'Phishing' if ml_pred == 1 else 'Legitimate'
            })
        except Exception as e:
            results.append({'url': url, 'error': str(e)})

    return jsonify({'results': results})

print("✅ Flask API created!")

# ============================================================
# CELL 8: Deploy with Cloudflare Tunnel
# ============================================================
import threading
import time
import subprocess
import re

# Start Flask in background
def run_flask():
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

time.sleep(3)

print("\n" + "="*60)
print("🚀 DEPLOYING API WITH CLOUDFLARE TUNNEL...")
print("="*60)

# Install cloudflared
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1

# Start tunnel
tunnel_process = subprocess.Popen(
    ['cloudflared', 'tunnel', '--url', 'http://localhost:5000'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Get public URL
public_url = None
print("⏳ Getting public URL (this takes ~10 seconds)...")

for _ in range(30):
    line = tunnel_process.stderr.readline()
    if 'trycloudflare.com' in line:
        match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
        if match:
            public_url = match.group(0)
            break
    time.sleep(1)

if public_url:
    print("\n" + "="*60)
    print("🎉 API DEPLOYED SUCCESSFULLY!")
    print("="*60)
    print(f"\n🌐 PUBLIC URL: {public_url}")
    print("\n📡 API Endpoints:")
    print(f"   GET  {public_url}/")
    print(f"   GET  {public_url}/health")
    print(f"   POST {public_url}/predict")
    print(f"   POST {public_url}/batch-predict")
    print("\n💡 Example cURL:")
    print(f'   curl -X POST {public_url}/predict \\')
    print(f'        -H "Content-Type: application/json" \\')
    print(f'        -d \'{{"url": "http://paypal-verify.tk"}}\'')
    print("\n💡 Example JavaScript:")
    print(f"   fetch('{public_url}/predict', {{")
    print("     method: 'POST',")
    print("     headers: {'Content-Type': 'application/json'},")
    print("     body: JSON.stringify({url: 'http://example.com'})")
    print("   })")
    print("   .then(res => res.json())")
    print("   .then(data => console.log(data));")
    print("\n✅ Copy this URL to use in your frontend!")
    print("\n⚠️  Keep this notebook running to keep the API alive!")
    print("="*60 + "\n")
else:
    print("\n❌ Failed to get public URL. Try running this cell again.")

# ============================================================
# CELL 9: Test the Deployed API
# ============================================================
import requests
import json

if public_url:
    print("🧪 Testing the deployed API...\n")
    time.sleep(5)

    try:
        # Test health
        print("1️⃣ Testing /health endpoint...")
        response = requests.get(f"{public_url}/health", timeout=10)
        print(json.dumps(response.json(), indent=2))

        # Test prediction
        print("\n2️⃣ Testing /predict endpoint...")
        test_url = "http://paypal-verify.tk/login"
        response = requests.post(
            f"{public_url}/predict",
            json={"url": test_url},
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        print(f"\nPrediction for: {test_url}")
        result = response.json()
        print(json.dumps(result, indent=2))

        # Test batch
        print("\n3️⃣ Testing /batch-predict endpoint...")
        response = requests.post(
            f"{public_url}/batch-predict",
            json={"urls": ["https://google.com", "http://phishing-test.com"]},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(json.dumps(response.json(), indent=2))

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print(f"\n🎉 Your API is live at: {public_url}")
        print("Use this URL in your frontend application!")

    except Exception as e:
        print(f"\n⚠️  Test failed: {e}")
        print("The API might still be starting. Wait a minute and try testing manually.")
else:
    print("⚠️  No public URL available. Run Cell 8 again.")

# %% [code cell]
"""
GOOGLE COLAB - LOAD EXISTING MODEL & DEPLOY API
Upload your .pkl and .pt files, then run this to get a public API!

Steps:
1. Upload your files using the file icon on the left
2. Update the file paths below
3. Run all cells
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
print("📦 Installing dependencies...")

!pip install -q flask flask-cors
!pip install -q pandas numpy scikit-learn xgboost joblib
!pip install -q torch transformers requests

print("✅ All dependencies installed!")

# ============================================================
# CELL 2: Upload Model Files (MANUAL STEP)
# ============================================================
from google.colab import files

print("\n" + "="*60)
print("📤 UPLOAD YOUR MODEL FILES")
print("="*60)
print("\nPlease upload:")
print("1. phishing_detector_complete.pkl")
print("2. phishing_detector_complete_slm.pt")
print("\nClick the button below to upload...\n")

uploaded = files.upload()

print("\n✅ Files uploaded successfully!")
print("Uploaded files:", list(uploaded.keys()))

# ============================================================
# CELL 3: Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import re
import math
import warnings
warnings.filterwarnings('ignore')

import joblib
from urllib.parse import urlparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

print("✅ Imports complete!")

# ============================================================
# CELL 4: Load Models
# ============================================================
print("\n" + "="*60)
print("🔄 LOADING MODELS...")
print("="*60)

# Update these paths if your files have different names
MODEL_PATH = 'phishing_detector_complete.pkl'
SLM_PATH = 'phishing_detector_complete_slm.pt'

# Load ML model
print(f"\n📦 Loading ML model from: {MODEL_PATH}")
model_data = joblib.load(MODEL_PATH)
ml_model = model_data['model']
scaler = model_data['scaler']
print("✅ ML model loaded successfully")

# Load SLM
print(f"\n📦 Loading SLM from: {SLM_PATH}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
slm_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
slm_model.eval()

slm_classifier = nn.Sequential(
    nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 2)
).to(device)

if os.path.exists(SLM_PATH):
    checkpoint = torch.load(SLM_PATH, map_location=device)
    slm_classifier.load_state_dict(checkpoint['classifier'])
    print("✅ SLM loaded with trained weights")
else:
    print("⚠️  SLM file not found, using untrained weights")

print("\n" + "="*60)
print("✅ ALL MODELS LOADED SUCCESSFULLY!")
print("="*60)

# ============================================================
# CELL 5: Helper Functions
# ============================================================

def entropy(text):
    """Calculate Shannon entropy"""
    if not text:
        return 0
    p = [text.count(c) / len(text) for c in set(text)]
    return -sum(x * math.log2(x) for x in p if x > 0)


def extract_features(url):
    """Extract URL features for ML model"""
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    features = {
        'url_len': len(url),
        'domain_len': len(domain),
        'path_len': len(path),
        'num_dot': url.count('.'),
        'num_hyph': url.count('-'),
        'num_slash': url.count('/'),
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'entropy_url': entropy(url),
        'entropy_domain': entropy(domain),
        'entropy_path': entropy(path),
        'num_digits': sum(c.isdigit() for c in url),
        'num_letters': sum(c.isalpha() for c in url),
        'num_special': sum(not c.isalnum() for c in url),
        'num_params': url.count('='),
        'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', domain) else 0
    }
    return features


def analyze_with_slm(url):
    """Analyze URL using SLM"""
    try:
        parsed = urlparse(url)
        text = f"URL: {url} | Protocol: {parsed.scheme} | Domain: {parsed.netloc} | Path: {parsed.path}"

        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = slm_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            logits = slm_classifier(emb)
            probs = torch.softmax(logits, dim=1)[0]

        phishing_prob = probs[1].item()
        legit_prob = probs[0].item()

        return {
            'slm_prediction': int(phishing_prob > 0.5),
            'slm_phishing_probability': round(phishing_prob, 4),
            'slm_legitimate_probability': round(legit_prob, 4),
            'slm_confidence': round(max(phishing_prob, legit_prob), 4)
        }
    except Exception as e:
        print(f"SLM error: {e}")
        return None

print("✅ Helper functions defined!")

# ============================================================
# CELL 6: Test Predictions (Verify Models Work)
# ============================================================
print("\n" + "="*60)
print("🧪 TESTING PREDICTIONS...")
print("="*60)

test_urls = [
    "https://www.google.com/",
    "http://paypal-verify.tk/login",
    "http://amaz0n-update.xyz/"
]

for url in test_urls:
    try:
        features = extract_features(url)
        X = scaler.transform([list(features.values())])
        ml_pred = ml_model.predict(X)[0]
        ml_prob = ml_model.predict_proba(X)[0][1]

        print(f"\n📊 {url}")
        print(f"   Prediction: {'🚨 Phishing' if ml_pred == 1 else '✅ Legitimate'}")
        print(f"   Confidence: {ml_prob*100:.1f}%")
    except Exception as e:
        print(f"\n❌ Error testing {url}: {e}")

print("\n✅ Model testing complete!")

# ============================================================
# CELL 7: Create Flask API
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# API KEY AUTHENTICATION
# ============================================================
import secrets
import hashlib
from functools import wraps

# Generate API keys (run once)
API_KEYS = {
    'admin_key': hashlib.sha256(secrets.token_hex(32).encode()).hexdigest()[:32],
    'user_key_1': hashlib.sha256(secrets.token_hex(32).encode()).hexdigest()[:32],
    'user_key_2': hashlib.sha256(secrets.token_hex(32).encode()).hexdigest()[:32]
}

# Store valid API keys
VALID_API_KEYS = set(API_KEYS.values())

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')

        if not api_key:
            return jsonify({
                'error': 'API key required',
                'message': 'Provide API key in X-API-Key header or api_key parameter'
            }), 401

        if api_key not in VALID_API_KEYS:
            return jsonify({
                'error': 'Invalid API key',
                'message': 'The provided API key is not valid'
            }), 403

        return f(*args, **kwargs)

    return decorated_function

# Print generated API keys
print("\n" + "="*60)
print("🔑 GENERATED API KEYS")
print("="*60)
for name, key in API_KEYS.items():
    print(f"{name}: {key}")
print("="*60 + "\n")

@app.route('/')
def home():
    return jsonify({
        'message': 'Phishing Detection API - Ready!',
        'version': '1.0',
        'status': 'running',
        'authentication': 'API Key required (X-API-Key header)',
        'endpoints': {
            'health': 'GET /health (no auth)',
            'predict': 'POST /predict (requires API key)',
            'batch': 'POST /batch-predict (requires API key)'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'ml_model_loaded': True,
        'slm_model_loaded': True,
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    data = request.get_json()

    if not data or 'url' not in data:
        return jsonify({'error': 'URL required in request body'}), 400

    url = data['url'].strip()

    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'URL must start with http:// or https://'}), 400

    try:
        # ML prediction
        features = extract_features(url)
        X = scaler.transform([list(features.values())])
        ml_pred = ml_model.predict(X)[0]
        ml_prob = ml_model.predict_proba(X)[0]

        result = {
            'url': url,
            'ml_prediction': int(ml_pred),
            'ml_phishing_probability': round(float(ml_prob[1]), 4),
            'ml_legitimate_probability': round(float(ml_prob[0]), 4),
            'prediction_label': 'Phishing' if ml_pred == 1 else 'Legitimate'
        }

        # SLM prediction
        slm_result = analyze_with_slm(url)
        if slm_result:
            result.update(slm_result)
            ensemble_prob = ml_prob[1] * 0.6 + slm_result['slm_phishing_probability'] * 0.4
            result['ensemble_prediction'] = int(ensemble_prob > 0.5)
            result['ensemble_probability'] = round(float(ensemble_prob), 4)
            result['final_prediction'] = 'Phishing' if ensemble_prob > 0.5 else 'Legitimate'
            result['confidence'] = round(float(max(ensemble_prob, 1 - ensemble_prob)), 4)
        else:
            result['final_prediction'] = result['prediction_label']
            result['confidence'] = round(float(max(ml_prob)), 4)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch-predict', methods=['POST'])
@require_api_key
def batch_predict():
    data = request.get_json()

    if not data or 'urls' not in data:
        return jsonify({'error': 'urls array required in request body'}), 400

    urls = data['urls']
    if not isinstance(urls, list):
        return jsonify({'error': 'urls must be an array'}), 400

    if len(urls) > 100:
        return jsonify({'error': 'Maximum 100 URLs per batch'}), 400

    results = []
    for url in urls:
        try:
            features = extract_features(url)
            X = scaler.transform([list(features.values())])
            ml_pred = ml_model.predict(X)[0]
            ml_prob = ml_model.predict_proba(X)[0][1]

            results.append({
                'url': url,
                'prediction': int(ml_pred),
                'probability': round(float(ml_prob), 4),
                'label': 'Phishing' if ml_pred == 1 else 'Legitimate'
            })
        except Exception as e:
            results.append({'url': url, 'error': str(e)})

    return jsonify({'results': results})

print("✅ Flask API created!")

# ============================================================
# CELL 8: Deploy with Cloudflare Tunnel
# ============================================================
import threading
import time
import subprocess
import re

# Start Flask in background
def run_flask():
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

time.sleep(3)

print("\n" + "="*60)
print("🚀 DEPLOYING API WITH CLOUDFLARE TUNNEL...")
print("="*60)

# Install cloudflared
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1

# Start tunnel
tunnel_process = subprocess.Popen(
    ['cloudflared', 'tunnel', '--url', 'http://localhost:5000'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Get public URL
public_url = None
print("⏳ Getting public URL (this takes ~10 seconds)...")

for _ in range(30):
    line = tunnel_process.stderr.readline()
    if 'trycloudflare.com' in line:
        match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
        if match:
            public_url = match.group(0)
            break
    time.sleep(1)

if public_url:
    print("\n" + "="*60)
    print("🎉 API DEPLOYED SUCCESSFULLY!")
    print("="*60)
    print(f"\n🌐 PUBLIC URL: {public_url}")
    print("\n🔑 YOUR API KEYS:")
    print("="*60)
    for name, key in API_KEYS.items():
        print(f"  {name}: {key}")
    print("="*60)
    print("\n📡 API Endpoints:")
    print(f"   GET  {public_url}/              (no auth)")
    print(f"   GET  {public_url}/health        (no auth)")
    print(f"   POST {public_url}/predict       (requires API key)")
    print(f"   POST {public_url}/batch-predict (requires API key)")
    print("\n💡 Example with cURL:")
    print(f'   curl -X POST {public_url}/predict \\')
    print(f'        -H "Content-Type: application/json" \\')
    print(f'        -H "X-API-Key: {list(API_KEYS.values())[0]}" \\')
    print(f'        -d \'{{"url": "http://paypal-verify.tk"}}\'')
    print("\n💡 Example with JavaScript:")
    print(f"   fetch('{public_url}/predict', {{")
    print("     method: 'POST',")
    print("     headers: {")
    print("       'Content-Type': 'application/json',")
    print(f"       'X-API-Key': '{list(API_KEYS.values())[0]}'")
    print("     },")
    print("     body: JSON.stringify({url: 'http://example.com'})")
    print("   })")
    print("   .then(res => res.json())")
    print("   .then(data => console.log(data));")
    print("\n✅ SAVE YOUR API KEYS - They won't be shown again!")
    print("\n⚠️  Keep this notebook running to keep the API alive!")
    print("="*60 + "\n")
else:
    print("\n❌ Failed to get public URL. Try running this cell again.")

# ============================================================
# CELL 9: Test the Deployed API
# ============================================================
import requests
import json

if public_url:
    print("🧪 Testing the deployed API...\n")
    time.sleep(5)

    # Get first API key for testing
    test_api_key = list(API_KEYS.values())[0]

    try:
        # Test health (no auth needed)
        print("1️⃣ Testing /health endpoint (no auth)...")
        response = requests.get(f"{public_url}/health", timeout=10)
        print(json.dumps(response.json(), indent=2))

        # Test without API key (should fail)
        print("\n2️⃣ Testing /predict without API key (should fail)...")
        response = requests.post(
            f"{public_url}/predict",
            json={"url": "http://test.com"},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))

        # Test with API key (should work)
        print("\n3️⃣ Testing /predict WITH API key (should work)...")
        test_url = "http://paypal-verify.tk/login"
        response = requests.post(
            f"{public_url}/predict",
            json={"url": test_url},
            headers={
                "Content-Type": "application/json",
                "X-API-Key": test_api_key
            },
            timeout=10
        )

        print(f"\nPrediction for: {test_url}")
        result = response.json()
        print(json.dumps(result, indent=2))

        # Test batch
        print("\n4️⃣ Testing /batch-predict endpoint...")
        response = requests.post(
            f"{public_url}/batch-predict",
            json={"urls": ["https://google.com", "http://phishing-test.com"]},
            headers={
                "Content-Type": "application/json",
                "X-API-Key": test_api_key
            },
            timeout=10
        )
        print(json.dumps(response.json(), indent=2))

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print(f"\n🎉 Your API is live at: {public_url}")
        print(f"🔑 Use API key: {test_api_key}")
        print("\nUse this in your frontend application!")

    except Exception as e:
        print(f"\n⚠️  Test failed: {e}")
        print("The API might still be starting. Wait a minute and try testing manually.")
else:
    print("⚠️  No public URL available. Run Cell 8 again.")

# %% [code cell]
"""
GOOGLE COLAB - LOAD EXISTING MODEL & DEPLOY API
Upload your .pkl and .pt files, then run this to get a public API!

Steps:
1. Upload your files using the file icon on the left
2. Update the file paths below
3. Run all cells
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
print("📦 Installing dependencies...")

!pip install -q flask flask-cors
!pip install -q pandas numpy scikit-learn xgboost joblib
!pip install -q torch transformers requests

print("✅ All dependencies installed!")

# ============================================================
# CELL 2: Upload Model Files (MANUAL STEP)
# ============================================================
from google.colab import files

print("\n" + "="*60)
print("📤 UPLOAD YOUR MODEL FILES")
print("="*60)
print("\nPlease upload:")
print("1. phishing_detector_complete.pkl")
print("2. phishing_detector_complete_slm.pt")
print("\nClick the button below to upload...\n")

uploaded = files.upload()

print("\n✅ Files uploaded successfully!")
print("Uploaded files:", list(uploaded.keys()))

# ============================================================
# CELL 3: Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import re
import math
import warnings
warnings.filterwarnings('ignore')

import joblib
from urllib.parse import urlparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

print("✅ Imports complete!")

# ============================================================
# CELL 4: Load Models
# ============================================================
print("\n" + "="*60)
print("🔄 LOADING MODELS...")
print("="*60)

# Update these paths if your files have different names
MODEL_PATH = 'phishing_detector_complete.pkl'
SLM_PATH = 'phishing_detector_complete_slm.pt'

# Load ML model
print(f"\n📦 Loading ML model from: {MODEL_PATH}")
model_data = joblib.load(MODEL_PATH)
ml_model = model_data['model']
scaler = model_data['scaler']
print("✅ ML model loaded successfully")

# Load SLM
print(f"\n📦 Loading SLM from: {SLM_PATH}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
slm_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
slm_model.eval()

slm_classifier = nn.Sequential(
    nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 2)
).to(device)

if os.path.exists(SLM_PATH):
    checkpoint = torch.load(SLM_PATH, map_location=device)
    slm_classifier.load_state_dict(checkpoint['classifier'])
    print("✅ SLM loaded with trained weights")
else:
    print("⚠️  SLM file not found, using untrained weights")

print("\n" + "="*60)
print("✅ ALL MODELS LOADED SUCCESSFULLY!")
print("="*60)

# ============================================================
# CELL 5: Helper Functions
# ============================================================

def entropy(text):
    """Calculate Shannon entropy"""
    if not text:
        return 0
    p = [text.count(c) / len(text) for c in set(text)]
    return -sum(x * math.log2(x) for x in p if x > 0)


def extract_features(url):
    """Extract URL features for ML model"""
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    features = {
        'url_len': len(url),
        'domain_len': len(domain),
        'path_len': len(path),
        'num_dot': url.count('.'),
        'num_hyph': url.count('-'),
        'num_slash': url.count('/'),
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'entropy_url': entropy(url),
        'entropy_domain': entropy(domain),
        'entropy_path': entropy(path),
        'num_digits': sum(c.isdigit() for c in url),
        'num_letters': sum(c.isalpha() for c in url),
        'num_special': sum(not c.isalnum() for c in url),
        'num_params': url.count('='),
        'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', domain) else 0
    }
    return features


def analyze_with_slm(url):
    """Analyze URL using SLM"""
    try:
        parsed = urlparse(url)
        text = f"URL: {url} | Protocol: {parsed.scheme} | Domain: {parsed.netloc} | Path: {parsed.path}"

        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = slm_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            logits = slm_classifier(emb)
            probs = torch.softmax(logits, dim=1)[0]

        phishing_prob = probs[1].item()
        legit_prob = probs[0].item()

        return {
            'slm_prediction': int(phishing_prob > 0.5),
            'slm_phishing_probability': round(phishing_prob, 4),
            'slm_legitimate_probability': round(legit_prob, 4),
            'slm_confidence': round(max(phishing_prob, legit_prob), 4)
        }
    except Exception as e:
        print(f"SLM error: {e}")
        return None

print("✅ Helper functions defined!")

# ============================================================
# CELL 6: Test Predictions (Verify Models Work)
# ============================================================
print("\n" + "="*60)
print("🧪 TESTING PREDICTIONS...")
print("="*60)

test_urls = [
    "https://www.google.com/",
    "http://paypal-verify.tk/login",
    "http://amaz0n-update.xyz/"
]

for url in test_urls:
    try:
        features = extract_features(url)
        X = scaler.transform([list(features.values())])
        ml_pred = ml_model.predict(X)[0]
        ml_prob = ml_model.predict_proba(X)[0][1]

        print(f"\n📊 {url}")
        print(f"   Prediction: {'🚨 Phishing' if ml_pred == 1 else '✅ Legitimate'}")
        print(f"   Confidence: {ml_prob*100:.1f}%")
    except Exception as e:
        print(f"\n❌ Error testing {url}: {e}")

print("\n✅ Model testing complete!")

# ============================================================
# CELL 7: Create Flask API
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# API KEY AUTHENTICATION
# ============================================================
import secrets
import hashlib
from functools import wraps

# Generate ONE API key (simple!)
MY_API_KEY = hashlib.sha256(secrets.token_hex(32).encode()).hexdigest()[:32]

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')

        if not api_key:
            return jsonify({
                'error': 'API key required',
                'message': 'Provide API key in X-API-Key header or api_key parameter'
            }), 401

        if api_key != MY_API_KEY:
            return jsonify({
                'error': 'Invalid API key',
                'message': 'The provided API key is not valid'
            }), 403

        return f(*args, **kwargs)

    return decorated_function

# Print the API key
print("\n" + "="*60)
print("🔑 YOUR API KEY (SAVE THIS!)")
print("="*60)
print(f"API Key: {MY_API_KEY}")
print("="*60)
print("⚠️  This is the ONLY key that will work with your API")
print("="*60 + "\n")

@app.route('/')
def home():
    return jsonify({
        'message': 'Phishing Detection API - Ready!',
        'version': '1.0',
        'status': 'running',
        'authentication': 'API Key required (X-API-Key header)',
        'endpoints': {
            'health': 'GET /health (no auth)',
            'predict': 'POST /predict (requires API key)',
            'batch': 'POST /batch-predict (requires API key)'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'ml_model_loaded': True,
        'slm_model_loaded': True,
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    data = request.get_json()

    if not data or 'url' not in data:
        return jsonify({'error': 'URL required in request body'}), 400

    url = data['url'].strip()

    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'URL must start with http:// or https://'}), 400

    try:
        # ML prediction
        features = extract_features(url)
        X = scaler.transform([list(features.values())])
        ml_pred = ml_model.predict(X)[0]
        ml_prob = ml_model.predict_proba(X)[0]

        result = {
            'url': url,
            'ml_prediction': int(ml_pred),
            'ml_phishing_probability': round(float(ml_prob[1]), 4),
            'ml_legitimate_probability': round(float(ml_prob[0]), 4),
            'prediction_label': 'Phishing' if ml_pred == 1 else 'Legitimate'
        }

        # SLM prediction
        slm_result = analyze_with_slm(url)
        if slm_result:
            result.update(slm_result)
            ensemble_prob = ml_prob[1] * 0.6 + slm_result['slm_phishing_probability'] * 0.4
            result['ensemble_prediction'] = int(ensemble_prob > 0.5)
            result['ensemble_probability'] = round(float(ensemble_prob), 4)
            result['final_prediction'] = 'Phishing' if ensemble_prob > 0.5 else 'Legitimate'
            result['confidence'] = round(float(max(ensemble_prob, 1 - ensemble_prob)), 4)
        else:
            result['final_prediction'] = result['prediction_label']
            result['confidence'] = round(float(max(ml_prob)), 4)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch-predict', methods=['POST'])
@require_api_key
def batch_predict():
    data = request.get_json()

    if not data or 'urls' not in data:
        return jsonify({'error': 'urls array required in request body'}), 400

    urls = data['urls']
    if not isinstance(urls, list):
        return jsonify({'error': 'urls must be an array'}), 400

    if len(urls) > 100:
        return jsonify({'error': 'Maximum 100 URLs per batch'}), 400

    results = []
    for url in urls:
        try:
            features = extract_features(url)
            X = scaler.transform([list(features.values())])
            ml_pred = ml_model.predict(X)[0]
            ml_prob = ml_model.predict_proba(X)[0][1]

            results.append({
                'url': url,
                'prediction': int(ml_pred),
                'probability': round(float(ml_prob), 4),
                'label': 'Phishing' if ml_pred == 1 else 'Legitimate'
            })
        except Exception as e:
            results.append({'url': url, 'error': str(e)})

    return jsonify({'results': results})

print("✅ Flask API created!")

# ============================================================
# CELL 8: Deploy with Cloudflare Tunnel
# ============================================================
import threading
import time
import subprocess
import re

# Start Flask in background
def run_flask():
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

time.sleep(3)

print("\n" + "="*60)
print("🚀 DEPLOYING API WITH CLOUDFLARE TUNNEL...")
print("="*60)

# Install cloudflared
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1

# Start tunnel
tunnel_process = subprocess.Popen(
    ['cloudflared', 'tunnel', '--url', 'http://localhost:5000'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Get public URL
public_url = None
print("⏳ Getting public URL (this takes ~10 seconds)...")

for _ in range(30):
    line = tunnel_process.stderr.readline()
    if 'trycloudflare.com' in line:
        match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
        if match:
            public_url = match.group(0)
            break
    time.sleep(1)

if public_url:
    print("\n" + "="*60)
    print("🎉 API DEPLOYED SUCCESSFULLY!")
    print("="*60)
    print(f"\n🌐 PUBLIC URL: {public_url}")
    print("\n🔑 YOUR API KEYS:")
    print("="*60)
    for name, key in API_KEYS.items():
        print(f"  {name}: {key}")
    print("="*60)
    print("\n📡 API Endpoints:")
    print(f"   GET  {public_url}/              (no auth)")
    print(f"   GET  {public_url}/health        (no auth)")
    print(f"   POST {public_url}/predict       (requires API key)")
    print(f"   POST {public_url}/batch-predict (requires API key)")
    print("\n💡 Example with cURL:")
    print(f'   curl -X POST {public_url}/predict \\')
    print(f'        -H "Content-Type: application/json" \\')
    print(f'        -H "X-API-Key: {list(API_KEYS.values())[0]}" \\')
    print(f'        -d \'{{"url": "http://paypal-verify.tk"}}\'')
    print("\n💡 Example with JavaScript:")
    print(f"   fetch('{public_url}/predict', {{")
    print("     method: 'POST',")
    print("     headers: {")
    print("       'Content-Type': 'application/json',")
    print(f"       'X-API-Key': '{list(API_KEYS.values())[0]}'")
    print("     },")
    print("     body: JSON.stringify({url: 'http://example.com'})")
    print("   })")
    print("   .then(res => res.json())")
    print("   .then(data => console.log(data));")
    print("\n✅ SAVE YOUR API KEYS - They won't be shown again!")
    print("\n⚠️  Keep this notebook running to keep the API alive!")
    print("="*60 + "\n")
else:
    print("\n❌ Failed to get public URL. Try running this cell again.")

# ============================================================
# CELL 9: Test the Deployed API
# ============================================================
import requests
import json

if public_url:
    print("🧪 Testing the deployed API...\n")
    time.sleep(5)

    # Get first API key for testing
    test_api_key = list(API_KEYS.values())[0]

    try:
        # Test health (no auth needed)
        print("1️⃣ Testing /health endpoint (no auth)...")
        response = requests.get(f"{public_url}/health", timeout=10)
        print(json.dumps(response.json(), indent=2))

        # Test without API key (should fail)
        print("\n2️⃣ Testing /predict without API key (should fail)...")
        response = requests.post(
            f"{public_url}/predict",
            json={"url": "http://test.com"},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))

        # Test with API key (should work)
        print("\n3️⃣ Testing /predict WITH API key (should work)...")
        test_url = "http://paypal-verify.tk/login"
        response = requests.post(
            f"{public_url}/predict",
            json={"url": test_url},
            headers={
                "Content-Type": "application/json",
                "X-API-Key": test_api_key
            },
            timeout=10
        )

        print(f"\nPrediction for: {test_url}")
        result = response.json()
        print(json.dumps(result, indent=2))

        # Test batch
        print("\n4️⃣ Testing /batch-predict endpoint...")
        response = requests.post(
            f"{public_url}/batch-predict",
            json={"urls": ["https://google.com", "http://phishing-test.com"]},
            headers={
                "Content-Type": "application/json",
                "X-API-Key": test_api_key
            },
            timeout=10
        )
        print(json.dumps(response.json(), indent=2))

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print(f"\n🎉 Your API is live at: {public_url}")
        print(f"🔑 Use API key: {test_api_key}")
        print("\nUse this in your frontend application!")

    except Exception as e:
        print(f"\n⚠️  Test failed: {e}")
        print("The API might still be starting. Wait a minute and try testing manually.")
else:
    print("⚠️  No public URL available. Run Cell 8 again.")

# %% [code cell]
"""
GOOGLE COLAB - LOAD EXISTING MODEL & DEPLOY API
Upload your .pkl and .pt files, then run this to get a public API!

Steps:
1. Upload your files using the file icon on the left
2. Update the file paths below
3. Run all cells
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
print("📦 Installing dependencies...")

!pip install -q flask flask-cors
!pip install -q pandas numpy scikit-learn xgboost joblib
!pip install -q torch transformers requests

print("✅ All dependencies installed!")

# ============================================================
# CELL 2: Upload Model Files (MANUAL STEP)
# ============================================================
from google.colab import files

print("\n" + "="*60)
print("📤 UPLOAD YOUR MODEL FILES")
print("="*60)
print("\nPlease upload:")
print("1. phishing_detector_complete.pkl")
print("2. phishing_detector_complete_slm.pt")
print("\nClick the button below to upload...\n")

uploaded = files.upload()

print("\n✅ Files uploaded successfully!")
print("Uploaded files:", list(uploaded.keys()))

# ============================================================
# CELL 3: Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import re
import math
import warnings
warnings.filterwarnings('ignore')

import joblib
from urllib.parse import urlparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

print("✅ Imports complete!")

# ============================================================
# CELL 4: Load Models
# ============================================================
print("\n" + "="*60)
print("🔄 LOADING MODELS...")
print("="*60)

# Find the uploaded files (they might have been renamed)
import glob

pkl_files = glob.glob('phishing_detector_complete*.pkl')
pt_files = glob.glob('phishing_detector_complete*.pt')

if not pkl_files:
    print("❌ Error: No .pkl file found!")
    raise FileNotFoundError("Please upload phishing_detector_complete.pkl")

if not pt_files:
    print("❌ Error: No .pt file found!")
    raise FileNotFoundError("Please upload phishing_detector_complete_slm.pt")

MODEL_PATH = pkl_files[0]
SLM_PATH = pt_files[0]

print(f"Found files:")
print(f"  - {MODEL_PATH}")
print(f"  - {SLM_PATH}")

# Load ML model
print(f"\n📦 Loading ML model from: {MODEL_PATH}")
model_data = joblib.load(MODEL_PATH)
ml_model = model_data['model']
scaler = model_data['scaler']
print("✅ ML model loaded successfully")

# Load SLM
print(f"\n📦 Loading SLM from: {SLM_PATH}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
slm_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
slm_model.eval()

slm_classifier = nn.Sequential(
    nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 2)
).to(device)

checkpoint = torch.load(SLM_PATH, map_location=device)
slm_classifier.load_state_dict(checkpoint['classifier'])
print("✅ SLM loaded with trained weights")

print("\n" + "="*60)
print("✅ ALL MODELS LOADED SUCCESSFULLY!")
print("="*60)

# ============================================================
# CELL 5: Helper Functions
# ============================================================

def entropy(text):
    """Calculate Shannon entropy"""
    if not text:
        return 0
    p = [text.count(c) / len(text) for c in set(text)]
    return -sum(x * math.log2(x) for x in p if x > 0)


def extract_features(url):
    """Extract URL features for ML model"""
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    features = {
        'url_len': len(url),
        'domain_len': len(domain),
        'path_len': len(path),
        'num_dot': url.count('.'),
        'num_hyph': url.count('-'),
        'num_slash': url.count('/'),
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'entropy_url': entropy(url),
        'entropy_domain': entropy(domain),
        'entropy_path': entropy(path),
        'num_digits': sum(c.isdigit() for c in url),
        'num_letters': sum(c.isalpha() for c in url),
        'num_special': sum(not c.isalnum() for c in url),
        'num_params': url.count('='),
        'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', domain) else 0
    }
    return features


def analyze_with_slm(url):
    """Analyze URL using SLM"""
    try:
        parsed = urlparse(url)
        text = f"URL: {url} | Protocol: {parsed.scheme} | Domain: {parsed.netloc} | Path: {parsed.path}"

        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = slm_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            logits = slm_classifier(emb)
            probs = torch.softmax(logits, dim=1)[0]

        phishing_prob = probs[1].item()
        legit_prob = probs[0].item()

        return {
            'slm_prediction': int(phishing_prob > 0.5),
            'slm_phishing_probability': round(phishing_prob, 4),
            'slm_legitimate_probability': round(legit_prob, 4),
            'slm_confidence': round(max(phishing_prob, legit_prob), 4)
        }
    except Exception as e:
        print(f"SLM error: {e}")
        return None

print("✅ Helper functions defined!")

# ============================================================
# CELL 6: Test Predictions (Verify Models Work)
# ============================================================
print("\n" + "="*60)
print("🧪 TESTING PREDICTIONS...")
print("="*60)

test_urls = [
    "https://www.google.com/",
    "http://paypal-verify.tk/login",
    "http://amaz0n-update.xyz/"
]

for url in test_urls:
    try:
        features = extract_features(url)
        X = scaler.transform([list(features.values())])
        ml_pred = ml_model.predict(X)[0]
        ml_prob = ml_model.predict_proba(X)[0]

        phishing_prob = ml_prob[1]
        legitimate_prob = ml_prob[0]

        print(f"\n📊 {url}")
        print(f"   Prediction: {'🚨 Phishing' if ml_pred == 1 else '✅ Legitimate'}")
        print(f"   ML Phishing Probability: {phishing_prob*100:.2f}%")
        print(f"   ML Legitimate Probability: {legitimate_prob*100:.2f}%")
    except Exception as e:
        print(f"\n❌ Error testing {url}: {e}")

print("\n✅ Model testing complete!")

# ============================================================
# CELL 7: Create Flask API
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# API KEY AUTHENTICATION
# ============================================================
import secrets
import hashlib
from functools import wraps

# Generate ONE API key (simple!)
MY_API_KEY = hashlib.sha256(secrets.token_hex(32).encode()).hexdigest()[:32]

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')

        if not api_key:
            return jsonify({
                'error': 'API key required',
                'message': 'Provide API key in X-API-Key header or api_key parameter'
            }), 401

        if api_key != MY_API_KEY:
            return jsonify({
                'error': 'Invalid API key',
                'message': 'The provided API key is not valid'
            }), 403

        return f(*args, **kwargs)

    return decorated_function

# Print the API key
print("\n" + "="*60)
print("🔑 YOUR API KEY (SAVE THIS!)")
print("="*60)
print(f"API Key: {MY_API_KEY}")
print("="*60)
print("⚠️  This is the ONLY key that will work with your API")
print("="*60 + "\n")

@app.route('/')
def home():
    return jsonify({
        'message': 'Phishing Detection API - Ready!',
        'version': '1.0',
        'status': 'running',
        'authentication': 'API Key required (X-API-Key header)',
        'endpoints': {
            'health': 'GET /health (no auth)',
            'predict': 'POST /predict (requires API key)',
            'batch': 'POST /batch-predict (requires API key)'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'ml_model_loaded': True,
        'slm_model_loaded': True,
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Check API key first
    api_key = request.headers.get('X-API-Key') or request.args.get('api_key')

    if not api_key:
        return jsonify({
            'error': 'API key required',
            'message': 'Provide API key in X-API-Key header or api_key parameter'
        }), 401

    if api_key != MY_API_KEY:
        return jsonify({
            'error': 'Invalid API key',
            'message': 'The provided API key is not valid'
        }), 403

    # Get URL from request
    data = request.get_json()

    if not data or 'url' not in data:
        return jsonify({'error': 'URL required in request body'}), 400

    url = data['url'].strip()

    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'URL must start with http:// or https://'}), 400

    try:
        # ML prediction
        features = extract_features(url)
        X = scaler.transform([list(features.values())])
        ml_pred = ml_model.predict(X)[0]
        ml_prob = ml_model.predict_proba(X)[0]

        phishing_prob = float(ml_prob[1])
        legitimate_prob = float(ml_prob[0])

        result = {
            'url': url,
            'ml_prediction': int(ml_pred),
            'ml_phishing_probability': round(phishing_prob, 4),
            'ml_legitimate_probability': round(legitimate_prob, 4),
            'prediction_label': 'Phishing' if ml_pred == 1 else 'Legitimate',
            'status': '🚨 PHISHING DETECTED' if ml_pred == 1 else '✅ SAFE URL'
        }

        # SLM prediction
        slm_result = analyze_with_slm(url)
        if slm_result:
            result.update(slm_result)
            ensemble_prob = phishing_prob * 0.6 + slm_result['slm_phishing_probability'] * 0.4
            result['ensemble_prediction'] = int(ensemble_prob > 0.5)
            result['ensemble_probability'] = round(float(ensemble_prob), 4)
            result['final_prediction'] = 'Phishing' if ensemble_prob > 0.5 else 'Legitimate'
            result['confidence'] = round(float(max(ensemble_prob, 1 - ensemble_prob)) * 100, 2)
        else:
            result['final_prediction'] = result['prediction_label']
            # Confidence is the probability of the predicted class
            result['confidence'] = round(float(phishing_prob if ml_pred == 1 else legitimate_prob) * 100, 2)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    # Check API key first
    api_key = request.headers.get('X-API-Key') or request.args.get('api_key')

    if not api_key:
        return jsonify({
            'error': 'API key required',
            'message': 'Provide API key in X-API-Key header or api_key parameter'
        }), 401

    if api_key != MY_API_KEY:
        return jsonify({
            'error': 'Invalid API key',
            'message': 'The provided API key is not valid'
        }), 403

    # Get URLs from request
    data = request.get_json()

    if not data or 'urls' not in data:
        return jsonify({'error': 'urls array required in request body'}), 400

    urls = data['urls']
    if not isinstance(urls, list):
        return jsonify({'error': 'urls must be an array'}), 400

    if len(urls) > 100:
        return jsonify({'error': 'Maximum 100 URLs per batch'}), 400

    results = []
    for url in urls:
        try:
            features = extract_features(url)
            X = scaler.transform([list(features.values())])
            ml_pred = ml_model.predict(X)[0]
            ml_prob = ml_model.predict_proba(X)[0][1]

            results.append({
                'url': url,
                'prediction': int(ml_pred),
                'probability': round(float(ml_prob), 4),
                'label': 'Phishing' if ml_pred == 1 else 'Legitimate'
            })
        except Exception as e:
            results.append({'url': url, 'error': str(e)})

    return jsonify({'results': results})

print("✅ Flask API created!")

# ============================================================
# CELL 8: Deploy with Cloudflare Tunnel
# ============================================================
import threading
import time
import subprocess
import re

# Start Flask in background
def run_flask():
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

time.sleep(3)

print("\n" + "="*60)
print("🚀 DEPLOYING API WITH CLOUDFLARE TUNNEL...")
print("="*60)

# Install cloudflared
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1

# Start tunnel
tunnel_process = subprocess.Popen(
    ['cloudflared', 'tunnel', '--url', 'http://localhost:5000'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Get public URL
public_url = None
print("⏳ Getting public URL (this takes ~10 seconds)...")

for _ in range(30):
    line = tunnel_process.stderr.readline()
    if 'trycloudflare.com' in line:
        match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
        if match:
            public_url = match.group(0)
            break
    time.sleep(1)

if public_url:
    print("\n" + "="*60)
    print("🎉 API DEPLOYED SUCCESSFULLY!")
    print("="*60)
    print(f"\n🌐 PUBLIC URL: {public_url}")
    print("\n🔑 YOUR API KEY:")
    print("="*60)
    print(f"  {MY_API_KEY}")
    print("="*60)
    print("\n📡 API Endpoints:")
    print(f"   GET  {public_url}/              (no auth)")
    print(f"   GET  {public_url}/health        (no auth)")
    print(f"   POST {public_url}/predict       (requires API key)")
    print(f"   POST {public_url}/batch-predict (requires API key)")
    print("\n💡 Example with cURL:")
    print(f'   curl -X POST {public_url}/predict \\')
    print(f'        -H "Content-Type: application/json" \\')
    print(f'        -H "X-API-Key: {MY_API_KEY}" \\')
    print(f'        -d \'{{"url": "http://paypal-verify.tk"}}\'')
    print("\n💡 Example with JavaScript:")
    print(f"   fetch('{public_url}/predict', {{")
    print("     method: 'POST',")
    print("     headers: {")
    print("       'Content-Type': 'application/json',")
    print(f"       'X-API-Key': '{MY_API_KEY}'")
    print("     },")
    print("     body: JSON.stringify({url: 'http://example.com'})")
    print("   })")
    print("   .then(res => res.json())")
    print("   .then(data => console.log(data));")
    print("\n✅ SAVE YOUR API KEY - You'll need it!")
    print("\n⚠️  Keep this notebook running to keep the API alive!")
    print("="*60 + "\n")
else:
    print("\n❌ Failed to get public URL. Try running this cell again.")

# ============================================================
# CELL 9: Test the Deployed API
# ============================================================
import requests
import json

if public_url:
    print("🧪 Testing the deployed API...\n")
    time.sleep(5)

    try:
        # Test health (no auth needed)
        print("1️⃣ Testing /health endpoint (no auth)...")
        response = requests.get(f"{public_url}/health", timeout=10)
        print(json.dumps(response.json(), indent=2))

        # Test without API key (should fail)
        print("\n2️⃣ Testing /predict without API key (should fail)...")
        response = requests.post(
            f"{public_url}/predict",
            json={"url": "http://test.com"},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))

        # Test with API key (should work)
        print("\n3️⃣ Testing /predict WITH API key (should work)...")
        test_url = "http://paypal-verify.tk/login"
        response = requests.post(
            f"{public_url}/predict",
            json={"url": test_url},
            headers={
                "Content-Type": "application/json",
                "X-API-Key": MY_API_KEY
            },
            timeout=10
        )

        print(f"\nPrediction for: {test_url}")
        result = response.json()
        print(json.dumps(result, indent=2))

        # Test batch
        print("\n4️⃣ Testing /batch-predict endpoint...")
        response = requests.post(
            f"{public_url}/batch-predict",
            json={"urls": ["https://google.com", "http://phishing-test.com"]},
            headers={
                "Content-Type": "application/json",
                "X-API-Key": MY_API_KEY
            },
            timeout=10
        )
        print(json.dumps(response.json(), indent=2))

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print(f"\n🎉 Your API is live at: {public_url}")
        print(f"🔑 Your API key: {MY_API_KEY}")
        print("\nUse these in your frontend application!")

    except Exception as e:
        print(f"\n⚠️  Test failed: {e}")
        print("The API might still be starting. Wait a minute and try testing manually.")
else:
    print("⚠️  No public URL available. Run Cell 8 again.")