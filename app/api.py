from flask import Flask, request, jsonify
from flask_cors import CORS
from .model import AdvancedPhishingDetector
import os

app = Flask(__name__)
CORS(app)

# Initialize detector
# We try to load a trained model, otherwise we initialize a fresh one (which will need training)
detector = AdvancedPhishingDetector(use_slm=True)
MODEL_PATH = "phishing_model"

if os.path.exists(f"{MODEL_PATH}.pkl"):
    detector.load(MODEL_PATH)
else:
    print("⚠ No trained model found. Please run train.py first.")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'slm_enabled': detector.slm.enabled if detector.slm else False,
        'model_loaded': detector.model is not None
    })

@app.route('/predict', methods=['POST'])
def predict_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL required'}), 400
    
    url = data['url']
    try:
        result = detector.predict_url(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-email', methods=['POST'])
def predict_email():
    data = request.get_json()
    if not data or 'subject' not in data or 'body' not in data:
        return jsonify({'error': 'Email subject and body required'}), 400
    
    try:
        result = detector.predict_email(data['subject'], data['body'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
