from app.model import AdvancedPhishingDetector
import os

def test_urls():
    print("\nTesting URLs...")
    detector = AdvancedPhishingDetector(use_slm=True)
    if os.path.exists("phishing_model.pkl"):
        detector.load("phishing_model")
    else:
        print("Model not found!")
        return

    test_cases = [
        "https://www.google.com",
        "http://paypal-verify-account.com/login",
        "https://secure-banking-update.net",
        "https://github.com/microsoft/vscode"
    ]

    for url in test_cases:
        res = detector.predict_url(url)
        print(f"\nURL: {url}")
        print(f"Verdict: {res.get('final_verdict', 'N/A')}")
        
        # Calculate confidence in the verdict
        prob = res.get('ensemble_probability', 0)
        confidence = prob if res.get('final_verdict') == 'Phishing' else (1 - prob)
        
        print(f"Confidence: {confidence:.4f}")
        print(f"SLM Enabled: {res.get('slm_enabled', False)}")

def test_emails():
    print("\nTesting Emails...")
    detector = AdvancedPhishingDetector(use_slm=True)
    if os.path.exists("phishing_model.pkl"):
        detector.load("phishing_model")

    emails = [
        {
            "subject": "Urgent: Account Suspended",
            "body": "Dear user, your account has been suspended due to suspicious activity. Click here to verify your identity immediately: http://verify-account.com"
        },
        {
            "subject": "Meeting Reminder",
            "body": "Hi team, just a reminder about the meeting tomorrow at 10 AM. See you there."
        }
    ]

    for email in emails:
        res = detector.predict_email(email['subject'], email['body'])
        print(f"\nSubject: {email['subject']}")
        print(f"Verdict: {res.get('final_verdict', 'N/A')}")
        print(f"Score: {res.get('ensemble_probability', 0):.4f}")

if __name__ == "__main__":
    test_urls()
    test_emails()
