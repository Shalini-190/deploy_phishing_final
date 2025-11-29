import math
import re
from urllib.parse import urlparse
from collections import Counter

def entropy(text):
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0
    p = [text.count(c) / len(text) for c in set(text)]
    return -sum(x * math.log2(x) for x in p if x > 0)

def extract_url_features(url):
    """Extract comprehensive features from a URL."""
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    
    # Basic lengths
    features = {
        'url_len': len(url),
        'domain_len': len(domain),
        'path_len': len(path),
    }
    
    # Character counts
    features.update({
        'num_dot': url.count('.'),
        'num_hyph': url.count('-'),
        'num_slash': url.count('/'),
        'num_question': url.count('?'),
        'num_equal': url.count('='),
        'num_at': url.count('@'),
        'num_percent': url.count('%'),
        'num_ampersand': url.count('&'),
        'num_digits': sum(c.isdigit() for c in url),
        'num_letters': sum(c.isalpha() for c in url),
        'num_special': sum(not c.isalnum() for c in url),
    })
    
    # Entropy features
    features.update({
        'entropy_url': entropy(url),
        'entropy_domain': entropy(domain),
        'entropy_path': entropy(path),
    })
    
    # Domain specific features
    features.update({
        'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', domain) else 0,
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'is_shortened': 1 if domain in ['bit.ly', 'goo.gl', 'tinyurl.com', 't.co'] else 0,
        'tld_in_path': 1 if any(tld in path for tld in ['.com', '.net', '.org', '.info']) else 0,
    })
    
    # Suspicious patterns
    suspicious_keywords = ['login', 'verify', 'update', 'account', 'secure', 'banking', 'confirm']
    features['has_suspicious_keyword'] = 1 if any(kw in url.lower() for kw in suspicious_keywords) else 0
    
    return features

def extract_email_features(subject, body):
    """Extract features from email content."""
    text = (subject + " " + body).lower()
    
    features = {
        'subject_len': len(subject),
        'body_len': len(body),
        'num_urls': len(re.findall(r'https?://\S+', text)),
        'has_urgent_words': 1 if any(w in text for w in ['urgent', 'immediate', 'action required', 'suspended']) else 0,
        'has_financial_words': 1 if any(w in text for w in ['bank', 'invoice', 'payment', 'credit card']) else 0,
    }
    
    return features
