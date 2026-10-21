"""
Spam & Phishing Detector Engine
Combines Machine Learning (TF-IDF + Soft-Voting Classifier) with
Heuristic Rules, Deep Link/URL Analysis, Urgency Scoring, and Explainable AI (XAI).
"""

import os
import re
import json
import joblib
import numpy as np
from typing import Dict, List, Any, Optional

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

class SpamDetectorEngine:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.metrics = {}
        self.word_weights = {}
        self.load_artifacts()
        
        # Heuristic Pattern Definitions
        self.urgency_patterns = [
            r"\b(urgent|immediately|immediate action|act now|hurry|final notice|within \d+ (hours?|mins?|days?))\b",
            r"\b(suspended|terminated|deactivated|locked|restricted|blocked|arrest|subpoena|legal action)\b",
            r"\b(critical security|unauthorized|breach|compromised|expire|expires today)\b"
        ]
        
        self.financial_patterns = [
            r"\b(wire transfer|bank details|routing number|credit card|cvv|ssn|social security)\b",
            r"\b(bitcoin|btc|eth|ethereum|crypto|wallet address|crypto giveaway|blockchain)\b",
            r"\b(\$?\d{1,3}(,\d{3})+(\.\d{2})?|\$\d{4,}|\b\d+\s?million (dollars|usd))\b",
            r"\b(inheritance|beneficiary|unclaimed funds|lottery|mega millions|prize winner|claim now)\b"
        ]
        
        self.credential_patterns = [
            r"\b(verify your (identity|account|password|pin|details))\b",
            r"\b(confirm your (credentials|identity|password|pin))\b",
            r"\b(update your (billing|payment|security info|password))\b",
            r"\b(reset password|click here to login|sign in to review)\b"
        ]
        
        self.suspicious_tlds = [
            r"\.(xyz|top|cc|pw|info|club|work|buzz|icu|live|click|tk|ml|ga|cf|gq)\b"
        ]

    def load_artifacts(self):
        """Loads trained model, vectorizer, and precomputed metrics."""
        model_path = os.path.join(self.model_dir, "spam_classifier.joblib")
        vec_path = os.path.join(self.model_dir, "tfidf_vectorizer.joblib")
        metrics_path = os.path.join(self.model_dir, "metrics.json")
        
        if os.path.exists(model_path) and os.path.exists(vec_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
            
            # Extract LR coefficients for token-level explanations
            try:
                lr_clf = self.model.named_estimators_["lr"]
                feature_names = self.vectorizer.get_feature_names_out()
                self.word_weights = dict(zip(feature_names, lr_clf.coef_[0]))
            except Exception as e:
                print(f"Warning: Could not extract word weights: {e}")
        else:
            print("Warning: Model artifacts not found. Please run train_model.py first.")
            
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)

    def extract_urls(self, text: str) -> List[str]:
        """Extracts all URLs (domains and IPs) from text."""
        url_regex = r"https?://(?:\d{1,3}(?:\.\d{1,3}){3}|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})(?::\d+)?(?:/[^\s]*)?"
        return re.findall(url_regex, text, re.IGNORECASE)

    def analyze_heuristics(self, text: str) -> Dict[str, Any]:
        """Calculates rule-based heuristic risk indicators."""
        text_lower = text.lower()
        threat_flags = []
        
        # 1. URL Analysis
        urls = self.extract_urls(text)
        ip_urls = re.findall(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", text)
        shorteners = re.findall(r"https?://(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|cutt\.ly|ow\.ly)/[a-zA-Z0-9]+", text, re.I)
        suspicious_tld_found = any(re.search(pat, text_lower) for pat in self.suspicious_tlds)
        
        if ip_urls:
            threat_flags.append({
                "type": "URL_IP_ADDRESS",
                "severity": "CRITICAL",
                "description": f"Raw IP address found in URL ({ip_urls[0]}). Common indicator of phishing host."
            })
        if shorteners:
            threat_flags.append({
                "type": "URL_SHORTENER",
                "severity": "MEDIUM",
                "description": f"URL shortener detected ({shorteners[0]}), masking the final landing destination."
            })
        if suspicious_tld_found:
            threat_flags.append({
                "type": "SUSPICIOUS_TLD",
                "severity": "MEDIUM",
                "description": "Links or mentions of low-reputation top-level domains (.xyz, .pw, .cc, etc.)."
            })
            
        # 2. Urgency and Coercion Analysis
        urgency_matches = []
        for pat in self.urgency_patterns:
            matches = re.findall(pat, text_lower)
            for m in matches:
                urgency_matches.append(m[0] if isinstance(m, tuple) else m)
        if urgency_matches:
            threat_flags.append({
                "type": "PSYCHOLOGICAL_URGENCY",
                "severity": "HIGH",
                "description": f"High artificial urgency / intimidation tactics: '{', '.join(list(set(urgency_matches))[:3])}'."
            })
            
        # 3. Financial and Lottery Scams
        financial_matches = []
        for pat in self.financial_patterns:
            matches = re.findall(pat, text_lower)
            for m in matches:
                financial_matches.append(m[0] if isinstance(m, tuple) else m)
        if financial_matches:
            threat_flags.append({
                "type": "FINANCIAL_SOLICITATION",
                "severity": "HIGH",
                "description": f"High-risk financial/crypto solicitation patterns: '{', '.join(list(set(financial_matches))[:3])}'."
            })
            
        # 4. Credential Harvesting / Account Takeover
        cred_matches = []
        for pat in self.credential_patterns:
            matches = re.findall(pat, text_lower)
            for m in matches:
                cred_matches.append(m[0] if isinstance(m, tuple) else m)
        if cred_matches:
            threat_flags.append({
                "type": "CREDENTIAL_HARVESTING",
                "severity": "CRITICAL",
                "description": f"Credential verification request detected: '{', '.join(list(set(cred_matches))[:3])}'."
            })
            
        # 5. Visual cues: ALL CAPS ratio and excessive punctuation
        alpha_chars = [c for c in text if c.isalpha()]
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
        exclamation_count = text.count("!")
        
        if caps_ratio > 0.45 and len(alpha_chars) > 20:
            threat_flags.append({
                "type": "EXCESSIVE_CAPITALIZATION",
                "severity": "LOW",
                "description": f"Abnormally high uppercase text ratio ({caps_ratio * 100:.1f}%), typical of promotional/scam spam."
            })
        if exclamation_count >= 3:
            threat_flags.append({
                "type": "EXCESSIVE_PUNCTUATION",
                "severity": "LOW",
                "description": f"Excessive exclamation marks ({exclamation_count}) found."
            })
            
        # Compute heuristic sub-scores
        urgency_score = min(100, len(urgency_matches) * 35)
        financial_score = min(100, len(financial_matches) * 35)
        phishing_score = min(100, (len(cred_matches) * 45) + (len(ip_urls) * 50) + (len(shorteners) * 20))
        
        return {
            "threat_flags": threat_flags,
            "url_count": len(urls),
            "urls": urls[:5],
            "urgency_score": urgency_score,
            "financial_score": financial_score,
            "phishing_score": phishing_score,
            "caps_ratio": round(caps_ratio * 100, 1),
            "exclamation_count": exclamation_count
        }

    def explain_tokens(self, text: str) -> List[Dict[str, Any]]:
        """
        Tokenizes text and scores each word according to model coefficients
        for Explainable AI (XAI) interactive rendering.
        """
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
        explained = []
        
        for token in tokens:
            cleaned = token.lower()
            weight = self.word_weights.get(cleaned, 0.0)
            
            # Classify token contribution
            if weight > 0.35:
                tag = "spam_trigger"
            elif weight < -0.35:
                tag = "ham_trigger"
            else:
                tag = "neutral"
                
            explained.append({
                "token": token,
                "weight": round(float(weight), 3),
                "tag": tag
            })
            
        return explained

    def predict(self, subject: str = "", body: str = "", raw_email: str = "") -> Dict[str, Any]:
        """
        Full prediction pipeline combining subject, body, or raw text.
        """
        # Synthesize combined content
        full_text = ""
        if raw_email.strip():
            full_text = raw_email.strip()
        else:
            parts = []
            if subject.strip():
                parts.append(f"Subject: {subject.strip()}")
            if body.strip():
                parts.append(body.strip())
            full_text = "\n\n".join(parts)
            
        if not full_text.strip():
            return {
                "status": "error",
                "message": "Input email content cannot be empty."
            }
            
        # Heuristics analysis
        heuristics = self.analyze_heuristics(full_text)
        
        # ML Inference
        if self.model and self.vectorizer:
            vec = self.vectorizer.transform([full_text])
            probabilities = self.model.predict_proba(vec)[0]
            ham_prob = float(probabilities[0])
            spam_prob = float(probabilities[1])
        else:
            # Fallback if model not loaded
            spam_prob = 0.5
            ham_prob = 0.5
            
        # Combined Hybrid Score:
        # Boost spam probability if critical heuristic flags are detected (e.g. IP URLs or Credential Harvesting)
        heuristic_boost = 0.0
        for flag in heuristics["threat_flags"]:
            if flag["severity"] == "CRITICAL":
                heuristic_boost += 0.20
            elif flag["severity"] == "HIGH":
                heuristic_boost += 0.10
                
        final_spam_score = min(1.0, max(0.0, spam_prob + heuristic_boost))
        spam_percentage = round(final_spam_score * 100, 1)
        
        # Categorize verdict
        if spam_percentage >= 70.0:
            verdict = "SPAM"
            risk_level = "HIGH RISK"
            verdict_badge = "dangerous"
            summary = "This email displays severe spam/phishing characteristics. Exercise extreme caution; do not click any links or download attachments."
        elif spam_percentage >= 35.0:
            verdict = "SUSPICIOUS"
            risk_level = "MODERATE RISK"
            verdict_badge = "warning"
            summary = "This email contains suspicious patterns or promotional keywords. Verify the sender before taking any action."
        else:
            verdict = "HAM (LEGITIMATE)"
            risk_level = "LOW RISK"
            verdict_badge = "safe"
            summary = "This email appears legitimate with standard communication patterns and no significant threat indicators."
            
        # Generate token-level explanation
        token_breakdown = self.explain_tokens(full_text)
        
        # Extract top 5 contributing spam keywords
        spam_keywords = sorted(
            [t for t in token_breakdown if t["tag"] == "spam_trigger"],
            key=lambda x: x["weight"],
            reverse=True
        )
        unique_spam_keywords = []
        seen = set()
        for k in spam_keywords:
            w = k["token"].lower()
            if w not in seen:
                seen.add(w)
                unique_spam_keywords.append({"word": k["token"], "weight": k["weight"]})
                if len(unique_spam_keywords) >= 8:
                    break

        return {
            "status": "success",
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "risk_level": risk_level,
            "spam_probability": spam_percentage,
            "ham_probability": round(100.0 - spam_percentage, 1),
            "raw_ml_spam_prob": round(spam_prob * 100, 1),
            "summary": summary,
            "heuristics": heuristics,
            "top_spam_triggers": unique_spam_keywords,
            "token_explanation": token_breakdown,
            "input_preview": {
                "char_count": len(full_text),
                "word_count": len(full_text.split()),
                "subject": subject if subject else (full_text[:60] + "...")
            }
        }

# Global singleton detector instance
detector = SpamDetectorEngine()

if __name__ == "__main__":
    test_email = "URGENT: Your PayPal account has been suspended! Click http://192.168.1.1/login to verify your password immediately or lose $5,000."
    res = detector.predict(body=test_email)
    print("Test Prediction:")
    print(json.dumps(res, indent=2))
