"""
Unit and Integration Test Suite for SpamGuard AI
"""

from fastapi.testclient import TestClient
from app import app
import json

client = TestClient(app)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "SpamGuard AI" in response.text
    assert "Live Scanner" in response.text

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "confusion_matrix" in data
    assert "top_spam_words" in data
    assert data["accuracy"] > 80.0

def test_predict_phishing_spam():
    payload = {
        "subject": "URGENT: Your PayPal account has been suspended",
        "body": "Unauthorized sign-in detected. Click http://192.168.1.1/paypal to verify your password immediately or lose $5,000."
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["verdict"] == "SPAM"
    assert data["spam_probability"] > 80.0
    assert data["risk_level"] == "HIGH RISK"
    assert len(data["heuristics"]["threat_flags"]) > 0
    assert "scan_id" in data

def test_predict_legitimate_ham():
    payload = {
        "subject": "Weekly Sprint Standup Sync",
        "body": "Hi team, please find attached the agenda for our Thursday 10:00 AM sprint planning meeting in Conference Room B."
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "HAM" in data["verdict"]
    assert data["spam_probability"] < 40.0
    assert data["verdict_badge"] == "safe"

def test_batch_scan():
    payload = {
        "items": [
            {"subject": "Urgent Security Notice", "body": "Verify account immediately or account is locked."},
            {"subject": "Team Lunch", "body": "Hey everyone, grab lunch at the cafe downstairs today at 12:30 PM?"}
        ]
    }
    response = client.post("/api/batch-scan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["spam_count"] >= 1
    assert data["ham_count"] >= 1

def test_feedback_submission():
    payload = {
        "scan_id": 1,
        "email_text": "Sample text for feedback",
        "predicted_verdict": "SPAM",
        "user_corrected_label": "ham",
        "notes": "Testing feedback endpoint"
    }
    response = client.post("/api/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "feedback_id" in data

def test_stats_and_history():
    stats_res = client.get("/api/stats")
    assert stats_res.status_code == 200
    assert stats_res.json()["total_scans"] > 0

    hist_res = client.get("/api/history")
    assert hist_res.status_code == 200
    assert len(hist_res.json()["scans"]) > 0

if __name__ == "__main__":
    print("Running test suite...")
    test_home_page()
    test_health_check()
    test_metrics_endpoint()
    test_predict_phishing_spam()
    test_predict_legitimate_ham()
    test_batch_scan()
    test_feedback_submission()
    test_stats_and_history()
    print("ALL TESTS PASSED SUCCESSFULLY! (8/8)")
