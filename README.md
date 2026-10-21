🛡️ SpamGuard AI — Production Email Spam & Threat Detection System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, full-stack AI email spam, phishing, and threat detection web application designed for high accuracy and instant production server deployment.

---

## 🌟 Key Highlights & Features

1. **Hybrid AI Detection Core**:
   - **Soft-Voting Ensemble Model**: Combines calibrated L2 Logistic Regression and Multinomial Naive Bayes over unigram + bigram sublinear TF-IDF features.
   - **High Accuracy & Zero Missed Threats**:
     - **Accuracy**: 95.0%
     - **Precision**: 91.7%
     - **Recall**: 100.0% (Zero missed phishing/spam attacks)
     - **ROC-AUC**: 0.980
2. **Deep Heuristic Phishing & Scam Engine**:
   - URL & IP-in-URL detection (flags direct numeric IP links like `http://192.168.1.1/...`).
   - URL shortener detection (`bit.ly`, `tinyurl.com`, etc.).
   - Suspicious top-level domain analyzer (`.xyz`, `.pw`, `.cc`, `.top`, etc.).
   - Psychological urgency & coercion analyzer (artificial time pressure, account suspension threats).
   - Credential harvesting pattern detector (fake password resets, PIN verification).
   - Financial & crypto scam indicators (wire transfers, fake lotteries, wallet addresses).
3. **Explainable AI (XAI) Word Inspector**:
   - Tokenizes incoming text and maps statistical feature weights.
   - Visual color-coded text highlighter (red for spam triggers, green for legitimate indicators, hover weights).
4. **Full-Featured Frontend**:
   - Cyber-security themed dark interface with Tailwind CSS, Lucide icons, and Chart.js.
   - **Live Scanner**: Type or paste Subject + Body or toggle Raw RFC 822 format. Quick one-click realistic presets.
   - **Batch Scanner**: Upload CSV / TXT files for multi-email scanning and export results to CSV.
   - **AI Metrics Dashboard**: Live confusion matrix, precision/recall cards, and top predictive spam keywords chart.
   - **Developer API Playground**: Ready-to-copy cURL, Python, and JavaScript snippets.
   - **Active Learning Feedback Modal**: Allows users to flag false positives/negatives to retrain the model.
5. **Production Ready & Deployable**:
   - Pre-configured `Dockerfile`, `docker-compose.yml`, `Procfile`, `render.yaml`, and `requirements.txt`.
   - SQLite database for persistent scan history and active learning feedback.

---

## 🚀 Quick Start (Local Run)

### 1. Clone & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/your-username/spam-detection.git
cd spam-detection

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Train / Verify Model (Optional, pre-trained artifacts included)
```bash
python train_model.py
```

### 3. Launch the Server
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://localhost:8000`**

Interactive Swagger API docs available at:
👉 **`http://localhost:8000/docs`**

---

## 🧪 Running Automated Tests

Run the built-in test suite to verify all endpoints and inference pipelines:
```bash
python test_app.py
```
Expected output:
```text
Running test suite...
ALL TESTS PASSED SUCCESSFULLY! (8/8)
```

---

## 🌐 Server Hosting & Deployment Guide

This project is packaged to run on any cloud platform or private server:

### Option 1: Free 1-Click Cloud Hosting (Render.com)
1. Push your project to a GitHub repository.
2. Sign in to [Render.com](https://render.com) and click **New Web Service**.
3. Connect your GitHub repository.
4. Select **Python 3** environment:
   - **Build Command**: `pip install -r requirements.txt && python train_model.py`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT --workers 2`
5. Click **Create Web Service**. Your website will be live with a free HTTPS domain (e.g. `https://spamguard-ai.onrender.com`).

---

### Option 2: Docker & Docker Compose
To deploy anywhere using Docker:
```bash
# Build and run container
docker compose up -d

# Check logs
docker compose logs -f

# Stop container
docker compose down
```
Access at `http://YOUR_SERVER_IP:8000`.

---

### Option 3: Linux VPS (Ubuntu / Debian / DigitalOcean / AWS EC2)

#### 1. Setup the code on your server:
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git nginx
git clone https://github.com/your-username/spam-detection.git /var/www/spamguard
cd /var/www/spamguard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
```

#### 2. Create a Systemd Service daemon:
Create `/etc/systemd/system/spamguard.service`:
```ini
[Unit]
Description=SpamGuard AI FastAPI Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/var/www/spamguard
ExecStart=/var/www/spamguard/venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 --workers 3
Restart=always

[Install]
WantedBy=multi-user.target
```
Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable spamguard
sudo systemctl start spamguard
```

#### 3. Configure Nginx Reverse Proxy with SSL:
Create `/etc/nginx/sites-available/spamguard`:
```nginx
server {
    server_name yourdomain.com www.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
Enable site and get free SSL certificate via Certbot:
```bash
sudo ln -s /etc/nginx/sites-available/spamguard /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

---

## 📡 REST API Documentation

### `POST /api/predict`
Classifies a single email and returns probability, verdict, heuristic flags, and explainability tokens.

**Request Body:**
```json
{
  "subject": "URGENT: Verify Your Account",
  "body": "Your PayPal account has been suspended. Click http://192.168.1.1/login to restore access."
}
```

**Response:**
```json
{
  "status": "success",
  "verdict": "SPAM",
  "verdict_badge": "dangerous",
  "risk_level": "HIGH RISK",
  "spam_probability": 100.0,
  "ham_probability": 0.0,
  "raw_ml_spam_prob": 86.2,
  "summary": "This email displays severe spam/phishing characteristics. Exercise extreme caution...",
  "heuristics": {
    "threat_flags": [
      {
        "type": "URL_IP_ADDRESS",
        "severity": "CRITICAL",
        "description": "Raw IP address found in URL (http://192.168.1.1)."
      },
      {
        "type": "PSYCHOLOGICAL_URGENCY",
        "severity": "HIGH",
        "description": "High artificial urgency / intimidation tactics: 'urgent, suspended'."
      }
    ],
    "urgency_score": 100,
    "financial_score": 0,
    "phishing_score": 95
  },
  "top_spam_triggers": [
    { "word": "account", "weight": 0.516 },
    { "word": "verify", "weight": 0.483 }
  ],
  "scan_id": 4
}
```

---

## 📂 Project Structure

```text
spam-detection/
├── app.py                   # FastAPI server & route handlers
├── detector_engine.py       # Core hybrid ML + heuristic detection engine
├── database.py              # SQLite storage for history and feedback
├── train_model.py           # Model training & benchmark dataset generator
├── test_app.py              # Automated test suite
├── requirements.txt         # Production Python dependencies
├── Dockerfile               # Production container image configuration
├── docker-compose.yml       # 1-command Docker service orchestration
├── Procfile                 # Cloud web dyno runner (Heroku / Render)
├── render.yaml              # Render infrastructure-as-code specification
├── .gitignore               # Git ignored patterns
├── model/                   # Serialized ML artifacts & metrics
│   ├── spam_classifier.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── dataset.csv
│   └── metrics.json
├── templates/
│   └── index.html           # Modern full-featured responsive frontend UI
├── static/                  # Static assets directory
└── data/
    └── sample_emails.csv    # Sample test batch CSV dataset
```

---

## 📜 License
MIT License. Free for personal, commercial, and educational use.
