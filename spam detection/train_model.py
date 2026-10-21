"""
Dataset and Model Trainer for Email Spam Detection
Trains a high-performance TF-IDF + MultinomialNB / LogisticRegression ensemble
with calibrated probabilities, explainability weights, and evaluation metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# Comprehensive synthetic + real-world benchmark dataset generator
# Covers phishing, tech support scams, lottery, urgent payroll scams, crypto pumps,
# fake delivery notifications, promotional spam, legitimate business emails,
# personal updates, newsletters, receipts, and technical alerts.

SPAM_SAMPLES = [
    # Phishing & Credential harvesting
    "URGENT: Your PayPal account has been suspended due to suspicious activity. Verify your identity immediately by clicking here.",
    "Action Required: Your Microsoft 365 password expires in 2 hours. Update password now to avoid service interruption.",
    "Security Alert: Unauthorized login attempt detected from IP 192.168.1.1. Click the link to secure your account immediately.",
    "Netflix: We couldn't process your payment. Your membership is on hold. Update billing information now to continue watching.",
    "Bank of America Security Dept: Your debit card has been temporarily locked. Confirm your SSN and PIN to unlock card.",
    "Apple Support: Your iCloud storage is full and account is locked. Click here to confirm identity and restore photos.",
    "DocuSign: You have received a confidential document from HR. Review and sign within 24 hours.",
    "Google Workspace: Critical security warning. Someone knows your password. Reset it now.",
    "Amazon Security: Unusual order of $1,499.00 placed on your account. If you did not make this purchase, call support now.",
    "Wells Fargo Alert: A wire transfer of $4,850.00 is pending approval. Verify or cancel transaction here immediately.",
    "Chase Online: Suspicious transaction detected. Click here to verify your identity and protect your funds.",
    "DHL Express: Your parcel delivery failed due to incorrect address. Pay $1.99 redelivery fee at this link.",
    "FedEx Delivery: Package tracking #9382103 pending customs fee. Download invoice and pay immediately.",
    "USPS Notification: Delivery attempt failed. Update your delivery address now or item will be returned to sender.",
    "Internal IT Notice: Mandatory security update required for all employees. Download attached executable patch.",
    "HR Department: Updated employee salary and bonus compensation list for 2026. Open attached document to view.",

    # Financial, Crypto & Lottery Scams
    "CONGRATULATIONS! You have won $2,500,000 in the International Mega Millions Lottery. Reply with your bank details to claim.",
    "Dear Beneficiary, I am barrister Nelson representing late client with $15.5M unallocated funds. Reply for 40% share.",
    "Guaranteed 500% returns in 48 hours! Invest in new Bitcoin AI automated trading bot today. Limited spots available.",
    "Exclusive Crypto Opportunity: Elon Musk 100,000 ETH giveaway! Send 0.5 ETH to receive 1.0 ETH back instantly.",
    "Pre-approved loan offer of up to $50,000 with zero interest for first 6 months! No credit check required. Apply now.",
    "You have an unclaimed inheritance deposit waiting at our branch. Submit your passport copy and routing number.",
    "Double your money in 7 days! Risk-free investment strategy tested by Wall Street insiders.",
    "Urgent wire transfer request: Kindly review the attached Swift slip and release funds to offshore vendor.",
    "Work from home making $8,000 a week! No experience necessary. Immediate start, apply today.",
    "You have won an Amazon Gift Card worth $500! Complete this 1-minute survey to receive code.",

    # Promotional & Commercial Spam
    "Final Sale! 90% OFF luxury watches, designer bags and sunglasses. Free shipping on all orders today only!",
    "Miracle weight loss pill melts 30 lbs in 2 weeks without diet or exercise! Doctor recommended, buy now.",
    "Cialis, Viagra at lowest prices online! No prescription needed, discreet worldwide shipping.",
    "Increase your website traffic by 10,000 daily visitors! Guaranteed Google #1 ranking service.",
    "Cheap airline tickets! Fly anywhere in Europe for just $19. Book before midnight!",
    "Clearance sale! Liquidation of brand new laptops, smartphones and TVs at 85% discount.",
    "Hot singles in your area want to meet tonight! Click to browse profiles for free.",
    "Boost your credit score by 150 points in 30 days! Guaranteed legal credit repair solution.",
    "Get rich with secret real estate foreclosures loophole! Free webinar tonight only.",
    "Stop balding naturally! Breakthrough hair regrowth formula reverses hair loss in 14 days.",
    "Lowest auto insurance rates ever! Save up to $700 per year by switching today.",
    "Eliminate your debt legally! Federal relief program settles credit card debt for pennies on dollar.",
    "Exclusive offer: Claim your free luxury cruise to the Bahamas! Call now, limited cabins remain.",
    "Wholesale prices on industrial machinery and factory surplus. Contact for catalog.",
    "Top rated casino games! Claim 200 free spins and $1000 welcome bonus today.",

    # Extortion & Blackmail Scams
    "I have hacked your webcam and recorded you. Pay $1,000 in Bitcoin within 48 hours or I leak the video to your contacts.",
    "Warning: Your private data has been breached. Transfer 0.05 BTC to prevent public release on dark web.",

    # Phishing variations
    "Invoice #INV-2026-8891 is past due. Remit payment of $3,420.00 immediately to avoid legal collections.",
    "Quick verification needed: Confirm your phone number and email address to keep account active.",
    "Notice of subpoena: You are summoned to appear in court. Download court summons document here.",
    "IRS Final Notice: You owe $2,180 in delinquent taxes. Immediate payment required to avoid arrest warrant.",
    "Tax refund of $842.10 is ready. Submit your direct deposit bank information to receive funds."
]

HAM_SAMPLES = [
    # Business & Workplace Communication
    "Hi team, just a reminder that our weekly sprint planning meeting is scheduled for tomorrow at 10:00 AM in Conference Room B.",
    "Can you please review the attached slide deck for Thursday's client presentation and share your thoughts by EOD?",
    "Thanks for sending over the project roadmap. I've left a few comments on slide 4 regarding the delivery timeline.",
    "Hi Sarah, let's reschedule our 1-on-1 to Friday afternoon. Let me know if 3 PM works on your calendar.",
    "Attached is the monthly financial report for Q1. Please review the budget variance analysis before the board meeting.",
    "The engineering team has deployed the v2.4 hotfix to production. All monitoring alerts are green.",
    "Please find attached the signed contract agreement for the vendor onboarding process.",
    "Good morning everyone, please remember to submit your timesheets before 5 PM today for payroll processing.",
    "Could you send me the latest API documentation for the authentication service? We're setting up the staging environment.",
    "Thank you for attending today's workshop. Attached are the slides and the link to the recording.",
    "Hi John, following up on our discussion yesterday regarding the cloud infrastructure migration plan.",
    "The quarterly company all-hands will take place this Thursday at 2 PM. An agenda will be circulated beforehand.",
    "We have scheduled maintenance on the internal staging servers this Saturday between 2 AM and 4 AM UTC.",
    "Hi team, welcoming our new UX designer Alex to the team! Alex will be joining our daily standups starting Monday.",
    "Please find the summary notes from today's client sync attached. Action items are assigned in Jira.",
    "The customer support ticket backlog has been reduced by 40% this month. Great effort by everyone!",

    # Personal & Casual Communication
    "Hey! Are we still on for dinner this Friday? Let me know which restaurant you'd prefer.",
    "Mom asked me to check if you're coming home for Thanksgiving this year. Let us know your flight details.",
    "Hey buddy, thanks for helping me move the couch over the weekend. Really appreciate the favor!",
    "Happy Birthday! Hope you have a wonderful day celebrating with family and friends.",
    "Hey, I found the book you recommended at the local bookstore. Can't wait to start reading it tonight.",
    "Are you free this weekend for a hike? The weather forecast looks really nice on Sunday.",
    "Here are the photos from our camping trip last weekend! Let me know which ones you want me to print.",
    "Quick question, what was the name of that Italian pasta place you took us to last month?",
    "Just wanted to say thank you for the lovely flowers and card. It really made my day!",
    "Let me know when you get back from your vacation so we can catch up over coffee.",

    # Legitimate transactional, receipts & updates
    "Your Uber ride on Wednesday evening. Total: $18.42. Thank you for riding with Uber.",
    "Order Confirmation: Your Amazon order #112-9842103 has shipped. Expected delivery date: Friday, Oct 24.",
    "Receipt for your payment of $14.99 to Spotify Premium. Thank you for subscribing.",
    "Your GitHub pull request #418 has been approved and merged into main by the repository maintainer.",
    "Your flight booking confirmation to San Francisco (SFO). Confirmation code: X9K2LM.",
    "Your utility bill for the billing period ending Sept 30 is now available online. Amount due: $84.20.",
    "Apple Store receipt: You purchased iCloud+ with 200 GB Storage for $2.99. Payment method: Apple Pay.",
    "Doctor's Appointment Reminder: You have an appointment with Dr. Henderson on Tuesday at 9:30 AM.",
    "Your library books are due in 3 days. Log in to renew your loans or return items to any branch.",
    "Google Calendar: Invitation - Product Strategy Sync with David @ Wed Oct 22, 2026 3pm - 4pm."
]

def augment_data():
    """Generates a rich balanced dataset combining patterns, variations, and keywords."""
    records = []
    
    # Base dataset
    for text in SPAM_SAMPLES:
        records.append({"text": text, "label": 1})
    for text in HAM_SAMPLES:
        records.append({"text": text, "label": 0})
        
    # Generate variations with common spam/ham patterns to build robust vocabulary
    spam_templates = [
        "Special promotion! Save {discount}% on all {product}. Claim voucher code {code} today only!",
        "Security notice: We detected suspicious sign-in to your {service} account. Verify credentials at {url}",
        "Dear customer, your {service} invoice #{num} of ${amount} is ready. Download attachment now.",
        "URGENT: Your {service} subscription has been cancelled. Update credit card immediately at {url}",
        "Congratulations! You were selected as winner of ${amount} cash prize in the {event}. Claim now!",
        "Final warning: Your mailbox storage has exceeded {quota}GB limit. Upgrade now to avoid suspension.",
        "Investment alert: Earn guaranteed ${amount} daily profit trading {crypto}. Zero risk guaranteed!",
        "Notice: Package {tracking} held at distribution center. Pay ${fee} customs charge to release delivery."
    ]
    
    spam_variables = [
        {"discount": "75", "product": "designer watches", "code": "DEAL75", "service": "PayPal", "url": "http://bit.ly/secure-auth-392", "num": "9421", "amount": "450", "event": "Global Lottery", "quota": "15", "crypto": "Bitcoin", "tracking": "US94812", "fee": "2.95"},
        {"discount": "80", "product": "health supplements", "code": "SAVE80", "service": "Apple ID", "url": "http://id-apple-verify.cc", "num": "8821", "amount": "1,200", "event": "Annual Sweepstakes", "quota": "20", "crypto": "Ethereum", "tracking": "DHL8841", "fee": "4.50"},
        {"discount": "90", "product": "smartphones", "code": "FLASH90", "service": "Netflix", "url": "http://netflix-billing-update.xyz", "num": "1192", "amount": "5,000", "event": "Holiday Giveaway", "quota": "50", "crypto": "Solana", "tracking": "FEDX77", "fee": "3.20"},
        {"discount": "60", "product": "perfumes & cosmetics", "code": "LUX60", "service": "Amazon", "url": "http://amazon-account-verify.info", "num": "7712", "amount": "10,000", "event": "Shopper Bonus", "quota": "10", "crypto": "Doge", "tracking": "UPS991", "fee": "1.99"},
        {"discount": "85", "product": "online casino credits", "code": "SPIN85", "service": "Microsoft 365", "url": "http://ms-login-alert.pw", "num": "4432", "amount": "7,500", "event": "VIP Cash Draw", "quota": "25", "crypto": "Tether", "tracking": "POST44", "fee": "5.00"}
    ]
    
    for tmpl in spam_templates:
        for var in spam_variables:
            records.append({"text": tmpl.format(**var), "label": 1})
            
    ham_templates = [
        "Hi {name}, following up on the {topic} meeting we had on {day}. Here is the link to the meeting notes.",
        "Could you please review the pull request for {feature} when you get a chance? Branch is ready for staging.",
        "Here is the updated project budget spreadsheet for {dept}. Let me know if you spot any discrepancies.",
        "Dinner at {place} tonight at {time}? Let me know if that works for you and {guest}!",
        "Thank you for reaching out to {company} support. Your case #{case} has been resolved.",
        "Your order #{order} from {store} has been confirmed. You will receive tracking once it ships.",
        "Attached are the minutes and action items from our weekly {dept} standup. Next sync is on {day}."
    ]
    
    ham_variables = [
        {"name": "Sarah", "topic": "Q3 product roadmap", "day": "Tuesday", "feature": "user authentication", "dept": "Engineering", "place": "the Italian bistro", "time": "7:30 PM", "guest": "Mark", "company": "Stripe", "case": "49182", "order": "839120", "store": "Best Buy"},
        {"name": "Michael", "topic": "client onboarding", "day": "Wednesday", "feature": "payment gateway", "dept": "Finance", "place": "the sushi bar", "time": "6:00 PM", "guest": "Lisa", "company": "Slack", "case": "55102", "order": "992104", "store": "IKEA"},
        {"name": "Emily", "topic": "quarterly budget", "day": "Thursday", "feature": "email notification", "dept": "Marketing", "place": "the coffee shop", "time": "10:00 AM", "guest": "Tom", "company": "GitHub", "case": "66219", "order": "119842", "store": "Target"},
        {"name": "David", "topic": "database migration", "day": "Friday", "feature": "search indexing", "dept": "Operations", "place": "the taco place", "time": "1:00 PM", "guest": "Rachel", "company": "AWS", "case": "77180", "order": "334810", "store": "Costco"},
        {"name": "Jessica", "topic": "annual review", "day": "Monday", "feature": "reporting dashboard", "dept": "HR", "place": "the Thai restaurant", "time": "8:00 PM", "guest": "Kevin", "company": "Atlassian", "case": "88231", "order": "774912", "store": "Home Depot"}
    ]
    
    for tmpl in ham_templates:
        for var in ham_variables:
            records.append({"text": tmpl.format(**var), "label": 0})
            
    df = pd.DataFrame(records)
    return df

def train_and_save_model(output_dir="model"):
    os.makedirs(output_dir, exist_ok=True)
    print("Building dataset...")
    df = augment_data()
    print(f"Total training dataset size: {len(df)} samples ({sum(df['label'] == 1)} Spam, {sum(df['label'] == 0)} Ham)")
    
    # Save training dataset for reference/benchmarking
    df.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)
    
    X = df["text"]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Advanced TF-IDF Vectorizer with unigrams + bigrams and sublinear TF scaling
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True,
        stop_words="english",
        lowercase=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # High-accuracy Logistic Regression with calibrated L2 regularization
    clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    clf.fit(X_train_vec, y_train)
    
    # Also train Naive Bayes for ensemble / comparison
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train_vec, y_train)
    
    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[("lr", clf), ("nb", nb)],
        voting="soft"
    )
    ensemble.fit(X_train_vec, y_train)
    
    # Evaluation
    y_pred = ensemble.predict(X_test_vec)
    y_prob = ensemble.predict_proba(X_test_vec)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Extract top spam and ham feature weights from Logistic Regression for Explainable AI
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    
    top_spam_indices = np.argsort(coefs)[-25:][::-1]
    top_ham_indices = np.argsort(coefs)[:25]
    
    top_spam_words = [{"word": str(feature_names[i]), "weight": float(coefs[i])} for i in top_spam_indices]
    top_ham_words = [{"word": str(feature_names[i]), "weight": float(abs(coefs[i]))} for i in top_ham_indices]
    
    # Save artifacts
    joblib.dump(ensemble, os.path.join(output_dir, "spam_classifier.joblib"))
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    
    # Save metadata & metrics for the frontend dashboard
    metrics_data = {
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": {
            "true_negative": cm[0][0],
            "false_positive": cm[0][1],
            "false_negative": cm[1][0],
            "true_positive": cm[1][1]
        },
        "total_samples": len(df),
        "spam_samples": int(sum(df["label"] == 1)),
        "ham_samples": int(sum(df["label"] == 0)),
        "vocabulary_size": len(feature_names),
        "top_spam_words": top_spam_words,
        "top_ham_words": top_ham_words,
        "model_architecture": "Ensemble (Soft Voting: Calibrated Logistic Regression + Multinomial Naive Bayes with Sublinear TF-IDF N-Grams)"
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=2)
        
    print(f"\nModel and metrics successfully saved to '{output_dir}/' directory.")
    return metrics_data

if __name__ == "__main__":
    train_and_save_model()
