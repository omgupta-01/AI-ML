"""
generate_sample_data.py — Generate synthetic labeled support tickets for demo/training
Produces 15,000 labeled tickets across 6 classes
"""

import random
import pandas as pd

random.seed(42)

TEMPLATES = {
    "billing": [
        "I have been charged twice for the same transaction.",
        "My invoice shows incorrect charges for last month.",
        "Why was I billed {amount} when my plan costs {plan_cost}?",
        "I need a refund for a duplicate payment made on {date}.",
        "There is an unauthorized charge of {amount} on my account.",
        "My bill is higher than expected this month.",
        "Please explain the additional fee on my latest statement.",
        "I was promised a discount but it's not reflected in my invoice.",
    ],
    "account": [
        "I cannot log into my account since yesterday.",
        "Please help me reset my password, the link is not working.",
        "My account has been locked after multiple failed attempts.",
        "I need to update my registered email address.",
        "How do I add a secondary user to my account?",
        "I want to close my account and retrieve my data.",
        "My profile information is showing incorrect details.",
        "Two-factor authentication is not working on my phone.",
    ],
    "technical": [
        "The application crashes every time I try to generate a report.",
        "I'm getting a 500 internal server error on the dashboard.",
        "The data export feature is not working correctly.",
        "Charts are not loading on my browser.",
        "API calls are returning incorrect data for endpoint /transactions.",
        "The mobile app freezes when I open the portfolio section.",
        "I cannot upload documents; the upload fails at 80%.",
        "Real-time data feed seems to have a delay of over 10 minutes.",
    ],
    "compliance": [
        "I need a statement for regulatory submission by {date}.",
        "Please provide documentation for my KYC update.",
        "My account verification has been pending for 3 weeks.",
        "I need an audited transaction history for the last 12 months.",
        "How do I submit a FATCA declaration through your platform?",
        "I have received a compliance notice and need assistance.",
        "Please clarify the data retention policy for my account records.",
        "I need a letter of confirmation for tax purposes.",
    ],
    "onboarding": [
        "I just signed up but I cannot access any features yet.",
        "I don't know how to set up my first portfolio.",
        "Can someone walk me through the account setup process?",
        "I completed registration but my account is still under review.",
        "How do I link my bank account to start investing?",
        "The welcome email link has expired. Please resend it.",
        "I need help understanding what documents I need to submit.",
        "I am new here and confused about where to start.",
    ],
    "general": [
        "What are your customer support hours?",
        "How do I contact a financial advisor?",
        "Do you have a mobile application available?",
        "What investment products do you offer?",
        "Can I transfer my account from another broker?",
        "What is the minimum investment amount?",
        "Do you provide tax-loss harvesting services?",
        "How is my data protected on your platform?",
    ],
}

DATES   = ["January 15", "February 3", "March 22", "April 10"]
AMOUNTS = ["₹5,000", "₹12,500", "₹3,200", "₹8,750"]
COSTS   = ["₹2,000", "₹4,500", "₹1,500"]

def fill_template(text):
    return (text
            .replace("{date}",      random.choice(DATES))
            .replace("{amount}",    random.choice(AMOUNTS))
            .replace("{plan_cost}", random.choice(COSTS)))


rows = []
for label, templates in TEMPLATES.items():
    # Each class gets ~2,500 samples
    while len([r for r in rows if r["label"] == label]) < 2500:
        t = fill_template(random.choice(templates))
        rows.append({"text": t, "label": label})

random.shuffle(rows)
df = pd.DataFrame(rows[:15000])

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/tickets.csv", index=False)
print(f"Saved data/tickets.csv  ({len(df):,} rows)")
print(df["label"].value_counts())
