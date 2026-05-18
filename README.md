# 🛡️ Credible: AI-Powered Fake News Detector
> **A hybrid fact-checking engine combining the speed of 5 Machine Learning algorithms with real-time web verification powered by the Groq API.**

**Credible** is an advanced fake news detection system designed to combat misinformation. By leveraging a multi-model ensemble of traditional ML algorithms for rapid text analysis and the Groq API for cross-referencing claims against live web data, Credible delivers high-accuracy credibility scores for news articles, tweets, and digital content.

---
## ⚡ How It Works

Credible uses a **two-step verification pipeline**:
1. **Initial ML Classification:** The text is passed through 5 distinct Machine Learning algorithms (e.g., Logistic Regression, Random Forest, SVM, Naive Bayes, Gradient Boosting) to analyze linguistic patterns, emotional tone, and structural red flags associated with misinformation.
2. **LLM Fact-Checking (Groq API):** If the content is flagged as suspicious or needs deeper context, it triggers an ultra-fast web search and evaluation using an LLM hosted on Groq's LPU infrastructure, cross-referencing the claims against trusted online sources in real-time.

--- 

## ✨ Features

- **Ensemble ML Detection:** Utilizes 5 different machine learning models to ensure robust baseline classification and minimize false positives.
- **Lightning-Fast LLM Fact-Checking:** Integrated with the Groq API for near-instantaneous natural language processing and web context retrieval.
- **Real-Time Web Search:** Doesn't just rely on static training data; validates claims against current web events.
- **Confidence Scoring:** Outputs a clear percentage-based credibility score alongside a brief explanation of *why* the text was flagged.
- **API Ready:** Easily integrate Credible's detection pipeline into browser extensions, social media bots, or news aggregator platforms.

---

## 🛠 Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-Learn
- **LLM / Inference:** [Groq API] (https://groq.com/)
- **Web Framework:** Streamlit 

---

## 🚀 Getting Started

Follow these instructions to get Credible running on your local machine.

### Prerequisites

* Python 3.8 or higher
* A [Groq API Key](https://console.groq.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/credible.git](https://github.com/yourusername/credible.git)
   cd credible
   
