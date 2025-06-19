YT-Comment-Sentiment-Analysis
==============================

**A browser-integrated machine learning pipeline for real-time sentiment analysis of YouTube comments, enabling data-driven decision-making for creators and digital marketers.**


---

## 📌 Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [Business Objective](#business-objective)
* [Problem Statement](#problem-statement)
* [Project Architecture](#project-architecture)
* [Modeling Pipeline](#modeling-pipeline)
* [Performance Metrics](#performance-metrics)
* [CI/CD & Deployment](#cicd--deployment)
* [Tech Stack](#tech-stack)
* [Installation & Setup](#installation--setup)
* [Future Scope](#future-scope)
* [License](#license)

---

## 🚀 Overview

This project delivers a **Chrome browser extension** that extracts YouTube comments in real time and classifies them into **positive, negative, or neutral sentiments** using a custom-trained NLP model. It bridges **natural language understanding** with **web augmentation** and deploys at production scale through a robust, auto-scaled infrastructure on AWS.

---

## ✨ Key Features

* Chrome Extension for extracting YouTube comments dynamically on any video page
* LightGBM-based sentiment classifier with **87% F1-score** on test data
* Advanced NLP pipeline with **spaCy + TF-IDF + trigrams**, improving performance by **17%** over baseline models
* **Real-time predictions** displayed directly within the YouTube interface
* Production-grade deployment using **Flask + Docker**, with **Rolling Deployment** on **AWS Auto Scaling Group**
* Full **CI/CD pipeline** integrated via GitHub Actions for scalable delivery and automatic retraining

---

## 🎯 Business Objective

Content creators and digital marketing professionals often struggle to gauge viewer sentiment from thousands of YouTube comments. This tool provides **instant sentiment insights** directly on the platform, enabling:

* Feedback mining for creative optimization
* Brand reputation monitoring
* Automated moderation support
* Audience engagement analysis

---

## ❓ Problem Statement

How can we classify the sentiment of YouTube comments in real-time with high accuracy, minimal latency, and full integration into the YouTube UI — while maintaining a scalable backend architecture?

---

## 🧠 Project Architecture
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

---

## ⚙️ Modeling Pipeline

* **Text Preprocessing**: Lowercasing, lemmatization, stopword removal with `spaCy`
* **Feature Engineering**: TF-IDF Vectorization with unigrams + trigrams
* **Model**: `LightGBMClassifier` with tuned hyperparameters
* **Baseline Comparison**: Outperformed `RandomForestClassifier` by **17% F1 improvement**
* **Inference API**: Flask REST service wrapped in a Docker container

---

## 📊 Performance Metrics

| Metric        | Value |
| ------------- | ----- |
| **F1-Score**  | 87%   |
| **Accuracy**  | 89.4% |
| **Precision** | 86%   |
| **Recall**    | 88%   |

The model demonstrated balanced performance across all three sentiment classes, making it ideal for real-world comment streams.

---

## 🚀 CI/CD & Deployment

* **Containerization**: Dockerized Flask API
* **CI/CD Pipeline**: GitHub Actions for:

  * Code linting
  * Unit testing
  * Image builds and pushes
* **Deployment Strategy**:

  * **Rolling Deployment** using **AWS CodeDeploy**
  * Hosted on **Auto Scaling Group (ASG)** for horizontal scalability
  * Supports live traffic without downtime or service disruption
* **Monitoring**: AWS CloudWatch for logging and health checks

---

## 🧰 Tech Stack

| Component            | Technology                          |
| -------------------- | ----------------------------------- |
| **Language**         | Python                              |
| **NLP**              | spaCy, scikit-learn, LightGBM       |
| **Frontend**         | JavaScript, Chrome Extension API    |
| **Model Serving**    | Flask                               |
| **Containerization** | Docker                              |
| **CI/CD**            | GitHub Actions, AWS CodeDeploy      |
| **Infrastructure**   | EC2, Auto Scaling Group, CloudWatch |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/youtube-sentiment-analysis.git
cd youtube-sentiment-analysis
```

### 2. Build Docker Image

```bash
docker build -t yt-sentiment-api .
```

### 3. Run API Locally

```bash
docker run -p 5000:5000 yt-sentiment-api
```

### 4. Load Chrome Extension

* Navigate to `chrome://extensions`
* Enable **Developer Mode**
* Click on **"Load Unpacked"**
* Select the `/extension` folder from this repository

### 5. Use the Tool

* Open any YouTube video
* Click the extension icon
* Get real-time sentiment analysis of visible comments

---

## 🔮 Future Scope

* Deploy deep learning alternatives (BERT, RoBERTa) for sentiment classification
* Integrate support for **multilingual comments**
* Add **emotion classification** beyond sentiment (joy, anger, etc.)
* Build an **analytics dashboard** for content creators to track trends
* Integrate with YouTube’s API v3 for authorized comment access

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

