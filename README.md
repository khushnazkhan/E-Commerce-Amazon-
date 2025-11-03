## 📊 Amazon Product Reviews - Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-red)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)

A comprehensive sentiment analysis project that classifies Amazon product reviews into **Positive**, **Negative**, and **Neutral** sentiments using traditional machine learning and deep learning approaches.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Business Objective](#-business-objective)
- [Dataset](#-dataset)
- [Technical Approach](#-technical-approach)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Future Enhancements](#-future-enhancements)

## 🎯 Project Overview

This project tackles the challenge of sentiment classification for Amazon product reviews, addressing critical aspects like **class imbalance** and implementing both traditional and advanced machine learning techniques. The solution provides valuable insights for e-commerce businesses to understand customer feedback at scale.

### 🎯 Business Objective
- Automate sentiment analysis of product reviews
- Help businesses understand customer satisfaction levels
- Identify areas for product improvement
- Monitor brand perception over time

## 📊 Dataset

The dataset contains Amazon product reviews with the following features:

| Column | Description |
|--------|-------------|
| `Name of the product` | Product name and description |
| `Product Brand` | Manufacturer/brand |
| `categories` | Product categorization |
| `primaryCategories` | Main product category |
| `reviews.date` | Date of review |
| `reviews.text` | Main review content |
| `reviews.title` | Review title |
| `sentiment` | Sentiment label (Positive/Negative/Neutral) |

### 📁 Files
- `train_data.csv` - Training dataset
- `test_data.csv` - Test dataset  
- `test_data_hidden.csv` - Hidden test set

### 📈 Data Statistics
- **Total Samples**: 6,000 reviews
- **Sentiment Distribution**:
  - Positive: 4,686 (78.1%)
  - Neutral: 197 (3.3%)
  - Negative: 117 (1.9%)
  - Missing: 1,000 (16.7%)

## 🛠️ Technical Approach

### 🔄 Workflow Pipeline
1. **Data Preprocessing & Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Class Imbalance Handling**
4. **Feature Engineering**
5. **Model Development & Training**
6. **Model Evaluation & Comparison**
7. **Topic Modeling**
8. **Model Optimization**

### 🤖 Models Implemented

#### Traditional Machine Learning
- **Multinomial Naive Bayes**
- **Support Vector Machine (SVM)**
- **XGBoost**

#### Deep Learning
- **Neural Networks**
- **LSTM (Long Short-Term Memory)**
- **Ensemble Methods**

### ⚙️ Feature Engineering
- **TF-IDF Vectorization**
- **Custom Sentiment Scores**
- **Text Preprocessing** (cleaning, tokenization)

### ⚖️ Handling Class Imbalance
- **Random Oversampling** using `imblearn`
- Appropriate evaluation metrics (F1-score, Precision, Recall)

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install imbalanced-learn tensorflow keras gensim xgboost
pip install jupyter notebook
Amazon_Sentiment_Analysis/
│
├── E_Commerce_Amazon_Project_Capstone_2.ipynb  # Main project notebook
├── train_data.csv                              # Training dataset
├── test_data.csv                               # Test dataset
├── test_data_hidden.csv                        # Hidden test dataset
├── README.md                                   # Project documentation
├── requirements.txt                            # Python dependencies
└── images/                                     # Visualization outputs
    ├── sentiment_distribution.png
    ├── model_comparison.png
    └── topic_modeling.png
