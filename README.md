# Tweet_Emotion_Classifier-Insights_into_Netflix_vs.Apple_Sentiment
A sentiment analysis project comparing tweets about Netflix and Apple. The project uses machine learning models like Logistic Regression and Random Forest, with TF-IDF for feature extraction, to classify sentiments as positive or negative.

---

# Netflix_vs_Apple_Sentiment_Analysis

A sentiment analysis project comparing public perceptions of Netflix and Apple using machine learning and natural language processing.

---

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Results and Insights](#results-and-insights)
- [Future Scope](#future-scope)
- [Contributors](#contributors)

---

## Introduction
This project focuses on analyzing tweets related to Netflix and Apple to identify sentiments as positive or negative. It combines machine learning and natural language processing to process and classify text data. Visualizations are included to compare sentiment trends between the two companies.

---

## Project Structure
```
Netflix_vs_Apple_Sentiment_Analysis/
│
├── data/
│   ├── netflix_tweets.csv
│   ├── apple_tweets.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│
├── app/
│   ├── app.py (Streamlit app)
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── model_training.ipynb
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Features
- Sentiment classification of tweets as positive or negative.
- Comparison of sentiment trends for Netflix and Apple.
- Visualization of tweet sentiment distribution.
- Real-time sentiment prediction through a Streamlit app.

---

## Dataset
- **Source**: Kaggle (or mention the actual dataset source).
- **Size**: Contains thousands of tweets for both Netflix and Apple.
- **Columns**: `Tweet`, `Company`, `Sentiment` (Positive/Negative).

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit
- **Machine Learning Models**: Logistic Regression, Random Forest

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Netflix_vs_Apple_Sentiment_Analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Netflix_vs_Apple_Sentiment_Analysis
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

---

## Results and Insights
- **Accuracy**: Achieved high classification accuracy using Logistic Regression and Random Forest models.
- **Insights**: Apple received more positive sentiments compared to Netflix, highlighting key public perceptions.

---

## Future Scope
- Extend the analysis to include other companies or industries.
- Improve the model by incorporating more complex NLP techniques like BERT.
- Deploy the Streamlit app online for broader accessibility.

---

## Contributors
- **Rishikesh** 
- Linkedin 
- Email_id 

---

