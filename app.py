import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the pre-trained model and vectorizer
try:
    model = joblib.load("logistic_regression_tweet_emotion.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError as e:
    st.error("Required files not found: logistic_regression_tweet_emotion.pkl or tfidf_vectorizer.pkl")
    st.stop()

# Set up Streamlit App
st.title("Tweet Emotion Analyzer")
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Choose a feature:",
    ["Home", "Real-time Tweet Analysis", "Dataset Analysis", "Model Comparison", "Download Model"]
)

# Feature 1: Home
if options == "Home":
    st.header("Welcome to the Tweet Emotion Analyzer")
    st.write(
        """
        This app analyzes the emotion of tweets (Positive or Negative) using Machine Learning.
        Upload a dataset, classify tweets, and explore sentiment trends for companies like Netflix and Apple!
        """
    )

# Feature 2: Real-time Tweet Analysis
elif options == "Real-time Tweet Analysis":
    st.header("Real-time Tweet Analysis")
    user_input = st.text_area("Enter a tweet for emotion analysis:")
    if st.button("Analyze"):
        if user_input:
            cleaned_input = [" ".join(word for word in user_input.lower().split())]  # Minimal cleaning
            vectorized_input = vectorizer.transform(cleaned_input)
            prediction = model.predict(vectorized_input)[0]
            confidence = model.predict_proba(vectorized_input).max()
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.write(f"Sentiment: **{sentiment}** with confidence of {confidence:.2f}")
        else:
            st.write("Please enter a tweet!")

# Feature 3: Dataset Analysis
elif options == "Dataset Analysis":
    st.header("Dataset Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.write(data.head())
        st.write("### Dataset Statistics:")
        st.write(data.describe())

        # Sentiment Distribution
        st.write("### Sentiment Distribution")
        sentiment_counts = data['label'].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', color=['red', 'green'], ax=ax)
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# Feature 4: Model Comparison
elif options == "Model Comparison":
    st.header("Model Comparison")
    st.write("This section compares Logistic Regression, SVM, and Naive Bayes models.")

    # Add dummy data for visualization
    comparison_data = {
        "Model": ["Logistic Regression", "SVM", "Naive Bayes"],
        "Accuracy": [0.85, 0.82, 0.80],
        "F1-Score": [0.86, 0.83, 0.81],
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.write(comparison_df)

    # Bar chart for comparison
    fig, ax = plt.subplots()
    comparison_df.set_index("Model").plot(kind="bar", ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    st.pyplot(fig)

# Feature 5: Download Model
elif options == "Download Model":
    st.header("Download Trained Model")
    with open("logistic_regression_tweet_emotion.pkl", "rb") as f:
        st.download_button(
            "Download Logistic Regression Model", f, file_name="logistic_regression_tweet_emotion.pkl"
        )
    st.write("Download the pre-trained Logistic Regression model for tweet emotion classification.")
