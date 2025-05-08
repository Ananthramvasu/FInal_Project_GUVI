import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page",  ["Sentiment Analysis"])

# Home page
if page == "Home":
    st.title("Welcome to the Insurance AI System")
    st.write("Select a page from the sidebar.")

# Sentiment Analysis page
elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis of Customer Feedback")

    # Input text
    feedback_text = st.text_area("Enter Customer Feedback:")

    # Sentiment analyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()

    if st.button("Analyze Sentiment"):
        if feedback_text:
            sentiment_score = sentiment_analyzer.polarity_scores(feedback_text)
            compound_score = sentiment_score['compound']
            sentiment_label = (
                "Positive" if compound_score >= 0.05 else
                "Negative" if compound_score <= -0.05 else
                "Neutral"
            )
            st.write(f"Sentiment Score: {compound_score}")
            st.success(f"Predicted Sentiment: {sentiment_label}")
        else:
            st.error("Please enter feedback text to analyze.")
