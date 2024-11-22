import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# App title
#st.title("NLC-BANK")
st.markdown(
    "<h1 style='text-align: center; font-size: 60px;'>NLC BANK</h1>", 
    unsafe_allow_html=True
)

st.sidebar.header("Navigation")

# Function to preprocess feedback
def preprocess_feedback(feedback_text):
    feedback_text = feedback_text.lower()
    feedback_text = re.sub(r'[^a-z\s]', '', feedback_text)
    doc = nlp(feedback_text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Function to perform sentiment analysis
def perform_sentiment_analysis(feedback):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(feedback)
    return sentiment_scores['compound']

# Function for topic modeling
def lda_topic_modeling(text_data, num_topics=5):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(text_data)
    nmf_model = NMF(n_components=num_topics, random_state=42)
    dtm_reduced = nmf_model.fit_transform(dtm)
    lda_model = LDA(n_components=num_topics, random_state=42)
    lda_model.fit(dtm_reduced)
    return lda_model, vectorizer, nmf_model

# Function to get top topics
def get_top_feedback_topics(lda_model, vectorizer, top_n=10):
    words = vectorizer.get_feature_names_out()
    topics = {}
    for idx, topic in enumerate(lda_model.components_):
        topics[f'Topic {idx}'] = [words[i] for i in topic.argsort()[-top_n:]]
    return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in topics.items()]))

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Preprocessing
    st.subheader("Preprocessing Feedback")
    df['Cleaned Feedback'] = df['Feedback Text'].apply(preprocess_feedback)
    st.write("Cleaned Feedback Example:")
    st.write(df[['Feedback Text', 'Cleaned Feedback']].head())

    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
    df['Sentiment Score'] = df['Cleaned Feedback'].apply(perform_sentiment_analysis)
    df['Sentiment Category'] = df['Sentiment Score'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )
    st.write("Sentiment Analysis Results:")
    st.write(df[['Feedback Text', 'Sentiment Score', 'Sentiment Category']].head())

    # Visualize Sentiment Distribution
    st.subheader("Sentiment Distribution")
    
    # Bar Chart
    fig_bar, ax_bar = plt.subplots()
    sns.countplot(data=df, x='Sentiment Category', ax=ax_bar)
    plt.title("Sentiment Distribution")
    st.pyplot(fig_bar)

    # Pie Chart
    st.subheader("Sentiment Distribution - Pie Chart")
    sentiment_counts = df['Sentiment Category'].value_counts()
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=['green', 'blue', 'red'])
    ax_pie.set_title("Sentiment Distribution")
    st.pyplot(fig_pie)

    # Topic Modeling
    st.subheader("Topic Modeling")
    lda_model, vectorizer, nmf_model = lda_topic_modeling(df['Cleaned Feedback'], num_topics=5)
    topics_df = get_top_feedback_topics(lda_model, vectorizer)
    st.write("Top Topics:")
    st.write(topics_df)

    # Ranked Areas Lacking
    st.subheader("Ranked Areas Where the Bank is Lacking")
    if 'Areas Bank is Lacking' in df.columns:
        complaint_counts = df['Areas Bank is Lacking'].value_counts().reset_index()
        complaint_counts.columns = ['Area Lacking', 'Count']
        st.write(complaint_counts)

        fig, ax = plt.subplots()
        sns.barplot(x='Count', y='Area Lacking', data=complaint_counts, palette='Set2', ax=ax)
        plt.title("Ranked Areas Where the Bank is Lacking")
        st.pyplot(fig)
    else:
        st.write("The dataset does not have a column named 'Areas Bank is Lacking'.")

    # Download cleaned data
    st.subheader("Download Cleaned Data")
    cleaned_data_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", cleaned_data_csv, "cleaned_feedback.csv", "text/csv")
else:
    st.write("Please upload a CSV file to start.")
