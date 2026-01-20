import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from src.preprocessing import clean_text
from src.sentiment_analysis import get_sentiment
from src.topic_modeling import train_lda

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ReviewScope", layout="wide")

st.title("📊 ReviewScope – Smart Review Analysis Platform")
st.caption(
    "A professional NLP platform for sentiment analysis, topic modeling, "
    "and keyword insights from customer reviews"
)

# ---------------- HELPER FUNCTION ----------------
def get_top_keywords(text_series, top_n=15):
    words = " ".join(text_series).split()
    return Counter(words).most_common(top_n)

# ---------------- MAIN TABS ----------------
tab_overview, tab_single, tab_dataset, tab_keywords, tab_summary = st.tabs(
    ["🏠 Overview", "📝 Single Text Analysis", "📂 Dataset Analysis", "🔑 Keyword Insights", "📊 Summary"]
)

# ================= OVERVIEW TAB =================
with tab_overview:
    st.subheader("🔍 About ReviewScope")

    st.markdown("""
    **ReviewScope** is an interactive text analytics platform designed to extract
    **sentiment trends, discussion topics, and key terms** from textual data such as
    customer reviews.

    ### 🔧 Core Capabilities
    - Sentiment Analysis (Positive / Neutral / Negative)
    - Topic Modeling using LDA
    - Keyword Frequency Analysis
    - Interactive visualizations and dashboards

    ### 🎯 Use Cases
    - Product review analysis
    - Customer feedback analysis
    - Opinion mining for decision support
    """)

# ================= SINGLE TEXT ANALYSIS =================
with tab_single:
    st.subheader("📝 Instant Review Analysis")
    st.caption("Analyze a single review instantly by copy–pasting text")

    user_text = st.text_area(
        "Paste a review below",
        placeholder="Example: The product quality is excellent and delivery was fast."
    )

    if st.button("🔍 Analyze Review"):
        if user_text.strip() == "":
            st.warning("Please enter some text to analyze")
        else:
            sentiment = get_sentiment(user_text)
            st.success(f"**Detected Sentiment:** {sentiment}")

# ================= DATASET ANALYSIS =================
with tab_dataset:
    st.subheader("📂 Dataset-Based Analysis")
    st.caption("Upload a CSV file to analyze large-scale review data")

    uploaded_file = st.file_uploader("Upload CSV file (must contain a 'review' column)", type=["csv"])

    num_topics = st.slider(
        "🧠 Select Number of Topics for Topic Modeling",
        min_value=2,
        max_value=10,
        value=5
    )

    run_button = st.button("🚀 Run Full Analysis")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "review" not in df.columns:
            st.error("CSV must contain a column named 'review'")
        else:
            st.subheader("📄 Dataset Preview")
            st.dataframe(df.head())

            if run_button:
                with st.spinner("Processing dataset..."):
                    df["clean_text"] = df["review"].apply(clean_text)
                    df["sentiment"] = df["review"].apply(get_sentiment)

                st.success("Dataset analysis completed")

                st.subheader("😊 Sentiment Distribution")

                sentiment_counts = df["sentiment"].value_counts()
                sentiment_percent = (sentiment_counts / sentiment_counts.sum()) * 100

                col1, col2 = st.columns(2)

                with col1:
                    st.bar_chart(sentiment_percent)

                with col2:
                    fig, ax = plt.subplots()
                    ax.pie(
                        sentiment_counts,
                        labels=sentiment_counts.index,
                        autopct="%1.1f%%",
                        startangle=90
                    )
                    ax.axis("equal")
                    st.pyplot(fig)

                st.subheader("🧠 Topic Modeling Results")
                lda, topics, coherence = train_lda(df["clean_text"], num_topics=num_topics)
                st.metric("Coherence Score", round(coherence, 3))

                for topic in topics:
                    st.write(topic)

# ================= KEYWORD INSIGHTS =================
with tab_keywords:
    st.subheader("🔑 Keyword Insights")
    st.caption("Explore the most frequent and influential terms in the dataset")

    if "df" in locals() and "clean_text" in df.columns:
        keywords = get_top_keywords(df["clean_text"], top_n=15)
        keyword_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])

        colA, colB = st.columns(2)

        with colA:
            st.subheader("Top Keywords (Frequency)")
            st.bar_chart(keyword_df.set_index("Keyword"))

        with colB:
            st.subheader("Keyword Frequency Table")
            st.dataframe(keyword_df)
    else:
        st.info("Run dataset analysis first to view keyword insights")

# ================= SUMMARY DASHBOARD =================
with tab_summary:
    st.subheader("📊 Analysis Summary Dashboard")
    st.caption("High-level insights and statistics")

    if "df" in locals() and "sentiment" in df.columns:
        colX, colY, colZ = st.columns(3)

        colX.metric("Total Reviews", len(df))
        colY.metric("Positive Reviews", (df["sentiment"] == "Positive").sum())
        colZ.metric("Negative Reviews", (df["sentiment"] == "Negative").sum())
    else:
        st.info("Run dataset analysis to view summary metrics")
