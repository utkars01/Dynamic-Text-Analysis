import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from src.preprocessing import clean_text
from src.sentiment_analysis import get_sentiment
from src.topic_modeling import train_lda

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ReviewScope", layout="wide")

# ---------------- THEME TOGGLE ----------------
theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        "<style>body{background-color:#0e1117;color:white}</style>",
        unsafe_allow_html=True
    )

st.title("📊 ReviewScope – Smart Review Analysis Platform")
st.caption(
    "An interactive NLP platform for sentiment analysis, topic modeling, "
    "and keyword insights from textual data"
)

# ---------------- HELPER FUNCTIONS ----------------
def get_top_keywords(text_series, top_n=20):
    words = " ".join(text_series).split()
    return Counter(words).most_common(top_n)

def sentiment_confidence(text):
    pos_words = ["good","great","excellent","amazing","love","perfect","best"]
    neg_words = ["bad","worst","poor","terrible","hate","waste","broken"]
    score = sum(w in text.lower() for w in pos_words) - sum(w in text.lower() for w in neg_words)
    return min(max((score + 5) * 10, 0), 100)

# ---------------- MAIN TABS ----------------
tabs = st.tabs([
    "🏠 Overview",
    "📝 Live Text Analysis",
    "📂 Dataset Analysis",
    "🔑 Keyword Insights",
    "📊 Dashboard"
])

# ================= OVERVIEW =================
with tabs[0]:
    st.subheader("🔍 About ReviewScope")
    st.markdown("""
    **ReviewScope** is a smart text analytics platform that helps users understand
    large volumes of textual data by extracting:
    - Sentiment trends
    - Hidden discussion topics
    - Frequently used keywords

    **Use cases:**
    - Product review analysis
    - Customer feedback monitoring
    - Opinion mining
    """)

# ================= LIVE TEXT ANALYSIS =================
with tabs[1]:
    st.subheader("📝 Instant Review Analysis")
    st.caption("Paste or type a review to analyze sentiment in real time")

    text = st.text_area("Enter review text")

    if text.strip():
        sentiment = get_sentiment(text)
        confidence = sentiment_confidence(text)

        st.success(f"Sentiment: **{sentiment}**")
        st.progress(confidence / 100)
        st.caption(f"Confidence Score: {confidence}%")

# ================= DATASET ANALYSIS =================
with tabs[2]:
    st.subheader("📂 Dataset-Based Analysis")
    st.caption("Upload a CSV file with a `review` column")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    num_topics = st.slider("Number of Topics", 2, 10, 5)
    run = st.button("🚀 Run Analysis")

    if uploaded:
        df = pd.read_csv(uploaded)

        if "review" not in df.columns:
            st.error("CSV must contain a column named 'review'")
        else:
            st.dataframe(df.head())

            if run:
                with st.spinner("Processing dataset..."):
                    df["clean_text"] = df["review"].apply(clean_text)
                    df["sentiment"] = df["review"].apply(get_sentiment)

                st.success("Analysis completed")

                st.subheader("😊 Sentiment Distribution")
                sentiment_counts = df["sentiment"].value_counts()
                sentiment_percent = sentiment_counts / sentiment_counts.sum() * 100

                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(sentiment_percent)
                with col2:
                    fig, ax = plt.subplots()
                    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%")
                    ax.axis("equal")
                    st.pyplot(fig)

                st.subheader("🧠 Topic Modeling")
                lda, topics, coherence = train_lda(df["clean_text"], num_topics)
                st.metric("Coherence Score", round(coherence, 3))
                for t in topics:
                    st.write(t)

                st.subheader("🔍 Search Reviews")
                keyword = st.text_input("Search keyword")
                if keyword:
                    st.dataframe(df[df["review"].str.contains(keyword, case=False)])

# ================= KEYWORD INSIGHTS =================
with tabs[3]:
    st.subheader("🔑 Keyword Overview")

    if "df" in locals() and "clean_text" in df.columns:
        keywords = get_top_keywords(df["clean_text"])
        kw_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])

        colA, colB = st.columns(2)
        with colA:
            st.bar_chart(kw_df.set_index("Keyword"))
        with colB:
            st.dataframe(kw_df)
    else:
        st.info("Run dataset analysis first to generate keyword insights")


# ================= DASHBOARD =================
with tabs[4]:
    st.subheader("📊 Summary Dashboard")

    if "df" in locals():
        colX, colY, colZ = st.columns(3)
        colX.metric("Total Reviews", len(df))
        colY.metric("Positive Reviews", (df["sentiment"] == "Positive").sum())
        colZ.metric("Negative Reviews", (df["sentiment"] == "Negative").sum())
    else:
        st.info("No data available yet")
