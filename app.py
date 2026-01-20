import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from src.preprocessing import clean_text
from src.sentiment_analysis import get_sentiment
from src.topic_modeling import train_lda


st.set_page_config(page_title="ReviewScope", layout="wide")

st.title("📊 ReviewScope – Smart Review Analysis Platform")
st.caption("Analyze customer reviews using sentiment analysis, topic modeling, and keyword insights")


st.subheader("📝 Instant Review Analysis")

user_text = st.text_area(
    "Paste a review below",
    placeholder="The product quality is excellent and delivery was very fast."
)

if st.button("🔍 Analyze Text"):
    if user_text.strip() == "":
        st.warning("Please enter some text")
    else:
        sentiment = get_sentiment(user_text)
        st.success(f"**Sentiment:** {sentiment}")

st.divider()


st.sidebar.header("⚙️ Dataset Analysis Controls")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])

num_topics = st.sidebar.slider(
    "🧠 Number of Topics",
    min_value=2,
    max_value=10,
    value=5
)

run_button = st.sidebar.button("🚀 Run Dataset Analysis")


def get_top_keywords(text_series, top_n=15):
    words = " ".join(text_series).split()
    return Counter(words).most_common(top_n)


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("❌ CSV must contain a column named 'review'")
    else:
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        if run_button:
            with st.spinner("🔄 Processing dataset..."):
                df["clean_text"] = df["review"].apply(clean_text)
                df["sentiment"] = df["review"].apply(get_sentiment)

            st.success("✅ Dataset analysis completed")

            tab1, tab2, tab3, tab4 = st.tabs(
                ["😊 Sentiment Analysis", "🧠 Topic Modeling", "🔑 Keyword Overview", "📊 Insights"]
            )

            
            with tab1:
                st.subheader("Sentiment Overview")

                sentiment_counts = df["sentiment"].value_counts()
                sentiment_percent = (sentiment_counts / sentiment_counts.sum()) * 100

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📈 Sentiment Percentage")
                    st.bar_chart(sentiment_percent)

                with col2:
                    st.subheader("🥧 Sentiment Distribution")
                    fig, ax = plt.subplots()
                    ax.pie(
                        sentiment_counts,
                        labels=sentiment_counts.index,
                        autopct="%1.1f%%",
                        startangle=90
                    )
                    ax.axis("equal")
                    st.pyplot(fig)

                st.subheader("📄 Sample Results")
                st.dataframe(df[["review", "sentiment"]].head(10))

            
            with tab2:
                st.subheader("Extracted Topics")

                lda, topics, coherence = train_lda(
                    df["clean_text"],
                    num_topics=num_topics
                )

                st.metric("Coherence Score", round(coherence, 3))

                for topic in topics:
                    st.write(topic)

            
            with tab3:
                st.subheader("🔑 Keyword Overview")

                keywords = get_top_keywords(df["clean_text"], top_n=15)
                keyword_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])

                colA, colB = st.columns(2)

                with colA:
                    st.subheader("Top Keywords (Frequency)")
                    st.bar_chart(keyword_df.set_index("Keyword"))

                with colB:
                    st.subheader("Keyword Frequency Table")
                    st.dataframe(keyword_df)

            
            with tab4:
                colX, colY, colZ = st.columns(3)

                colX.metric("Total Reviews", len(df))
                colY.metric(
                    "Positive Reviews",
                    (df["sentiment"] == "Positive").sum()
                )
                colZ.metric(
                    "Negative Reviews",
                    (df["sentiment"] == "Negative").sum()
                )

else:
    st.info("⬅️ Upload a CSV file from the sidebar to analyze a dataset")
