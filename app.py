import streamlit as st
import pandas as pd

from src.preprocessing import clean_text
from src.sentiment_analysis import get_sentiment
from src.topic_modeling import train_lda

st.set_page_config(
    page_title="ReviewScope",
    layout="wide"
)


st.title("ReviewScope – Smart Review Analysis Platform")

st.caption("Analyze customer reviews to discover sentiment trends and hidden topics")

st.sidebar.header("⚙️ Controls")
st.sidebar.write("Follow the steps below")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV file",
    type=["csv"]
)

start_analysis = st.sidebar.button("🚀 Run Analysis")

if uploaded_file is None:
    st.info("⬅️ Upload a CSV file from the sidebar to begin")
else:
    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("CSV must contain a column named 'review'")
    else:
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        if start_analysis:
            with st.spinner("🔄 Processing text..."):
                df["clean_text"] = df["review"].apply(clean_text)
                df["sentiment"] = df["review"].apply(get_sentiment)

            st.success("✅ Text processing completed")

            tab1, tab2, tab3 = st.tabs(
                ["😊 Sentiment Analysis", "🧠 Topic Modeling", "📊 Insights"]
            )

            with tab1:
                st.subheader("Sentiment Distribution")
                st.bar_chart(df["sentiment"].value_counts())

                st.subheader("Sample Sentiment Results")
                st.dataframe(df[["review", "sentiment"]].head(10))

            with tab2:
                st.subheader("Extracted Topics")
                lda, topics, coherence = train_lda(df["clean_text"])

                st.write("**Coherence Score:**", coherence)
                for t in topics:
                    st.write(t)

            with tab3:
                st.subheader("Key Insights")
                st.metric("Total Reviews", len(df))
                st.metric("Positive Reviews", (df["sentiment"] == "Positive").sum())
                st.metric("Negative Reviews", (df["sentiment"] == "Negative").sum())
