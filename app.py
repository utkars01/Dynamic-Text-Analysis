import streamlit as st
import pandas as pd

from src.preprocessing import clean_text
from src.sentiment_analysis import get_sentiment
from src.topic_modeling import train_lda

st.set_page_config(
    page_title="ReviewScope",
    layout="wide"
)

st.title("📊 ReviewScope – Smart Review Analysis Platform")
st.caption("Analyze customer reviews using sentiment analysis and topic modeling")


st.subheader("📝 Instant Review Analysis")

user_text = st.text_area(
    "Paste a review below",
    placeholder="Example: The product quality is amazing and delivery was fast..."
)

analyze_text_btn = st.button("🔍 Analyze Text")

if analyze_text_btn and user_text.strip() != "":
    clean = clean_text(user_text)
    sentiment = get_sentiment(user_text)

    st.success(f"**Sentiment:** {sentiment}")
elif analyze_text_btn:
    st.warning("Please enter some text to analyze")

st.divider()


st.sidebar.header("⚙️ Dataset Analysis Controls")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

num_topics = st.sidebar.slider(
    "🧠 Number of Topics",
    min_value=2,
    max_value=10,
    value=5
)

run_button = st.sidebar.button("🚀 Run Dataset Analysis")


if uploaded_file is None:
    st.info("⬅️ Upload a CSV file from the sidebar to analyze a dataset")
else:
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

            tab1, tab2, tab3 = st.tabs(
                ["😊 Sentiment Analysis", "🧠 Topic Modeling", "📊 Insights"]
            )

            with tab1:
    st.subheader("Sentiment Overview")

    sentiment_counts = df["sentiment"].value_counts()
    sentiment_percent = (sentiment_counts / sentiment_counts.sum()) * 100

    colA, colB = st.columns(2)

    with colA:
        st.subheader("📈 Sentiment Percentage")
        st.bar_chart(sentiment_percent)

    with colB:
        st.subheader("🥧 Sentiment Distribution")
        import matplotlib.pyplot as plt
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
                col1, col2, col3 = st.columns(3)

                col1.metric("Total Reviews", len(df))
                col2.metric(
                    "Positive Reviews",
                    (df["sentiment"] == "Positive").sum()
                )
                col3.metric(
                    "Negative Reviews",
                    (df["sentiment"] == "Negative").sum()
                )

