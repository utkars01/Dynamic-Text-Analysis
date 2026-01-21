import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from src.preprocessing import clean_text
from src.sentiment_analysis import get_sentiment
from src.topic_modeling import train_lda


st.set_page_config(
    page_title="ReviewScope – Smart Review Analysis Platform",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7fa, #e4ecf7);
    font-family: 'Segoe UI', sans-serif;
}

.card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 22px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.08);
    animation: fadeUp 0.6s ease-in-out;
}

.kpi-card {
    background: linear-gradient(135deg, #ffffff, #f1f5fb);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.06);
    text-align: center;
}

.kpi-icon {
    font-size: 28px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
}

.kpi-label {
    font-size: 14px;
    color: #555;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


st.title("📊 ReviewScope – Smart Review Analysis Platform")
st.caption("A high-end AI web application for sentiment intelligence and text insights")


def get_top_keywords(text_series, top_n=20):
    words = " ".join(text_series).split()
    return Counter(words).most_common(top_n)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "📝 Single Text Analysis",
    "📂 Dataset Analysis",
    "🔑 Keyword Insights",
    "📊 Dashboard"
])


with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 About ReviewScope")
    st.markdown("""
    **ReviewScope** is an AI-powered text analytics platform designed to extract
    actionable insights from unstructured textual data.

    **Core Capabilities**
    - Sentiment Intelligence
    - Topic Modeling (LDA)
    - Keyword Pattern Discovery
    - Executive-level Dashboards
    """)
    st.markdown('</div>', unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Single Text Analysis")
    st.caption("Paste any text to instantly detect sentiment")

    single_text = st.text_area("Enter text")

    if single_text.strip():
        sentiment = get_sentiment(single_text)
        st.success(f"Detected Sentiment: **{sentiment}**")

    st.markdown('</div>', unsafe_allow_html=True)


with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📂 Dataset Analysis")
    st.caption("Upload a CSV file for automated sentiment analysis")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    run = st.button("🚀 Run Analysis")

    if uploaded:
        df = pd.read_csv(uploaded)

        if df.select_dtypes(include=["object"]).empty:
            st.error("No text column found in the dataset")
            st.stop()

        text_column = df.select_dtypes(include=["object"]).columns[0]

        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        if run:
            with st.spinner("Processing dataset..."):
                df["clean_text"] = df[text_column].astype(str).apply(clean_text)
                df["sentiment"] = df[text_column].astype(str).apply(get_sentiment)

            st.success("Analysis completed")

            st.subheader("😊 Sentiment Distribution")
            sentiment_counts = df["sentiment"].value_counts()
            sentiment_percent = sentiment_counts / sentiment_counts.sum() * 100

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

            # Topic modeling (internal, UI summary only)
            train_lda(df["clean_text"], num_topics=5)

            st.subheader("🧠 Topic Modeling Insights")
            st.markdown("""
            Topic modeling was applied internally to identify recurring themes.
            Results are summarized at a high level for better interpretability.
            """)

            c1, c2, c3 = st.columns(3)
            c1.metric("Themes Identified", "5")
            c2.metric("Model Used", "LDA")
            c3.metric("Analysis Scope", "Text Corpus")

    st.markdown('</div>', unsafe_allow_html=True)


with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔑 Keyword Insights")

    if "df" in locals() and "clean_text" in df.columns:
        keywords = get_top_keywords(df["clean_text"])
        kw_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])

        colA, colB = st.columns(2)
        with colA:
            st.bar_chart(kw_df.set_index("Keyword"))
        with colB:
            st.dataframe(kw_df)
    else:
        st.info("Run dataset analysis to view keyword insights")

    st.markdown('</div>', unsafe_allow_html=True)


with tab5:
    if "df" in locals() and "sentiment" in df.columns:

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Summary Dashboard")

        total = len(df)
        positive = (df["sentiment"] == "Positive").sum()
        neutral = (df["sentiment"] == "Neutral").sum()
        negative = (df["sentiment"] == "Negative").sum()

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-icon">📄</div>
                <div class="kpi-value">{}</div>
                <div class="kpi-label">Total Records</div>
            </div>
            """.format(total), unsafe_allow_html=True)

        with k2:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-icon">😊</div>
                <div class="kpi-value">{}</div>
                <div class="kpi-label">Positive</div>
            </div>
            """.format(positive), unsafe_allow_html=True)

        with k3:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-icon">😐</div>
                <div class="kpi-value">{}</div>
                <div class="kpi-label">Neutral</div>
            </div>
            """.format(neutral), unsafe_allow_html=True)

        with k4:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-icon">😞</div>
                <div class="kpi-value">{}</div>
                <div class="kpi-label">Negative</div>
            </div>
            """.format(negative), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🤖 AI Insight Summary")

        dominant = df["sentiment"].value_counts().idxmax()
        dominance_pct = round((df["sentiment"].value_counts().max() / total) * 100, 2)

        st.markdown(f"""
        - Dataset contains **{total} records**
        - **{dominant} sentiment** dominates (**{dominance_pct}%**)
        - Overall user perception is **{dominant.lower()}**
        """)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Run dataset analysis to view dashboard insights")
