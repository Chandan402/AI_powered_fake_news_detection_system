import streamlit as st
import pandas as pd
import importlib.util
import os

backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend.py")
spec = importlib.util.spec_from_file_location("backend_module", backend_path)
backend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_module)
analyze_news = backend_module.analyze_news
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
def create_pdf_report(result, news):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    content = []

    # REPORT HEADING

    content.append(
        Paragraph(
            "Real-Time AI-Based Fake News Detection Report",
            styles["Title"]
        )
    )

    content.append(
        Paragraph(
            "<b>NITRA Technical Campus</b>",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            "Department of Computer Science & Engineering (AI & ML)",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            "Real-Time Fake News Detection using Explainable AI (LIME), "
            "Retrieval-Augmented Generation (RAG) and Google News Verification",
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 18))

    # News
    content.append(
        Paragraph(
            "<b>News Article:</b>",
            styles["Heading3"]
        )
    )

    content.append(
    Paragraph(
        news[:250] + "...",
        styles["BodyText"]
    )
)

    content.append(Spacer(1, 12))

    from datetime import datetime

    content.append(
        Paragraph(
            f"<b>Date & Time:</b> {datetime.now().strftime('%d-%m-%Y %I:%M %p')}",
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 12))

    # Prediction
    content.append(
        Paragraph(
            "<b>MODEL PREDICTION</b>",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Machine Learning Prediction:</b> {result['ml_prediction']}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Confidence Score:</b> {result['confidence']}%",
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 12))

    # AI Reasoning
    content.append(
        Paragraph(
            "<b>AI REASONING</b>",
            styles["Heading3"]
        )
    )

    content.append(
        Paragraph(
            result["reason"],
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 12))

    # IMPORTANT WORDS (LIME)

    content.append(
        Paragraph(
            "<b>IMPORTANT WORDS (LIME)</b>",
            styles["Heading3"]
        )
    )

    # Top 5 important words
    for word, weight in result["important_words"][:5]:

        influence = "REAL" if weight > 0 else "FAKE"

        content.append(
            Paragraph(
                f"• {word} ({round(weight,4)}) → {influence}",
                styles["BodyText"]
            )
        )

    content.append(Spacer(1, 12))

    #  RAG Evidence

    content.append(
        Paragraph(
            "<b>RAG PREDICTION</b>",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Prediction:</b> {result['rag_prediction']}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Best Similarity:</b> {round(result['best_similarity']*100,2)}%",
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 12))

    content.append(
        Paragraph(
            "<b>TOP RETRIEVED EVIDENCE</b>",
            styles["Heading3"]
        )
    )

    unique_titles = []

    for doc_item in result["evidence"]:

        title = doc_item["evidence"]

        if (
            title not in unique_titles
            and title != "No readable evidence available in the dataset."
        ):
            unique_titles.append(title)

    for i, title in enumerate(unique_titles[:3], start=1):

        content.append(
            Paragraph(
                f"{i}. {title}",
                styles["BodyText"]
            )
        )

    if len(unique_titles) == 0:

        content.append(
            Paragraph(
                "No readable evidence available in the dataset.",
                styles["BodyText"]
            )
        )

    content.append(Spacer(1,12))

    # GNEWS VERIFICATION

    content.append(
        Paragraph(
            "<b>GNEWS VERIFICATION</b>",
            styles["Heading2"]
        )
    )

    status = "VERIFIED" if result["gnews_prediction"] == "FOUND" else "NOT VERIFIED"

    content.append(
        Paragraph(
            f"<b>Status:</b> {status}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Semantic Score:</b> {round(result['gnews_score']*100,2)}%",
            styles["BodyText"]
        )
    )

    if len(result["gnews_articles"]) > 0:

        content.append(
            Paragraph(
                "<b>Top Matched Article:</b>",
                styles["Heading3"]
            )
        )

        content.append(
            Paragraph(
                result["gnews_articles"][0]["title"],
                styles["BodyText"]
            )
        )

        content.append(
            Paragraph(
                result["gnews_articles"][0]["url"],
                styles["BodyText"]
            )
        )

    content.append(Spacer(1,12))

    # HYBRID DECISION

    content.append(
        Paragraph(
            "<b>HYBRID DECISION</b>",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            f"{result['hybrid_decision']}",
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 12))

    #final result

    content.append(Spacer(1, 15))

    content.append(
        Paragraph(
            "<b>FINAL VERIFIED RESULT</b>",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Prediction :</b> VERIFIED {result['final_prediction']}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Confidence :</b> {result['confidence']}%",
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 12))

    # Footer

    content.append(
        Paragraph(
            "<b>Generated by</b>",
            styles["Heading3"]
        )
    )

    content.append(
        Paragraph(
            "AI-Based Fake News Detection System",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            "NITRA Technical Campus",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            "Department of Computer Science & Engineering (AI & ML)",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            "Academic Project Report",
            styles["Italic"]
        )
    )

    doc.build(content)

    buffer.seek(0)

    return buffer

# Page Configuration
st.set_page_config(
    page_title="Fake News Detection",
    layout="wide"
)

st.markdown("""
<style>

        /*Background*/
.stApp {
    background-color: white;
}

        /*Sidebar*/
section[data-testid="stSidebar"] {
    background-color: #0F172A;
}
    /*Sidebar Text Color*/

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] p {
    font-size: 22px !important;
    font-weight: 600 !important;
}
       
         /*Button*/
.stButton > button,
.stDownloadButton > button {
    background-color: #0F172A !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
   
             background-color: #1E293B !important;
}
           
textarea {
    border-radius: 12px !important;
    border: 1px solid #D1D5DB !important;
    background-color: white !important;
}
            
        /*Remove extra spacing b/w navigation andd fake news detection */

            div[data-testid="stRadio"] {
                margin-top: -40px;
            }

    /*Effect*/
    div[data-testid="stVerticalBlock"] > div {
        border-radius: 10px;
    }

            /* Sidebar Radio Buttons */

div[role="radiogroup"] label {
    background-color: rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 12px !important;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}
div[role="radiogroup"] label:hover {
    background-color: rgba(255,255,255,0.15);
}
            
    section[data-testid="stSidebar"] {
    width: 320px !important;
}
</style>
""", unsafe_allow_html=True)

#Navigation part
st.sidebar.markdown("""
# 📰 FAKE NEWS DETECTOR""")
st.sidebar.write(" ")

page = st.sidebar.radio(
    "",
    [
        "🏠 Dashboard",
        "📰 Fake News Detection",
        "📜 History",
        "🚀 About Project"
    ]
)

if page == "📰 Fake News Detection":
    st.markdown("""
    <div style="
    background: linear-gradient(135deg,#0F172A,#1E40AF);
    padding:20px;
    border-radius:15px;
    text-align:center;
    margin-bottom:20px;
    ">

    <h1 style="
    color:white;
    font-weight:800;
    margin:0;
    font-size:32px;
    ">
    📰 REAL-TIME FAKE NEWS DETECTION USING EXPLAINABLE AI AND RETRIEVAL-AUGMENTED GENERATION (RAG)
    </h1>

    </div>
    """, unsafe_allow_html=True)
    st.write(
        "Enter a news article and the system will analyze it using Machine Learning, RAG and Explainable AI."
    )

    # News Input
    news = st.text_area(
        "Enter News Article",
        height=250
    )

    # 🔥 IMPORTANT FIX: ALWAYS DEFINE result FIRST
    result = None

    # Analyze Button
    if st.button("Analyze News") and news.strip() != "":
        with st.spinner("🔍 Analyzing News... Please wait"):
            result = analyze_news(news)

    # 🔥 SAFE CHECK
    if result is not None:
        
        # Model Prediction
        st.subheader("📊 Model Prediction")

        prediction = result["ml_prediction"]

        if prediction == "REAL":
            st.success("🟢 REAL NEWS")

        else:
            st.error("🔴 FAKE NEWS")

        st.markdown("---")

        # Confidence Score

        st.subheader("📈 Confidence Score")

        confidence = result["confidence"]

        fig = go.Figure(go.Pie(
            values=[confidence, 100 - confidence],
            hole=0.75,
            marker=dict(colors=["green", "#E5E5E5"]),
            textinfo="none"
        ))

        fig.update_layout(
            showlegend=False,
            height=300,
            width=300,
            margin=dict(t=20, b=20, l=20, r=20),
            annotations=[
                dict(
                    text=f"<b>{confidence}%</b>",
                    x=0.5,
                    y=0.5,
                    font_size=22,
                    showarrow=False
                )
            ]
        )

        col1, col2, col3 = st.columns([1,2,1])

        with col2:
            st.plotly_chart(fig)
        st.markdown("---")

            # AI Reasoning
        st.subheader("🧠 AI Reasoning")
        st.write(result["reason"])
        st.markdown("---")
        
            # Important Words
        st.subheader("🔑 Important Words (LIME Explanation)")

        real_words = []
        fake_words = []

        for word, weight in result["important_words"]:

            if weight > 0:
                real_words.append([word, round(weight, 4)])

            else:
                fake_words.append([word, round(weight, 4)])

        real_df = pd.DataFrame(
            real_words,
            columns=["Word", "Weight"]
        )

        fake_df = pd.DataFrame(
            fake_words,
            columns=["Word", "Weight"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.success("🟢 Words Supporting REAL")
            st.dataframe(
                real_df,
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.error("🔴 Words Supporting FAKE")
            st.dataframe(
                fake_df,
                use_container_width=True,
                hide_index=True
            )
        st.markdown("---")

            # Retrieved Evidence
        
        st.subheader("📑 RAG Retrieved Evidence")

        for i, doc in enumerate(result["evidence"], start=1):

            with st.container(border=True):

                st.markdown(f"### 📄 Evidence {i}")
                
                if doc.get("show_similarity", True):

                    st.write("**Title**")
                    st.write(doc["evidence"])

                    st.write("**Similarity Score**")

                    st.write(f"### {doc['similarity']*100:.2f}%")

                if doc.get("show_similarity", True):

                    if doc["label"] == 1:
                        st.success("🟢 REAL")
                    else:
                        st.error("🔴 FAKE")
                else:

                    st.write(doc["evidence"])
            st.markdown("---")

             #RAG Prediction

        st.subheader("🧩 RAG Prediction")

        if result["rag_prediction"] == "REAL":
            st.success("🟢 REAL")

        elif result["rag_prediction"] == "FAKE":
            st.error("🔴 FAKE")

        else:
            st.warning("🟡 UNKNOWN")
        st.markdown("---")

            # GNews Retrieved Evidence

        st.subheader("🌍 GNews Retrieved Evidence")

        if len(result["gnews_articles"]) > 0:

            for i, article in enumerate(result["gnews_articles"], start=1):

                with st.container(border=True):

                    st.markdown(f"### 🌐 Article {i}")

                    st.write("**Title**")
                    st.write(article["title"])

                    if "score" in article:
                        st.write("**Similarity Score**")
                        st.write(f"### {article['score']*100:.2f}%")

                    if "url" in article:
                        st.write("**URL**")
                        st.write(article["url"])

        else:

            st.info("No related articles found from Google News.")
        st.markdown("---")

            # GNews Prediction

        st.subheader("🌍 GNews Prediction")

        if result["gnews_prediction"] == "FOUND":
            st.success("🟢 VERIFIED")

        else:
            st.warning("🟡 NOT VERIFIED")
        st.markdown("---")

            #  Hybrid Decision

        st.subheader("🧠 Hybrid Decision")

        decision = result["hybrid_decision"]

        if decision == "ML + RAG + GNews":
            st.success("🟢 ML + RAG + GNews")

        elif decision == "ML + RAG":
            st.info("🔵 ML + RAG")

        elif decision == "GNews Verified":
            st.success("🌍 GNews Verified")

        elif decision == "GNews Verified (RAG unavailable)":
            st.success("🌍 GNews Verified (RAG Unavailable)")

        elif decision == "ML Only":
            st.warning("🟡 ML Only")

        elif decision == "UNCERTAIN":
            st.error("🔴 UNCERTAIN")

        else:
            st.info(decision)
        st.markdown("---")

            #final result

        st.subheader("✅ Final Verified Result")

        if result["final_prediction"] == "REAL":
            st.success("🟢 VERIFIED REAL")

        elif result["final_prediction"] == "FAKE":
            st.error("🔴 VERIFIED FAKE")

        else:
            st.warning("🟡 UNCERTAIN")
        st.markdown("---")

        # PDF downloader

        pdf_file = create_pdf_report(
            result,
            news
        )

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_file,
            file_name="Fake_News_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

     # Dashboard Page
elif page == "🏠 Dashboard":

    st.markdown("""
<div style="
background: linear-gradient(135deg,#0F172A,#1E40AF);
padding:20px;
border-radius:15px;
text-align:center;
margin-bottom:20px;
">

<h1 style="
color:white;
font-weight:800;
margin:0;
font-size:32px;
">
🏠 DASHBOARD
</h1>

</div>
""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="
        background:#F8FAFC;
        border-radius:12px;
        padding:18px;
        text-align:center;
        border:1px solid #E5E7EB;
        ">
            <div style="font-size:16px;color:#555;">
                📊 Accuracy
            </div>
            <div style="font-size:30px;font-weight:bold;color:#1E40AF;">
                98.99%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
        background:#F8FAFC;
        border-radius:12px;
        padding:18px;
        text-align:center;
        border:1px solid #E5E7EB;
        ">
            <div style="font-size:16px;color:#555;">
                📰 Dataset
            </div>
            <div style="font-size:30px;font-weight:bold;color:#1E40AF;">
                44,898
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
        background:#F8FAFC;
        border-radius:12px;
        padding:18px;
        text-align:center;
        border:1px solid #E5E7EB;
        ">
            <div style="font-size:16px;color:#555;">
                ⚡ Retrieval
            </div>
            <div style="font-size:30px;font-weight:bold;color:#1E40AF;">
                FAISS
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="
        background:#F8FAFC;
        border-radius:12px;
        padding:18px;
        text-align:center;
        border:1px solid #E5E7EB;
        ">
            <div style="font-size:16px;color:#555;">
                🤖 AI Model
            </div>
            <div style="font-size:22px;font-weight:bold;color:#1E40AF;line-height:1.3;">
                Logistic<br>Regression
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📂 Dataset Information")

    st.write("📄 Fake.csv")
    st.write("📄 True.csv")
    st.write("📰 Total News : 44,898")

    st.markdown("---")

    st.subheader("🔄 Project Workflow")

    st.info(
        "📰 User Input  →  🧹 Preprocessing  →  🤖 Machine Learning  →  🧠 LIME  →  📑 RAG  →  🌍 GNews  →  ⚙️ Hybrid Decision  →  ✅ Final Prediction"
    )

    st.subheader("⚙ Technologies Used")

    col1, col2 = st.columns(2)

    with col1:
        st.write("📝 TF-IDF Vectorization")
        st.write("🤖 Logistic Regression")
        st.write("🌍 Google News API")
        st.write("💻 Streamlit")

    with col2:
        st.write("📂 FAISS Retrieval")
        st.write("🧠 Sentence Transformers")
        st.write("💡 LIME Explainability")

        # Analysis History
elif page == "📜 History":

    st.markdown("""
<div style="
background: linear-gradient(135deg,#0F172A,#1E40AF);
padding:20px;
border-radius:15px;
text-align:center;
margin-bottom:20px;
">

<h1 style="
color:white;
font-weight:800;
margin:0;
font-size:32px;
">
📜 ANALYSIS HISTORY
</h1>

</div>
""", unsafe_allow_html=True)

    try:

        history_df = pd.read_csv("history.csv")

        # Search Box
        search = st.text_input("🔍 Search News")

        if search:
            history_df = history_df[
                history_df["News"].str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        st.write(f"📊 Total Records: {len(history_df)}")

        total_news = len(history_df)

        real_news = len(
            history_df[
                history_df["Prediction"] == "REAL"
            ]
        )

        fake_news = len(
            history_df[
                history_df["Prediction"] == "FAKE"
            ]
        )

        uncertain_news = len(
            history_df[
                history_df["Prediction"] == "UNCERTAIN"
            ]
        )
        
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "📰 Total News",
                total_news
            )

        with col2:
            st.metric(
                "🟢 Real News",
                real_news
            )

        with col3:
            st.metric(
                "🔴 Fake News",
                fake_news
            )

        with col4:
            st.metric(
                "🟡 Uncertain News",
                  uncertain_news)

        with col5:
            st.metric(
                "🎯 Accuracy",
                "98.7%"
            )

        col1, col2 = st.columns(2)

        with col1:

                #History Download Button
            with open("history.csv", "rb") as file:

                st.download_button(
                label="📥 Download History",
                data=file,
                file_name="history.csv",
                mime="text/csv",
                use_container_width=True
            )

                #Clear History Button
        with col2:

            if st.button(
                "🗑 Clear History",
                use_container_width=True
            ):

                empty_df = pd.DataFrame(
                    columns=[
                        "DateTime",
                        "News",
                        "Prediction",
                        "Confidence"
                    ]
                )

                empty_df.to_csv(
                    "history.csv",
                    index=False
                )

                st.success("History Cleared Successfully!")

                st.rerun()

        for _, row in history_df[::-1].iterrows():

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:

                    st.markdown(
                        f"### 📰 {row['News']}"
                    )

                    st.caption(f"🕒 {row['DateTime']}")

                    confidence = row.get("Confidence", "N/A")

                    st.write(
                        f"📈 Confidence: {confidence}%"
                    )

                with col2:

                    prediction = str(row["Prediction"]).upper()

                    if prediction == "REAL":
                        st.success("🟢 REAL")

                    elif prediction == "FAKE":
                        st.error("🔴 FAKE")

                    else:
                        st.warning("🟡 UNCERTAIN")

    except Exception as e:

        st.error(f"Error: {e}")

    # About Project
elif page == "🚀 About Project":

    st.markdown("""
<div style="
background: linear-gradient(135deg,#0F172A,#1E40AF);
padding:20px;
border-radius:15px;
text-align:center;
margin-bottom:20px;
">

<h1 style="
color:white;
font-weight:800;
margin:0;
font-size:32px;
">
🚀 ABOUT PROJECT
</h1>

</div>
""", unsafe_allow_html=True)

    st.subheader(
        "📰 REAL-TIME FAKE NEWS DETECTION USING EXPLAINABLE AI AND RETRIEVAL-AUGMENTED GENERATION (RAG)"
    )

    st.subheader("📰 Project Overview")

    st.info("""
    The AI-Based Fake News Detection System is designed to classify news
    articles as REAL or FAKE using Machine Learning, Retrieval-Augmented
    Generation (RAG), Google News Verification, and Explainable AI (LIME).

    The hybrid framework combines multiple verification techniques to
    improve prediction accuracy, reliability, and transparency while
    providing clear explanations for every prediction.
    """)

    st.markdown("---")

    st.subheader("🎯 Project Objectives")
    st.info("📰 Detect Fake and Real News")
    st.info("🤖 Improve Prediction Accuracy using Hybrid")
    st.info("🧠 Explain AI Decisions with LIME")
    st.info("📑 Retrieve Similar News using FAISS")
    st.info("🌍 Verify News using Google News")
    st.info("✅ Generate Reliable Final Predictions")

    st.markdown("---")

    st.subheader("⚙️ Technologies Used")
    col1, col2 = st.columns(2)
    with col1:
        st.info("🔤 **TF-IDF Vectorization**")
        st.info("🤖 **Logistic Regression**")
        st.info("🌍 **Google News API**")
        st.info("💻 **Streamlit**")
    with col2:
        st.info("📂 **FAISS Retrieval**")
        st.info("🧠 **Sentence Transformers**")
        st.info("💡 **LIME Explainability**")
        st.info("🐍 **Python**")

        st.markdown("---")
    st.subheader("🏫 Developed At")
    st.info("🏛️ NITRA Technical Campus")
    st.info("🎓 Department of Computer Science & Engineering (AI & ML)")
    st.info("📍 Ghaziabad, Uttar Pradesh")

    st.markdown("---")
    st.subheader("📌 Project Information")
    col1, col2 = st.columns(2)
    with col1:
        st.info("📂 **Project Type**\n\nMajor Project")
        st.info("🤖 **Domain**\n\nArtificial Intelligence")
    with col2:
        st.info("📰 **Category**\n\nFake News Detection")
        st.info("🚀 **Version**\n\nVersion 1.0")

        st.markdown("---")
    st.caption(
        "Developed for Academic Research | AI-Based Fake News Detection System | NITRA Technical Campus"
    )