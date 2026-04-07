import streamlit as st

st.write("Hello! Nenu live app lo update chesa 😎")
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI HEADER ----------------
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
🚀 AI Job Recommendation System
</h1>
<p style='text-align: center;'>
Smart Resume Analyzer with ATS Scoring
</p>
""", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)")

# ---------------- FUNCTIONS ----------------

def extract_text(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()


def extract_skills(text):
    skills_list = [
        "python", "java", "sql", "machine learning",
        "html", "css", "javascript", "react",
        "node", "tensorflow", "pytorch", "excel"
    ]

    return [skill for skill in skills_list if skill in text]


def load_jobs():
    return pd.read_csv("jobs.csv")


def recommend(resume, jobs):
    corpus = jobs['skills'].tolist() + [resume]

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(corpus)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    jobs['score'] = similarity.flatten() * 100

    return jobs.sort_values(by='score', ascending=False)


def calculate_ats_score(resume_skills, job_skills):
    job_skills_list = job_skills.lower().split()
    match = len(set(resume_skills) & set(job_skills_list))

    if len(job_skills_list) == 0:
        return 0

    return (match / len(job_skills_list)) * 100


# ---------------- MAIN ----------------

if uploaded_file:
    resume = extract_text(uploaded_file)

    st.subheader("📄 Resume Preview")
    st.text_area("", resume, height=200)

    skills = extract_skills(resume)

    # 🔥 SKILLS BADGES
    st.markdown("### 🧠 Detected Skills")

    if skills:
        skills_html = ""
        for skill in skills:
            skills_html += f'<span style="background-color:#4CAF50;color:white;padding:6px 12px;border-radius:20px;margin:5px;display:inline-block;">{skill}</span>'
        st.markdown(skills_html, unsafe_allow_html=True)
    else:
        st.write("No skills detected")

    st.markdown("---")

    jobs = load_jobs()

    if st.button("🔍 Recommend Jobs"):
        result = recommend(resume, jobs)

        # 🔥 BEST MATCH
        best_job = result.iloc[0]
        st.success(f"🎯 Best Match: {best_job['title']}")

        # 🔥 PROFILE SCORE
        st.info(f"📊 Overall Profile Strength: {best_job['score']:.2f}%")

        st.markdown("""
        <h2 style='color:#4CAF50;'>
        🔥 Top Job Matches
        </h2>
        """, unsafe_allow_html=True)

        for i, row in result.head(5).iterrows():
            ats = calculate_ats_score(skills, row['skills'])

            st.markdown(f"### 💼 {row['title']}")
            st.write(f"📊 Match Score: {row['score']:.2f}%")
            st.write(f"🎯 ATS Score: {ats:.2f}%")

            progress_value = int(max(0, min(ats, 100)))
            st.progress(progress_value)

            st.write(f"🛠 Skills Required: {row['skills']}")
            st.markdown("---")