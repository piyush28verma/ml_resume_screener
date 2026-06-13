import re

import ssl

import sys

import nltk

import joblib

import PyPDF2

import numpy as np

import pandas as pd

import streamlit as st

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns



                                                   

try:

    _create_unverified_https_context = ssl._create_unverified_context

except AttributeError:

    pass

else:

    ssl._create_default_https_context = _create_unverified_https_context



                    

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))



                                            

GLASS_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

/* Apply Outfit font globally */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif !important;
}

/* Background gradient styling */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(20, 20, 35, 1) 0%, rgba(10, 10, 20, 1) 90%) !important;
    color: #e2e8f0 !important;
}

/* Card design (Glassmorphism) */
.glass-card {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}

/* Header Text styling */
.main-title {
    font-size: 3rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4f46e5 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 8px !important;
}

.subtitle {
    font-size: 1.2rem !important;
    color: #94a3b8 !important;
    margin-bottom: 30px !important;
    font-weight: 300 !important;
}

/* Score displays */
.score-container {
    display: flex;
    align-items: center;
    justify-content: space-around;
    flex-wrap: wrap;
    margin: 20px 0;
}

.score-circle {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    background: conic-gradient(#6366f1 var(--pct), rgba(255, 255, 255, 0.05) 0%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
    position: relative;
}

.score-circle::after {
    content: "";
    position: absolute;
    width: 130px;
    height: 130px;
    border-radius: 50%;
    background: #0f101f;
}

.score-text {
    position: relative;
    z-index: 10;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
}

.score-details {
    flex: 1;
    min-width: 250px;
    padding-left: 20px;
}

.score-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #a5b4fc;
    margin-bottom: 8px;
}

.score-bar-bg {
    width: 100%;
    height: 10px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 5px;
    overflow: hidden;
    margin-top: 10px;
}

.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #a5b4fc);
    border-radius: 5px;
    width: var(--pct);
}

/* Skills Badges */
.badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

.badge {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    display: inline-block;
}

.badge-matched {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.1);
}

.badge-missing {
    background: rgba(239, 68, 68, 0.1);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.25);
}

/* Sections headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: #ffffff;
    border-bottom: 2px solid rgba(99, 102, 241, 0.2);
    padding-bottom: 6px;
    margin-bottom: 15px;
}
</style>
"""



                        

def clean_text(text):

    if not isinstance(text, str):

        return ""

    text = text.lower()

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)



                    

def extract_text_from_pdf(uploaded_file):

    try:

        reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

        return text

    except Exception as e:

        st.error(f"Error reading PDF: {e}")

        return ""



                                       

SKILLS_LIST = [

                           

    "python", "java", "c++", "javascript", "typescript", "r", "sql", "html", "css", "go", "rust", "scala", "kotlin", "swift",

                       

    "machine learning", "deep learning", "nlp", "natural language processing", "computer vision", "statistics", "pandas", "numpy", 

    "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras", "xgboost", "lightgbm", "matplotlib", "seaborn", "scipy",

                                

    "spring boot", "django", "flask", "fastapi", "react", "angular", "vue", "node", "express", "hibernate", "rest api", "rest apis", 

    "microservices", "graphql", "docker", "kubernetes", "aws", "gcp", "azure", "git", "github", "ci/cd", "system design", "agile",

    "scrum", "jira", "confluence",

                                  

    "power bi", "tableau", "excel", "spark", "hadoop", "hive", "kafka", "dbt", "airflow", "redshift", "bigquery", "snowflake",

                        

    "product roadmap", "user stories", "stakeholder management", "product backlog", "wireframing", "scrum", "kanban"

]



def analyze_skills(jd_text, resume_text):

    jd_clean = jd_text.lower()

    resume_clean = resume_text.lower()

    

                                                      

    required_skills = []

    for skill in SKILLS_LIST:

                                                                         

        pattern = r'\b' + re.escape(skill) + r'\b'

        if re.search(pattern, jd_clean):

            required_skills.append(skill)

            

                                                        

    matched = []

    missing = []

    for skill in required_skills:

        pattern = r'\b' + re.escape(skill) + r'\b'

        if re.search(pattern, resume_clean):

            matched.append(skill)

        else:

            missing.append(skill)

            

    return matched, missing



                    

st.set_page_config(

    page_title="Resume Screener & Job Matcher",

    page_icon=None,

    layout="wide",

    initial_sidebar_state="expanded",

)



                     

st.markdown(GLASS_STYLE, unsafe_allow_html=True)



                  

st.markdown('<div class="main-title">ML Resume Screener & Job Matcher</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">Screen resumes using machine learning and evaluate job-description matching with advanced analytics.</div>', unsafe_allow_html=True)



                        

st.sidebar.markdown("### Configuration")



               

import os



@st.cache_resource

def load_models_and_vectorizer():

    models = {

        "vectorizer": None,

        "svm": None,

        "xgboost": None,

        "lightgbm": None

    }

    errors = {}

    base_dir = os.path.dirname(__file__)

    

                     

    try:

        models["vectorizer"] = joblib.load(os.path.join(base_dir, "tfidf_vectorizer.pkl"))

    except Exception as e:

        errors["Vectorizer"] = str(e)

        

              

    try:

        models["svm"] = joblib.load(os.path.join(base_dir, "svm_model.pkl"))

    except Exception as e:

        errors["Support Vector Machine (SVM)"] = str(e)

        

                  

    try:

        models["xgboost"] = joblib.load(os.path.join(base_dir, "xgboost_model.pkl"))

    except Exception as e:

        errors["XGBoost Classifier"] = str(e)

        

                   

    try:

        models["lightgbm"] = joblib.load(os.path.join(base_dir, "lightgbm_model.pkl"))

    except Exception as e:

        errors["LightGBM Classifier"] = str(e)

        

    return models, errors



models, errors = load_models_and_vectorizer()

vectorizer = models["vectorizer"]

svm_model = models["svm"]

xgb_model = models["xgboost"]

lgb_model = models["lightgbm"]



                                           

if errors:

    for model_name, err_msg in errors.items():

        st.sidebar.error(f"**{model_name}** could not be loaded.\n\n*Details: {err_msg}*")



                         

available_options = []

if svm_model is not None:

    available_options.append("Support Vector Machine (SVM)")

if lgb_model is not None:

    available_options.append("LightGBM Classifier")

if xgb_model is not None:

    available_options.append("XGBoost Classifier")



if not available_options:

    st.error("No prediction models could be loaded. Please check model files and dependencies.")

    model_option = None

else:

    model_option = st.sidebar.selectbox(

        "Choose Prediction Model",

        available_options,

        help="Select the ML model trained on the resume job matching dataset to perform inference."

    )



st.sidebar.markdown("---")

st.sidebar.markdown("### Model Performance Info")

if model_option is None:

    st.sidebar.warning("No models loaded.")

elif model_option.startswith("Support Vector Machine"):

    st.sidebar.info("**SVM Accuracy**: ~53.2%\n\nFast, stable linear model that captures text frequencies well.")

elif model_option.startswith("LightGBM"):

    st.sidebar.info("**LightGBM Accuracy**: ~56.1%\n\nTop performing boosting model capturing complex non-linear combinations of keywords.")

elif model_option.startswith("XGBoost"):

    st.sidebar.info("**XGBoost Accuracy**: ~54.4%\n\nPowerful gradient boosting tree model with robust generalization capabilities.")



                                       

default_jd = """ML Engineer needed with experience in Python, Machine Learning, scikit-learn, Pandas, Statistics, Deep Learning, SQL, Git, and Jupyter Notebook."""



                           

col_jd, col_resume = st.columns([1, 1])



with col_jd:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Job Description / Requirements</div>', unsafe_allow_html=True)

    jd_input = st.text_area(

        "Paste the Job Description here:",

        value=default_jd,

        height=250,

        help="Specify the technical requirements and skills needed for the role."

    )

    st.markdown('</div>', unsafe_allow_html=True)



with col_resume:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Resume Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(

        "Upload a resume (PDF or TXT format):",

        type=["pdf", "txt"],

        help="Ensure the resume is text-readable."

    )

    

    resume_text_manual = ""

    if not uploaded_file:

        st.markdown("<p style='text-align:center; color:#64748b;'>-- OR --</p>", unsafe_allow_html=True)

        resume_text_manual = st.text_area(

            "Paste Resume Text manually:",

            height=130,

            placeholder="Paste raw resume text here if you don't have a PDF file..."

        )

    st.markdown('</div>', unsafe_allow_html=True)



                    

st.markdown("<br>", unsafe_allow_html=True)

btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1, 1.5])

with btn_col2:

    submit_btn = st.button("Screen & Match Resume", use_container_width=True)



if submit_btn:

                          

    resume_text = ""

    if uploaded_file:

        if uploaded_file.name.endswith(".pdf"):

            with st.spinner("Extracting text from PDF resume..."):

                resume_text = extract_text_from_pdf(uploaded_file)

        else:

            resume_text = extract_text_from_txt(uploaded_file)

    else:

        resume_text = resume_text_manual



    if not resume_text.strip():

        st.warning("Please upload a resume or paste resume text before screening.")

    elif not jd_input.strip():

        st.warning("Please specify the Job Description requirements.")

    elif vectorizer is None:

        st.error("TF-IDF Vectorizer is not loaded. Please make sure tfidf_vectorizer.pkl is available.")

    else:

                                   

        from sklearn.metrics.pairwise import cosine_similarity

        

        with st.spinner("Processing texts & running matching models..."):

                                                   

            resume_clean = clean_text(resume_text)

            jd_clean = clean_text(jd_input)

            

                       

            v_res = vectorizer.transform([resume_clean])

            v_jd = vectorizer.transform([jd_clean])

            

                                                        

            ml_score = 1

            if model_option.startswith("Support Vector Machine"):

                ml_score = svm_model.predict(v_res)[0]

            elif model_option.startswith("LightGBM"):

                ml_score = lgb_model.predict(v_res)[0] + 1

            else:

                ml_score = xgb_model.predict(v_res)[0] + 1

            ml_score = int(np.clip(ml_score, 1, 5))

            

                                            

            sim_score = cosine_similarity(v_jd, v_res)[0][0]

            if sim_score >= 0.7:

                cosine_score = 5

            elif sim_score >= 0.5:

                cosine_score = 4

            elif sim_score >= 0.3:

                cosine_score = 3

            elif sim_score >= 0.15:

                cosine_score = 2

            else:

                cosine_score = 1

                

                                            

            matched_skills, missing_skills = analyze_skills(jd_input, resume_text)

            total_required = len(matched_skills) + len(missing_skills)

            

            if total_required > 0:

                match_ratio = len(matched_skills) / total_required

                if match_ratio >= 0.8:

                    skill_score = 5

                elif match_ratio >= 0.6:

                    skill_score = 4

                elif match_ratio >= 0.35:

                    skill_score = 3

                elif match_ratio >= 0.15:

                    skill_score = 2

                else:

                    skill_score = 1

                

                                                                   

                score = round((ml_score * 0.30) + (skill_score * 0.50) + (cosine_score * 0.20))

            else:

                                                                                

                score = round((ml_score * 0.60) + (cosine_score * 0.40))

                

                                              

            score = int(np.clip(score, 1, 5))

            

                                                 

            labels_map = {

                1: "Very Poor Match",

                2: "Poor Match",

                3: "Fair Match",

                4: "Good Match",

                5: "Excellent Match"

            }

            colors_map = {

                1: "#ef4444",

                2: "#f97316",

                3: "#eab308",

                4: "#3b82f6",

                5: "#10b981"

            }

            score_label = labels_map[score]

            score_pct = int(score * 20)

            score_color = colors_map[score]

            

                         

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Match Results Dashboard</div>', unsafe_allow_html=True)

        

                                                                            

        score_styles = f"""
        <style>
        .score-circle { 
            background: conic-gradient({score_color} {score_pct}%, rgba(255, 255, 255, 0.05) 0%) !important;
            box-shadow: 0 0 20px {score_color}4d !important;
        } 
        .score-bar-fill { 
            background: {score_color} !important;
            width: {score_pct}% !important;
        } 
        </style>
        """

        st.markdown(score_styles, unsafe_allow_html=True)

        

                                  

        st.markdown(f"""
        <div class="score-container">
            <div class="score-circle">
                <div class="score-text">{score}/5</div>
            </div>
            <div class="score-details">
                <div class="score-title" style="color: {score_color};">{score_label}</div>
                <div style="font-size: 1.1rem; color: #94a3b8;">
                    Overall compatibility rating: <strong>{score_pct}%</strong>
                </div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill"></div>
                </div>
                <p style="margin-top: 10px; font-size: 0.95rem; color: #cbd5e1; font-weight: 300;">
                    Calculated using <strong>{model_option}</strong> by matching semantic TF-IDF components.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        

                                  

        col_m, col_mis = st.columns([1, 1])

        

        with col_m:

            st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="color: #34d399; border-color: rgba(52, 211, 153, 0.2)">Matched Skills & Keywords</div>', unsafe_allow_html=True)

            if matched_skills:

                badges_html = "".join([f'<span class="badge badge-matched">{skill}</span>' for skill in matched_skills])

                st.markdown(f'<div class="badge-container">{badges_html}</div>', unsafe_allow_html=True)

            else:

                st.markdown("<p style='color: #64748b; font-style: italic;'>No common technical keywords identified in both job requirements and resume.</p>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            

        with col_mis:

            st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="color: #f87171; border-color: rgba(248, 113, 113, 0.2)">Missing Skills & Keywords</div>', unsafe_allow_html=True)

            if missing_skills:

                badges_html = "".join([f'<span class="badge badge-missing">{skill}</span>' for skill in missing_skills])

                st.markdown(f'<div class="badge-container">{badges_html}</div>', unsafe_allow_html=True)

            else:

                st.markdown("<p style='color: #34d399; font-style: italic; font-weight: 600;'>Excellent fit! No major technical skills from the job description are missing.</p>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)



        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("Cleaned Text Preview"):

            col_jd_txt, col_res_txt = st.columns([1, 1])

            with col_jd_txt:

                st.subheader("Job Description Cleaned Text")

                st.code(clean_text(jd_input), language="text")

            with col_res_txt:

                st.subheader("Resume Cleaned Text")

                st.code(clean_text(resume_text), language="text")



                                                                 

st.markdown("<br><br>", unsafe_allow_html=True)

tab_comp, tab_pipeline = st.tabs(["Model Performance Comparison", "ML Pipeline Details"])



with tab_comp:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Classified ML Models Comparison</div>', unsafe_allow_html=True)

    

                                        

    comparison_data = {

        "Model Name": ["Logistic Regression", "Support Vector Machine", "XGBoost", "LightGBM"],

        "Accuracy": [0.5440, 0.5320, 0.5440, 0.5605],

        "Precision": [0.5401, 0.5410, 0.5369, 0.5595],

        "Recall": [0.5440, 0.5320, 0.5440, 0.5605],

        "F1-score": [0.5369, 0.5207, 0.5372, 0.5552]

    }

    comp_df = pd.DataFrame(comparison_data)

    

                          

    def highlight_max(s):

        is_max = s == s.max()

        return ['color: #10b981; font-weight: bold' if v else '' for v in is_max]

    

    st.dataframe(

        comp_df.style.format({

            "Accuracy": "{:.4f}",

            "Precision": "{:.4f}",

            "Recall": "{:.4f}",

            "F1-score": "{:.4f}"

        }).apply(highlight_max, subset=["Accuracy", "Precision", "Recall", "F1-score"]),

        use_container_width=True,

        hide_index=True

    )

    

                       

    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(10, 4))

    fig.patch.set_facecolor('#0f101f')

    ax.set_facecolor('#141423')

    

    x = np.arange(len(comp_df["Model Name"]))

    width = 0.2

    

    ax.bar(x - width*1.5, comp_df["Accuracy"], width, label='Accuracy', color='#6366f1')

    ax.bar(x - width/2, comp_df["Precision"], width, label='Precision', color='#a5b4fc')

    ax.bar(x + width/2, comp_df["Recall"], width, label='Recall', color='#3b82f6')

    ax.bar(x + width*1.5, comp_df["F1-score"], width, label='F1-score', color='#10b981')

    

    ax.set_title("Evaluation Metric Scores by Model", fontsize=12, color='#ffffff', pad=15)

    ax.set_xticks(x)

    ax.set_xticklabels(comp_df["Model Name"], color='#cbd5e1')

    ax.tick_params(colors='#64748b')

    ax.spines['bottom'].set_color('#334155')

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_color('#334155')

    ax.set_ylim(0.4, 0.6)

    ax.legend(facecolor='#0f101f', edgecolor='#334155', labelcolor='#cbd5e1')

    ax.grid(axis='y', linestyle='--', alpha=0.2)

    

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    

with tab_pipeline:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Machine Learning Architecture</div>', unsafe_allow_html=True)

    

    st.markdown("""
    ### 1. Text Preprocessing & Cleaning (NLP)
    Before feature extraction, raw resumes and job description texts are normalized:
    - **Normalization**: Converting all text to lowercase.
    - **Noise Removal**: Removing numerical values and punctuation using regular expressions.
    - **Stopwords Filtering**: Filtering out common English stopwords (using `NLTK`) that do not carry technical significance (e.g., 'the', 'is', 'at').
    
    ### 2. Feature Extraction (TF-IDF Vectorization)
    We convert the clean, preprocessed text into a numerical representation using **TF-IDF (Term Frequency - Inverse Document Frequency)**:
    - **Vocabulary**: Fixed at a maximum of **800 features** (unigrams) to prevent overfitting and capture the most relevant technical terms across the dataset.
    - **Method**: Term frequencies are normalized by document frequency to assign higher weight to rare, unique skill keywords.
    
    ### 3. Classification Models
    Three robust Machine Learning classifiers are integrated for matching prediction:
    - **Support Vector Machine (SVM)**: Fits a maximum-margin linear hyperplane to separate match score categories.
    - **LightGBM (Light Gradient Boosting Machine)**: A high-performance, leaf-wise tree growth gradient boosting model optimized for speed and accuracy.
    - **XGBoost (eXtreme Gradient Boosting)**: A robust tree boosting model using regularized learning objectives.
    """)

    st.markdown('</div>', unsafe_allow_html=True)

