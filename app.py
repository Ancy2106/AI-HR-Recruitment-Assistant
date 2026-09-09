import re
import json
import csv
import io

import streamlit as st
from pypdf import PdfReader

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI HR Recruitment Assistant",
    page_icon="🧑‍💼",
    layout="wide"
)

# ============================================================
# 🎨 MODERN HR RECRUITMENT DASHBOARD CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   GLOBAL
   ============================================================ */

html, body, [class*="css"] {
    font-family: Arial, Helvetica, sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(99, 102, 241, 0.08),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 10%,
            rgba(37, 99, 235, 0.07),
            transparent 25%
        ),
        #f7f8fc;
}


/* ============================================================
   HERO HEADER
   ============================================================ */

.hero {
    background: linear-gradient(
        135deg,
        #312e81,
        #4f46e5,
        #2563eb
    );

    color: white;

    padding: 32px 36px;

    border-radius: 24px;

    margin-bottom: 28px;

    box-shadow:
        0 15px 40px rgba(49, 46, 129, 0.22);
}

.hero h1 {
    margin: 0;

    font-size: 34px;

    font-weight: 800;

    letter-spacing: -0.5px;
}

.hero p {
    margin-top: 9px;

    margin-bottom: 0;

    font-size: 15px;

    opacity: 0.9;
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {
    font-size: 24px;

    font-weight: 800;

    color: #111827;

    margin-top: 12px;

    margin-bottom: 18px;
}


/* ============================================================
   INFORMATION CARDS
   ============================================================ */

.info-card {
    background: white;

    border: 1px solid #e5e7eb;

    border-radius: 18px;

    padding: 20px;

    margin-bottom: 15px;

    box-shadow:
        0 6px 20px rgba(15, 23, 42, 0.05);
}

.info-card h3 {
    margin-top: 0;

    margin-bottom: 8px;

    color: #111827;

    font-size: 18px;
}

.info-card p {
    color: #6b7280;

    line-height: 1.6;

    font-size: 14px;
}


/* ============================================================
   STREAMLIT METRIC CARDS
   ============================================================ */

[data-testid="stMetric"] {
    background: white;

    border: 1px solid #e5e7eb;

    border-radius: 18px;

    padding: 18px 20px;

    min-height: 110px;

    box-shadow:
        0 6px 20px rgba(15, 23, 42, 0.05);
}

[data-testid="stMetricLabel"] {
    color: #6b7280 !important;

    font-size: 13px !important;

    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;

    font-size: 28px !important;

    font-weight: 800 !important;
}


/* ============================================================
   MAIN SCORE
   ============================================================ */

.score-container {
    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #f5f3ff
        );

    border: 1px solid #ddd6fe;

    border-radius: 22px;

    padding: 22px;

    margin: 15px 0;

    box-shadow:
        0 8px 25px rgba(79, 70, 229, 0.08);
}


/* ============================================================
   SKILL BADGES
   ============================================================ */

.skill {
    display: inline-block;

    padding: 7px 13px;

    margin: 4px;

    border-radius: 999px;

    font-size: 13px;

    font-weight: 600;
}

.skill-match {
    background: #dcfce7;

    color: #166534;

    border: 1px solid #bbf7d0;
}

.skill-missing {
    background: #fee2e2;

    color: #991b1b;

    border: 1px solid #fecaca;
}


/* ============================================================
   AGENT CARD
   ============================================================ */

.agent-card {
    background:
        linear-gradient(
            135deg,
            #111827,
            #312e81
        );

    color: white;

    border-radius: 22px;

    padding: 28px;

    margin-bottom: 22px;

    box-shadow:
        0 12px 30px rgba(17, 24, 39, 0.18);
}

.agent-title {
    font-size: 24px;

    font-weight: 800;

    margin-bottom: 5px;
}

.agent-subtitle {
    font-size: 14px;

    opacity: 0.75;
}


/* ============================================================
   AI RESPONSE
   ============================================================ */

.ai-response {
    background: white;

    border-left: 5px solid #6366f1;

    border-radius: 16px;

    padding: 20px;

    margin-top: 16px;

    box-shadow:
        0 6px 20px rgba(15, 23, 42, 0.06);
}


/* ============================================================
   RANKING CARDS
   ============================================================ */

.rank-card {
    background: white;

    border: 1px solid #e5e7eb;

    border-radius: 18px;

    padding: 20px;

    margin: 12px 0;

    box-shadow:
        0 6px 20px rgba(15, 23, 42, 0.05);
}

.rank-number {
    font-size: 25px;

    font-weight: 800;

    color: #4f46e5;
}

.rank-name {
    font-size: 18px;

    font-weight: 750;

    color: #111827;
}

.rank-score {
    font-size: 24px;

    font-weight: 800;

    color: #16a34a;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #111827,
            #1e1b4b
        );
}

[data-testid="stSidebar"] * {
    color: #f9fafb !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    border-radius: 12px;

    border: none;

    font-weight: 700;

    min-height: 42px;

    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 7px 18px rgba(79, 70, 229, 0.15);
}


/* ============================================================
   INPUTS
   ============================================================ */

textarea,
input {
    border-radius: 12px !important;
}


/* ============================================================
   TABS
   ============================================================ */

button[data-baseweb="tab"] {
    font-weight: 700;

    font-size: 14px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #4f46e5 !important;
}


/* ============================================================
   DIVIDERS
   ============================================================ */

hr {
    border-color: #e5e7eb;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    padding: 28px;

    color: #6b7280;

    font-size: 13px;
}


/* ============================================================
   ALERT / STATUS BOXES
   ============================================================ */

[data-testid="stAlert"] {
    border-radius: 14px;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {
    border-radius: 14px;

    overflow: hidden;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    background: white;

    border-radius: 16px;

    padding: 8px;

    border: 1px solid #e5e7eb;
}


/* ============================================================
   RESPONSIVE SPACING
   ============================================================ */

.block-container {
    padding-top: 2rem;

    padding-bottom: 3rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "jd" not in st.session_state:
    st.session_state.jd = ""

if "resume" not in st.session_state:
    st.session_state.resume = ""

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "mem" not in st.session_state:
    st.session_state.mem = []


# ============================================================
# DEMO DATA
# ============================================================

JD = """We are hiring a Python Backend Developer.

Required skills:
Python, FastAPI, SQL, Git, Docker, AWS.

Preferred:
React, Machine Learning.

Education:
Bachelor's degree in Computer Science or related field.

Experience:
2+ years of software development experience.

Responsibilities:
Build REST APIs, work with SQL, Docker, AWS, and collaborate
with engineering teams.
"""


RESUME = """Alex Johnson
Software Engineer

Education:
B.Tech in Computer Science

Experience:
2.5 years as a Software Developer.

Skills:
Python, FastAPI, SQL, Git, Docker, AWS, React.

Built REST APIs, SQL applications, Docker deployments
and worked with AWS.
"""


POLICY = """HR Recruitment Policy - Demonstration

1. Candidates should be evaluated against published job requirements using job-related evidence only.

2. Recruiters should see matched skills, missing skills, experience evidence, and recommendation reasons.

3. AI output is decision support only. A qualified recruiter must review the candidate before any hiring decision.

4. Resume information should be handled only for the recruitment purpose for which it was collected.

5. Do not use protected characteristics or unrelated personal attributes in scoring or recommendations.

6. Interview questions must be role-related and must not target protected characteristics.

7. If the resume does not provide evidence, mark it unknown or missing rather than inventing evidence.

8. The final hiring decision belongs to a human recruiter.
"""


# ============================================================
# SKILLS / EDUCATION
# ============================================================

SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "angular",
    "vue",
    "fastapi",
    "django",
    "flask",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "git",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "machine learning",
    "deep learning",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "html",
    "css",
    "rest api",
    "api",
    "linux",
    "jenkins",
    "communication",
    "leadership",
    "excel",
    "power bi",
    "tableau",
    "spark"
]


DEGREES = [
    "phd",
    "doctorate",
    "master",
    "m.tech",
    "mtech",
    "mba",
    "bachelor",
    "b.tech",
    "btech",
    "b.sc",
    "bsc",
    "degree",
    "computer science",
    "engineering"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def norm(text):
    """Normalize text for matching."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def skills(text):
    """
    Extract skills without counting partial matches.

    Example:
    - FastAPI -> fastapi
    - REST API -> rest api
    - FastAPI should NOT also count as api
    """

    text = norm(text)
    found = set()

    # Longer phrases first
    ordered_skills = sorted(
        SKILLS,
        key=len,
        reverse=True
    )

    for skill in ordered_skills:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"

        if re.search(pattern, text):
            found.add(skill)

    # Avoid double-counting "api" when "fastapi" or "rest api" exists
    if "fastapi" in found:
        found.discard("api")

    if "rest api" in found:
        found.discard("api")

    return sorted(found)


def years(text):
    """Extract maximum years of experience mentioned in text."""
    values = [
        float(match.group(1))
        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
            norm(text)
        )
    ]

    return max(values) if values else 0.0


def score(resume, jd):
    """
    Explainable candidate scoring.

    Skill match: 70 points
    Experience: 20 points
    Education: 10 points
    """

    resume_skills = set(skills(resume))
    jd_skills = set(skills(jd))

    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)

    required_experience = years(jd)
    candidate_experience = years(resume)

    # Skill score
    if not jd_skills:
        skill_score = 70
    else:
        skill_score = 70 * len(matched) / len(jd_skills)

    # Experience score
    if required_experience <= 0:
        experience_score = 20
    else:
        experience_score = min(
            candidate_experience / required_experience,
            1
        ) * 20

    # Education score
    education_required = any(
        item in norm(jd)
        for item in [
            "bachelor",
            "master",
            "b.tech",
            "m.tech",
            "degree",
            "phd"
        ]
    )

    education_found = any(
        item in norm(resume)
        for item in DEGREES
    )

    if not education_required:
        education_score = 10
    elif education_found:
        education_score = 10
    else:
        education_score = 0

    total = round(
        skill_score +
        experience_score +
        education_score,
        1
    )

    if total >= 80:
        recommendation = "Strong Match"
    elif total >= 65:
        recommendation = "Review"
    else:
        recommendation = "Low Match"

    return {
        "score": total,
        "skill_score": round(skill_score, 1),
        "experience_score": round(experience_score, 1),
        "education_score": education_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "candidate_experience": candidate_experience,
        "required_experience": required_experience,
        "recommendation": recommendation
    }


def readpdf(file):
    """Extract text from uploaded PDF."""
    return "\n".join(
        (page.extract_text() or "")
        for page in PdfReader(file).pages
    ).strip()


# ============================================================
# OLLAMA
# ============================================================

def llm(model):
    """Create local Ollama chat model."""
    return ChatOllama(
        model=model,
        temperature=0.2,
        num_predict=450
    )


def models():
    """Return installed Ollama models."""
    try:
        import ollama

        result = ollama.list()

        if isinstance(result, dict):
            model_list = result.get("models", [])
        else:
            model_list = getattr(result, "models", [])

        output = []

        for item in model_list:

            if isinstance(item, dict):
                name = (
                    item.get("model")
                    or item.get("name")
                )
            else:
                name = (
                    getattr(item, "model", None)
                    or getattr(item, "name", None)
                )

            if name:
                output.append(name)

        return output

    except Exception:
        return []


# ============================================================
# LANGCHAIN RECRUITMENT AGENT
# ============================================================

def agent_run(model, question):
    """
    Run the LangChain Recruitment Agent using the currently
    analyzed candidate stored in Streamlit session state.
    """

    # --------------------------------------------------------
    # Get candidate directly BEFORE creating the agent
    # --------------------------------------------------------

    analysis = st.session_state.get("analysis")
    resume = st.session_state.get("resume", "")
    jd = st.session_state.get("jd", "")

    if analysis is None:
        return (
            "No candidate has been analyzed yet. "
            "Please run Single Candidate Screening first."
        )

    # --------------------------------------------------------
    # Candidate Analysis Tool
    # --------------------------------------------------------

    @tool
    def candidate_analysis() -> str:
        """
        Return the currently analyzed candidate's
        deterministic screening result.
        """

        return json.dumps(
            {
                "candidate_status": "available",
                "analysis": analysis,
                "job_description": jd[:6000],
                "resume": resume[:7000]
            },
            indent=2
        )

    # --------------------------------------------------------
    # Matched Skills Tool
    # --------------------------------------------------------

    @tool
    def matched_skills() -> str:
        """
        Return the candidate's matched skills.
        """

        matched = analysis.get(
            "matched_skills",
            []
        )

        if not matched:
            return "No matched skills."

        return ", ".join(matched)

    # --------------------------------------------------------
    # Missing Skills Tool
    # --------------------------------------------------------

    @tool
    def missing_skills() -> str:
        """
        Return the candidate's missing skills.
        """

        missing = analysis.get(
            "missing_skills",
            []
        )

        if not missing:
            return "No missing skills."

        return ", ".join(missing)

    # --------------------------------------------------------
    # HR Policy Tool
    # --------------------------------------------------------

    @tool
    def recruitment_policy(topic: str) -> str:
        """
        Return relevant HR recruitment policy.
        """

        words = [
            word
            for word in norm(topic).split()
            if len(word) > 3
        ]

        paragraphs = [
            paragraph.strip()
            for paragraph in POLICY.split("\n\n")
            if paragraph.strip()
        ]

        matches = []

        for paragraph in paragraphs:

            paragraph_normalized = norm(
                paragraph
            )

            if any(
                word in paragraph_normalized
                for word in words
            ):
                matches.append(paragraph)

        if matches:
            return "\n\n".join(matches[:3])

        return "\n\n".join(paragraphs[:3])

    # --------------------------------------------------------
    # Create Agent
    # --------------------------------------------------------

    system_prompt = """
You are an AI HR Recruitment Decision-Support Agent.

A candidate has ALREADY been analyzed by the application.

You have tools containing the candidate's actual:
- job description
- resume
- deterministic screening score
- matched skills
- missing skills
- experience
- education

IMPORTANT:

1. The candidate_analysis tool contains the actual candidate.
2. Always use candidate_analysis for candidate-specific questions.
3. Use matched_skills for questions about matched skills.
4. Use missing_skills for questions about missing skills.
5. Use recruitment_policy for HR policy questions.
6. Never ask the recruiter to upload the resume again.
7. Never claim that no candidate exists if candidate_analysis
   provides candidate data.
8. Do not invent candidate information.
9. Use only job-related evidence.
10. Never use protected characteristics.
11. AI is decision support only.
12. A human recruiter makes the final hiring decision.

Answer the recruiter's question directly and concisely.
"""

    try:

        agent = create_agent(
            model=llm(model),
            tools=[
                candidate_analysis,
                matched_skills,
                missing_skills,
                recruitment_policy
            ],
            system_prompt=system_prompt
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            }
        )

        return result[
            "messages"
        ][-1].content

    except Exception as error:

        return (
            f"LangChain Agent Error: {error}"
        )


# ============================================================
# HR POLICY RAG
# ============================================================

def rag(model, question):
    """
    Local HR Policy RAG using:
    OllamaEmbeddings + InMemoryVectorStore + Ollama.
    """

    policy_chunks = [
        paragraph.strip()
        for paragraph in POLICY.split("\n\n")
        if paragraph.strip()
    ]

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    store = InMemoryVectorStore.from_texts(
        policy_chunks,
        embedding=embeddings
    )

    documents = store.similarity_search(
        question,
        k=2
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
Answer the HR policy question using ONLY the
retrieved policy context.

If the policy does not specify the answer,
say that the policy does not specify it.

Do not invent HR rules.

POLICY CONTEXT:
{context}

QUESTION:
{question}
"""

    answer = llm(model).invoke(prompt).content

    return answer, context


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🧑‍💼 AI HR Recruitment Assistant</h1>
    <p>
        Intelligent recruitment decision support powered by
        LangChain, Ollama & local AI
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Local AI")

    model = st.text_input(
        "Ollama chat model",
        "qwen2.5:3b"
    )

    installed_models = models()

    if installed_models:

        st.success("Ollama connected")

        if model in installed_models:
            st.success(
                f"Model ready: {model}"
            )
        else:
            st.warning(
                f"{model} is not installed."
            )

            st.write("Installed models:")

            for installed in installed_models:
                st.code(installed)

    else:

        st.warning(
            "Ollama model not detected"
        )

    st.code(
        "ollama pull qwen2.5:3b\n"
        "ollama pull nomic-embed-text"
    )

    st.info(
        "AI is decision support. "
        "Human recruiter review is required."
    )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "📄 Single Candidate",
        "🏆 Candidate Ranking",
        "🤖 Recruitment Agent",
        "📚 HR Policy RAG",
        "🧠 Recruiter Memory",
        "ℹ️ Architecture"
    ]
)


# ============================================================
# TAB 1 - SINGLE CANDIDATE
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">📄 Candidate Screening</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # ========================================================
    # JOB DESCRIPTION
    # ========================================================

    with col1:

        st.markdown(
            """
            <div class="info-card">
                <h3>💼 Job Requirements</h3>
                <p>Enter the job description and required skills.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        jd = st.text_area(
            "Job Description",
            value=st.session_state.jd or JD,
            height=300,
            key="single_candidate_jd"
        )

    # ========================================================
    # RESUME
    # ========================================================

    with col2:

        st.markdown(
            """
            <div class="info-card">
                <h3>📄 Candidate Resume</h3>
                <p>Upload a PDF resume or paste resume text.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_resume = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"],
            key="single_resume_upload"
        )

        extracted_text = ""

        if uploaded_resume:

            try:

                extracted_text = readpdf(
                    uploaded_resume
                )

                st.success(
                    f"Extracted {len(extracted_text):,} characters"
                )

            except Exception as error:

                st.error(
                    f"PDF extraction error: {error}"
                )

        resume = st.text_area(
            "Or paste resume text",
            value=(
                extracted_text
                if extracted_text
                else (
                    st.session_state.resume
                    if st.session_state.resume
                    else RESUME
                )
            ),
            height=250,
            key="single_candidate_resume"
        )

    # ========================================================
    # ANALYZE CANDIDATE
    # ========================================================

    if st.button(
        "🔍 Analyze Candidate",
        type="primary",
        use_container_width=True,
        key="analyze_candidate_button"
    ):

        if not jd.strip():

            st.error(
                "Please provide a job description."
            )

        elif not resume.strip():

            st.error(
                "Please provide a resume."
            )

        else:

            try:

                with st.spinner(
                    "Analyzing candidate..."
                ):

                    candidate_result = score(
                        resume,
                        jd
                    )

                # Save everything to session
                st.session_state.analysis = (
                    candidate_result
                )

                st.session_state.resume = resume

                st.session_state.jd = jd

                st.success(
                    "✅ Candidate analysis completed!"
                )

            except Exception as error:

                st.error(
                    f"Candidate analysis error: {error}"
                )

    # ========================================================
    # CANDIDATE INTELLIGENCE
    # ========================================================

    if st.session_state.analysis:

        analysis = st.session_state.analysis

        st.divider()

        st.markdown(
            "## 👤 Candidate Intelligence"
        )

        # ====================================================
        # MAIN MATCH SCORE
        # ====================================================

        score_col1, score_col2, score_col3 = st.columns(
            [1, 2, 1]
        )

        with score_col2:

            st.metric(
                label="⭐ AI MATCH SCORE",
                value=f"{analysis['score']}/100"
            )

        # ====================================================
        # KPI CARDS
        # ====================================================

        k1, k2, k3, k4 = st.columns(4)

        with k1:

            st.metric(
                "⭐ Match Score",
                f"{analysis['score']}/100"
            )

        with k2:

            st.metric(
                "🟢 Matched Skills",
                len(
                    analysis["matched_skills"]
                )
            )

        with k3:

            st.metric(
                "🔴 Skill Gaps",
                len(
                    analysis["missing_skills"]
                )
            )

        with k4:

            st.metric(
                "💼 Experience",
                f"{analysis['candidate_experience']} yrs"
            )

        st.write("")

        # ====================================================
        # MATCHED SKILLS
        # ====================================================

        st.subheader(
            "🟢 Matched Skills"
        )

        if analysis["matched_skills"]:

            matched_text = "  ".join(
                f"✓ `{skill}`"
                for skill in analysis["matched_skills"]
            )

            st.markdown(
                matched_text
            )

        else:

            st.info(
                "No matched skills found."
            )

        # ====================================================
        # MISSING SKILLS
        # ====================================================

        st.subheader(
            "🔴 Skill Gaps"
        )

        if analysis["missing_skills"]:

            missing_text = "  ".join(
                f"❌ `{skill}`"
                for skill in analysis["missing_skills"]
            )

            st.markdown(
                missing_text
            )

        else:

            st.success(
                "✓ No major skill gaps"
            )

        # ====================================================
        # RECOMMENDATION
        # ====================================================

        st.subheader(
            "🎯 Recruitment Recommendation"
        )

        recommendation = (
            analysis["recommendation"]
        )

        if recommendation == "Strong Match":

            st.success(
                f"🟢 {recommendation}"
            )

        elif recommendation == "Moderate Match":

            st.warning(
                f"🟡 {recommendation}"
            )

        else:

            st.error(
                f"🔴 {recommendation}"
            )

        st.caption(
            "Recommendation is based only on "
            "job-related evidence from the resume."
        )

        # ====================================================
        # EXPLAINABLE SCORE BREAKDOWN
        # ====================================================

        st.subheader(
            "📊 Explainable Score Breakdown"
        )

        b1, b2, b3 = st.columns(3)

        with b1:

            st.metric(
                "🛠️ Skill Match",
                f"{analysis['skill_score']}/70"
            )

        with b2:

            st.metric(
                "💼 Experience",
                f"{analysis['experience_score']}/20"
            )

        with b3:

            st.metric(
                "🎓 Education",
                f"{analysis['education_score']}/10"
            )

        # ====================================================
        # AI RECRUITER TOOLS
        # ====================================================

        st.subheader(
            "✨ AI Recruiter Tools"
        )

        ai_col1, ai_col2 = st.columns(2)

        # ----------------------------------------------------
        # AI SUMMARY
        # ----------------------------------------------------

        with ai_col1:

            if st.button(
                "✨ Generate AI Summary",
                use_container_width=True,
                key="generate_ai_summary"
            ):

                prompt = f"""
Create a concise recruiter summary using ONLY
the evidence below.

Do not make a final hiring decision.

JOB DESCRIPTION:
{st.session_state.jd[:6000]}

RESUME:
{st.session_state.resume[:7000]}

DETERMINISTIC ANALYSIS:
{json.dumps(analysis)}

Mention:
- overall fit
- matched skills
- missing skills
- experience
- education
- recommendation

Keep it professional and concise.
"""

                try:

                    with st.spinner(
                        "Generating recruiter summary..."
                    ):

                        response = llm(model).invoke(
                            prompt
                        )

                    st.markdown(
                        "### 🤖 AI Recruiter Summary"
                    )

                    st.write(
                        response.content
                    )

                except Exception as error:

                    st.error(
                        f"Ollama error: {error}"
                    )

        # ----------------------------------------------------
        # INTERVIEW QUESTIONS
        # ----------------------------------------------------

        with ai_col2:

            if st.button(
                "📝 Generate Interview Questions",
                use_container_width=True,
                key="generate_interview_questions"
            ):

                prompt = f"""
Generate 6 numbered interview questions
for this candidate.

Questions must be:
- job-related
- technical or behavioral
- based on the job description
- based on the resume
- useful for validating skills

Do not ask about:
- age
- gender
- religion
- race
- disability
- marital status
- other protected characteristics

JOB:
{st.session_state.jd[:5000]}

RESUME:
{st.session_state.resume[:6000]}

ANALYSIS:
{json.dumps(analysis)}
"""

                try:

                    with st.spinner(
                        "Generating interview questions..."
                    ):

                        response = llm(model).invoke(
                            prompt
                        )

                    st.markdown(
                        "### 📝 Suggested Interview Questions"
                    )

                    st.write(
                        response.content
                    )

                except Exception as error:

                    st.error(
                        f"Ollama error: {error}"
                    )


# ============================================================
# TAB 2 — CANDIDATE RANKING
# ============================================================

with tabs[1]:

    st.markdown(
        "## 🏆 Candidate Ranking"
    )

    ranking_jd = st.text_area(
        "Job description",
        value=JD,
        key="ranking_jd",
        height=180
    )

    st.markdown(
        """
        **Enter candidates in this format:**

        Candidate A  
        Name: Alex Johnson  
        Education: B.Tech in Computer Science  
        Experience: 2.5 years  
        Skills: Python, FastAPI, SQL, Git, Docker, AWS, React  

        Candidate B  
        Name: Sam Lee  
        Education: B.Sc Computer Science  
        Experience: 1 year  
        Skills: Python, SQL, Git, JavaScript
        """
    )

    candidate_blob = st.text_area(
        "Candidates",
        value="""Candidate A
Name: Alex Johnson
Education: B.Tech in Computer Science
Experience: 2.5 years as a Software Developer
Skills: Python, FastAPI, SQL, Git, Docker, AWS, React
Built REST APIs, SQL applications, Docker deployments and worked with AWS.

Candidate B
Name: Sam Lee
Education: B.Sc Computer Science
Experience: 1 year as a Software Developer
Skills: Python, SQL, Git, JavaScript.

Candidate C
Name: Priya Shah
Education: B.Tech
Experience: 4 years as a Backend Developer
Skills: Python, FastAPI, SQL, Git, Docker, AWS, Kubernetes.""",
        height=360,
        key="candidate_blob"
    )

    # ========================================================
    # RANK CANDIDATES
    # ========================================================

    if st.button(
        "📊 Rank Candidates",
        type="primary",
        use_container_width=True,
        key="rank_candidates_button"
    ):

        if not ranking_jd.strip():

            st.error(
                "Please provide a job description."
            )

        elif not candidate_blob.strip():

            st.error(
                "Please provide candidate information."
            )

        else:

            # ------------------------------------------------
            # FIND CANDIDATE HEADINGS
            # ------------------------------------------------

            pattern = (
                r"(?im)^(Candidate\s+[A-Za-z0-9_-]+)\s*$"
            )

            matches = list(
                re.finditer(
                    pattern,
                    candidate_blob
                )
            )

            rows = []

            if not matches:

                st.error(
                    "No candidates detected. "
                    "Please use headings such as "
                    "'Candidate A', 'Candidate B', "
                    "'Candidate C'."
                )

            else:

                # ============================================
                # PARSE EACH CANDIDATE
                # ============================================

                for index, match in enumerate(matches):

                    candidate_name = (
                        match.group(1).strip()
                    )

                    start = match.end()

                    if index + 1 < len(matches):

                        end = matches[
                            index + 1
                        ].start()

                    else:

                        end = len(
                            candidate_blob
                        )

                    candidate_text = (
                        candidate_blob[
                            start:end
                        ].strip()
                    )

                    # ----------------------------------------
                    # Remove Name line
                    # ----------------------------------------

                    candidate_resume = re.sub(
                        r"(?im)^Name\s*:\s*.*$",
                        "",
                        candidate_text
                    ).strip()

                    if not candidate_resume:

                        continue

                    try:

                        candidate_result = score(
                            candidate_resume,
                            ranking_jd
                        )

                        rows.append(
                            {
                                "Candidate": candidate_name,
                                "Score": candidate_result[
                                    "score"
                                ],
                                "Recommendation":
                                    candidate_result[
                                        "recommendation"
                                ],
                                "Matched Skills":
                                    ", ".join(
                                        candidate_result[
                                            "matched_skills"
                                        ]
                                ) or "None",
                                "Missing Skills":
                                    ", ".join(
                                        candidate_result[
                                            "missing_skills"
                                        ]
                                ) or "None"
                            }
                        )

                    except Exception as error:

                        st.warning(
                            f"Could not analyze "
                            f"{candidate_name}: {error}"
                        )

                # ============================================
                # SORT RESULTS
                # ============================================

                rows.sort(
                    key=lambda row: row["Score"],
                    reverse=True
                )

                if rows:

                    st.success(
                        f"Ranked {len(rows)} candidates."
                    )

                    # ========================================
                    # RANKING TABLE
                    # ========================================

                    display_rows = []

                    for rank, row in enumerate(
                        rows,
                        start=1
                    ):

                        display_rows.append(
                            {
                                "Rank": rank,
                                **row
                            }
                        )

                    st.dataframe(
                        display_rows,
                        use_container_width=True,
                        hide_index=True
                    )

                    # ========================================
                    # TOP CANDIDATE
                    # ========================================

                    st.markdown(
                        "### 🥇 Top Candidate"
                    )

                    top = rows[0]

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.metric(
                            "Candidate",
                            top["Candidate"]
                        )

                    with c2:

                        st.metric(
                            "Score",
                            f"{top['Score']}/100"
                        )

                    with c3:

                        st.metric(
                            "Recommendation",
                            top["Recommendation"]
                        )

                    # ========================================
                    # CSV DOWNLOAD
                    # ========================================

                    csv_buffer = io.StringIO()

                    writer = csv.DictWriter(
                        csv_buffer,
                        fieldnames=[
                            "Candidate",
                            "Score",
                            "Recommendation",
                            "Matched Skills",
                            "Missing Skills"
                        ]
                    )

                    writer.writeheader()

                    writer.writerows(
                        rows
                    )

                    st.download_button(
                        "⬇️ Download Ranking CSV",
                        csv_buffer.getvalue(),
                        "candidate_ranking.csv",
                        "text/csv",
                        key="download_ranking_csv"
                    )

                else:

                    st.warning(
                        "No valid candidates were found."
                    )


# ============================================================
# TAB 3 - RECRUITMENT AGENT
# ============================================================

with tabs[2]:

    st.markdown(
        """
        <div class="agent-card">
            <div class="agent-title">
                🤖 Recruitment Intelligence Agent
            </div>
            <div class="agent-subtitle">
                Ask questions about candidate fit, skills,
                experience and recruitment policy.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.analysis:

        st.info(
            "Analyze a candidate first in "
            "Single Candidate Screening."
        )

    else:

        analysis = st.session_state.analysis

        st.success(
            f"Candidate available • "
            f"Score: {analysis['score']}/100"
        )

        st.caption(
            "The agent has access to the analyzed "
            "candidate, matched skills, missing skills "
            "and HR recruitment policy."
        )

        question = st.text_input(
            "Ask the agent",
            "Explain why this candidate should be considered for an interview."
        )

        if st.button(
            "▶ Run Recruitment Agent",
            type="primary"
        ):

            with st.spinner("Recruitment agent is thinking..."):

                answer = agent_run(model, question)

            st.markdown(
                """
                <div class="ai-response">
                    <strong>🤖 AI Recruitment Analysis</strong>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(answer)

            st.session_state.mem.append(
                (question, answer)
            )


# ============================================================
# TAB 4 - HR POLICY RAG
# ============================================================

with tabs[3]:

    st.subheader(
        "📚 HR Policy RAG"
    )

    policy_question = st.text_input(
        "Ask an HR policy question",
        "Should AI make the final hiring decision?"
    )

    if st.button(
        "🔎 Ask Policy RAG",
        type="primary"
    ):

        try:

            with st.spinner(
                "Embedding, retrieving and asking Ollama..."
            ):

                answer, context = rag(
                    model,
                    policy_question
                )

            st.markdown(
                "### Answer"
            )

            st.write(
                answer
            )

            with st.expander(
                "Retrieved policy context"
            ):

                st.write(
                    context
                )

        except Exception as error:

            st.error(
                "RAG error. Make sure the embedding model is installed:"
            )

            st.code(
                "ollama pull nomic-embed-text"
            )

            st.exception(error)

    with st.expander(
        "Demo HR Policy"
    ):

        st.write(
            POLICY
        )


# ============================================================
# TAB 5 - RECRUITER MEMORY
# ============================================================

with tabs[4]:

    st.subheader(
        "🧠 Recruiter Session Memory"
    )

    if not st.session_state.mem:

        st.info(
            "No agent interactions yet."
        )

    else:

        for index, (question, answer) in enumerate(
            st.session_state.mem,
            start=1
        ):

            st.markdown(
                f"**Interaction {index}**"
            )

            st.write(
                "Recruiter:",
                question
            )

            st.write(
                "Agent:",
                answer
            )

            st.divider()

    if st.button(
        "🗑️ Clear Memory"
    ):

        st.session_state.mem = []

        st.rerun()


# ============================================================
# TAB 6 - ARCHITECTURE
# ============================================================

with tabs[5]:

    st.subheader(
        "System Architecture"
    )

    st.code(
        """
Streamlit UI
      │
      ├── Resume + Job Description
      │          │
      │          ▼
      │   Explainable Scoring
      │          │
      │          ├── Skill Match
      │          ├── Experience
      │          └── Education
      │
      ├── Candidate Ranking
      │
      ├── LangChain Recruitment Agent
      │          │
      │          ├── Candidate Analysis Tool
      │          ├── Matched Skills Tool
      │          ├── Missing Skills Tool
      │          └── HR Policy Tool
      │
      └── HR Policy RAG
                 │
                 ├── OllamaEmbeddings
                 ├── InMemoryVectorStore
                 └── Ollama LLM

Ollama
   │
   └── Qwen2.5 3B

Human Recruiter
   │
   └── Final Hiring Decision
        """,
        language="text"
    )

    st.markdown(
        """
### Responsible AI

- Deterministic and explainable candidate scoring
- Job-related evidence only
- No protected characteristics used for scoring
- HR policy retrieval through RAG
- AI provides decision support
- Human recruiter makes the final decision
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "AI HR Recruitment Assistant • "
    "Internship Demo • Ollama + LangChain"
)
