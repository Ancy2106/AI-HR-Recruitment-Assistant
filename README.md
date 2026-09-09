You're right. You want **only the raw README content**, ready to copy directly into `README.md` — no explanation outside it.

# AI HR Recruitment Assistant 🧑‍💼

### Agentic AI-Powered Recruitment Decision Support System

**AI HR Recruitment Assistant** is an Agentic AI-powered recruitment assistant for resume analysis, skill matching, candidate ranking, and HR support.

---

## 📌 Project Description

Recruiters often spend significant time reviewing resumes, comparing candidate qualifications with job requirements, identifying skill gaps, and evaluating multiple candidates.

The AI HR Recruitment Assistant provides a centralized platform that combines:

* 📄 Resume analysis
* 🎯 Job description matching
* 📊 Explainable candidate scoring
* ✅ Matched skill identification
* ❌ Missing skill identification
* 🤖 AI Recruiter Agent
* 📚 HR Policy RAG
* 🏆 Candidate ranking
* 🧠 Recruiter session memory
* 📥 CSV export

The goal of the project is to make recruitment screening faster, structured, explainable, and AI-assisted while keeping the final hiring decision with the human recruiter.

---

## ✨ Features

### 📄 Resume Analysis

* Upload PDF resumes
* Paste resume text
* Extract candidate information
* Analyze resumes against job descriptions

### 🎯 Candidate Matching

The system compares candidate qualifications with job requirements and provides:

* Match score
* Matched skills
* Missing skills
* Experience
* Education
* Recruitment recommendation

### 📊 Explainable Candidate Scoring

| Category   |   Weight |
| ---------- | -------: |
| Skills     |      70% |
| Experience |      20% |
| Education  |      10% |
| **Total**  | **100%** |

### 🤖 AI Recruiter Agent

The LangChain-based AI Recruiter Agent helps recruiters answer candidate-related questions using recruitment-specific tools.

Example questions:

* What are the candidate's strongest skills?
* What skills are missing?
* Would this candidate be a strong match?
* Explain the candidate's score.

### 📚 HR Policy RAG

The system uses Retrieval-Augmented Generation to answer HR policy questions using relevant policy information.

```text
HR Policy
    ↓
Text Chunks
    ↓
Ollama Embeddings
    ↓
Vector Store
    ↓
Relevant Context
    ↓
Qwen2.5 3B
    ↓
HR Policy Answer
```

### 🏆 Candidate Ranking

Recruiters can compare multiple candidates and rank them based on the same explainable scoring methodology.

Ranking results can be exported as CSV.

### 🧠 Recruiter Memory

Recruiter-agent interactions are stored during the current session to maintain context.

### 👤 Human-in-the-Loop

The system provides recruitment decision support. The final hiring decision remains with the human recruiter.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Frontend

* Streamlit

### AI / LLM

* Ollama
* Qwen2.5 3B

### Agent Framework

* LangChain

### RAG

* OllamaEmbeddings
* nomic-embed-text
* InMemoryVectorStore

### Document Processing

* pypdf

---

## 🏗️ System Architecture

```text
                 ┌──────────────────────┐
                 │      Recruiter       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      Streamlit       │
                 │     Web Interface    │
                 └──────────┬───────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
   Resume Analysis    Candidate Ranking    HR Policy RAG
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │      LangChain       │
                 │   Recruitment Agent  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │       Ollama         │
                 │     Qwen2.5 3B       │
                 └──────────────────────┘
```

---

## 📂 Project Structure

```text
AI-HR-Recruitment-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
└── HR Policy Knowledge Base
```

---

## ⚙️ Installation and Setup

### 1. Install Python

Check the Python installation:

```bash
python --version
```

### 2. Install Ollama

Install Ollama and start the Ollama service.

Pull the required model:

```bash
ollama pull qwen2.5:3b
```

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

### 3. Clone the Repository

```bash
git clone <your-repository-url>
cd AI-HR-Recruitment-Assistant
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 🔄 Application Workflow

```text
Enter Job Description
          ↓
Upload / Paste Resume
          ↓
Resume Analysis
          ↓
Skill Matching
          ↓
Experience & Education Analysis
          ↓
Candidate Score
          ↓
Skill Gap Analysis
          ↓
Recruitment Recommendation
          ↓
AI Recruiter Agent
          ↓
HR Policy RAG
          ↓
Candidate Ranking
```

---

## 🧩 Main Modules

### 1. Candidate Screening Module

Handles resume upload, job description input, resume processing, candidate analysis, and scoring.

### 2. Skill Matching Module

Identifies matched skills and missing skills based on the job requirements.

### 3. Candidate Scoring Module

Calculates the candidate score using:

```text
Skills       → 70 points
Experience   → 20 points
Education    → 10 points
Total        → 100 points
```

### 4. AI Recruiter Agent Module

Provides interactive recruitment assistance using LangChain, Ollama, Qwen2.5 3B, and recruitment tools.

### 5. HR Policy RAG Module

Retrieves relevant HR policy information using embeddings and vector search before generating an answer.

### 6. Candidate Ranking Module

Compares multiple candidates, calculates scores, and generates a ranked list.

### 7. Recruiter Memory Module

Stores recruiter-agent interactions during the current session.

---

## 🧪 Testing

The major functionalities tested include:

* Resume PDF upload
* Resume text input
* Job description input
* Candidate analysis
* Skill matching
* Missing skill detection
* Candidate scoring
* Recruitment recommendation
* AI Recruiter Agent
* HR Policy RAG
* Candidate ranking
* CSV export
* Recruiter memory
* Streamlit interface

---

## 📈 Sample Result

```text
AI Match Score : 91.2 / 100
Matched Skills : 7
Skill Gaps     : 1
Experience     : 2.5 years
Recommendation : Strong Match
```

### Score Breakdown

```text
Skills       : 61.2 / 70
Experience   : 20.0 / 20
Education    : 10.0 / 10
Total        : 91.2 / 100
```

### Matched Skills

```text
Python
FastAPI
SQL
Git
Docker
AWS
React
```

### Missing Skill

```text
Machine Learning
```

---

## 🎯 Project Objectives

1. To automate important parts of resume screening.
2. To compare candidate skills with job requirements.
3. To identify candidate strengths and skill gaps.
4. To provide transparent candidate scoring.
5. To assist recruiters through Agentic AI.
6. To provide HR policy assistance using RAG.
7. To support candidate comparison and ranking.
8. To maintain human oversight in recruitment decisions.

---


## ⚠️ Responsible AI

This project is designed as an AI-assisted recruitment decision-support system.

The AI should not be used as an autonomous hiring authority. Recruiters should review AI-generated results and make final decisions based on appropriate organizational policies and human judgment.

---

## 👨‍💻 Project Information

**Project Name:** AI HR Recruitment Assistant

**Project Type:** Agentic AI Internship Project

**Domain:** Human Resources / Recruitment / Artificial Intelligence

**Programming Language:** Python

**Frontend:** Streamlit

**AI Framework:** LangChain

**LLM:** Qwen2.5 3B

**Model Runtime:** Ollama

**RAG:** OllamaEmbeddings + InMemoryVectorStore

---

## 📄 License

This project was developed as an academic internship project for educational purposes.

© 2026 AI HR Recruitment Assistant
