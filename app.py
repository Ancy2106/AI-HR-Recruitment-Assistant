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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import base64
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI HR Recruitment Assistant",
    page_icon="🧑‍💼",
    layout="wide"
)


HEADER_IMAGE = Path(__file__).parent / "assets" / "hr_header_bg.png"

if HEADER_IMAGE.exists():
    with open(HEADER_IMAGE, "rb") as image_file:
        HEADER_IMAGE_BASE64 = base64.b64encode(
            image_file.read()
        ).decode("utf-8")
else:
    HEADER_IMAGE_BASE64 = ""


BACKGROUND_IMAGE = Path(__file__).parent / "assets" / "hr_background.png"

if BACKGROUND_IMAGE.exists():
    with open(BACKGROUND_IMAGE, "rb") as image_file:
        BACKGROUND_IMAGE_BASE64 = base64.b64encode(
            image_file.read()
        ).decode("utf-8")
else:
    BACKGROUND_IMAGE_BASE64 = ""

# ============================================================
# 🎨 PROFESSIONAL AI HR RECRUITMENT DASHBOARD CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   DESIGN SYSTEM
   ============================================================ */
:root {
    --navy: #0b1220;
    --navy-2: #111b31;
    --indigo: #4f46e5;
    --blue: #2563eb;
    --violet: #7c3aed;
    --text: #0f172a;
    --muted: #64748b;
    --line: #dbe3ef;
    --surface: #ffffff;
    --surface-2: #f1f5f9;
}

html, body, [class*="css"] {
    font-family: Arial, Helvetica, sans-serif;
}

.stApp {
    background-image:
        linear-gradient(
           rgba(20, 35, 100, 0.20),
           rgba(235, 240, 255, 0.30)
        ),
        url("data:image/png;base64,""" + BACKGROUND_IMAGE_BASE64 + """");

    background-size: cover;

    background-position: center top;

    background-repeat: no-repeat;

    background-attachment: fixed;

    color: var(--text);
}


/* Make the main workspace feel like a designed dashboard */
.main .block-container {
    position: relative;
    z-index: 1;
}

/* Give the content workspace a subtle glass effect */
.main .block-container::before {
    content: "";
    position: fixed;
    z-index: -1;

    left: 4%;
    right: 4%;
    top: 120px;
    bottom: 0;

    pointer-events: none;

    background: rgba(255, 255, 255, 0.18);

    border-radius: 40px;

    box-shadow:
        inset 0 0 80px rgba(99, 102, 241, 0.025);
}

/* ============================================================
   HERO / PRODUCT HEADER
   ============================================================ */
.hero {
    position: relative;
    overflow: hidden;
    min-height: 205px;
    padding: 32px 38px 30px;
    margin-bottom: 24px;
    border-radius: 26px;
    color: #fff;
    background:
        radial-gradient(circle at 92% 8%, rgba(255,255,255,.18), transparent 18%),
        radial-gradient(circle at 72% 115%, rgba(124,58,237,.42), transparent 30%),
        linear-gradient(135deg, #0b1220 0%, #172554 42%, #4338ca 74%, #2563eb 100%);
    box-shadow: 0 24px 55px rgba(30,41,59,.24);
}

.hero::before {
    content: "";
    position: absolute;
    width: 320px;
    height: 320px;
    right: -125px;
    top: -155px;
    border: 1px solid rgba(255,255,255,.14);
    border-radius: 50%;
    box-shadow: 0 0 0 48px rgba(255,255,255,.035), 0 0 0 96px rgba(255,255,255,.025);
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 900px;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 11px;
    margin-bottom: 14px;
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 999px;
    background: rgba(255,255,255,.08);
    color: rgba(255,255,255,.88);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .9px;
    text-transform: uppercase;
}

.hero h1 {
    margin: 0;
    color: #fff !important;
    font-size: 38px;
    font-weight: 850;
    line-height: 1.1;
    letter-spacing: -1px;
}

.hero p {
    margin: 12px 0 0;
    max-width: 720px;
    color: rgba(255,255,255,.82);
    font-size: 15px;
    line-height: 1.6;
}

.hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 20px;
}

.hero-chip {
    padding: 7px 11px;
    border-radius: 10px;
    background: rgba(15,23,42,.34);
    border: 1px solid rgba(255,255,255,.13);
    color: rgba(255,255,255,.9);
    font-size: 12px;
    font-weight: 650;
}

/* ============================================================
   SIDEBAR / NAVIGATION
   ============================================================ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #08111f 0%, #0f172a 48%, #1e1b4b 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

.sidebar-brand {
    padding: 8px 4px 18px;
    margin-bottom: 16px;
    border-bottom: 1px solid rgba(255,255,255,.10);
}

.sidebar-brand-title {
    font-size: 17px;
    font-weight: 850;
    letter-spacing: -.2px;
}

.sidebar-brand-subtitle {
    margin-top: 4px;
    color: #94a3b8 !important;
    font-size: 12px;
    line-height: 1.45;
}

[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(255,255,255,.06);
    border-color: rgba(255,255,255,.12);
}

/* ============================================================
   TAB NAVIGATION
   ============================================================ */
[data-baseweb="tab-list"] {
    gap: 8px;
    padding: 8px;
    margin-bottom: 26px;
    border: 1px solid #dbe3ef;
    border-radius: 16px;
    background: rgba(255,255,255,.72);
    box-shadow: 0 8px 24px rgba(15,23,42,.05);
}

button[data-baseweb="tab"] {
    min-height: 42px;
    padding: 8px 14px;
    border-radius: 10px;
    color: #475569 !important;
    font-size: 13px;
    font-weight: 750;
    transition: all .18s ease;
}

button[data-baseweb="tab"]:hover {
    background: #eef2ff;
    color: #4338ca !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #fff !important;
    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        ) !important;
    box-shadow:
        0 8px 20px rgba(79,70,229,.30) !important;
}

/* ============================================================
   SECTION TITLES
   ============================================================ */
.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0 20px;
    color: #0f172a;
    font-size: 25px;
    font-weight: 850;
    letter-spacing: -.5px;
}

.section-kicker {
    margin: -11px 0 22px;
    color: #64748b;
    font-size: 13px;
    line-height: 1.55;
}

h1, h2, h3, h4 {
    color: #0f172a;
}

/* ============================================================
   SURFACE / INFO CARDS
   ============================================================ */
.info-card {
    min-height: 112px;
    padding: 20px 21px;
    margin-bottom: 14px;

    border: 1px solid rgba(148, 163, 184, 0.28);
    border-radius: 20px;

    background: rgba(255, 255, 255, 0.96);

    box-shadow:
        0 12px 30px rgba(30, 41, 59, 0.08),
        0 2px 6px rgba(30, 41, 59, 0.04);
}

.info-card h3 {
    margin: 0 0 7px;
    color: #111827;
    font-size: 17px;
    font-weight: 800;
}

.info-card p {
    margin: 0;
    color: #64748b;
    font-size: 13px;
    line-height: 1.6;
}

/* ============================================================
   STREAMLIT METRICS
   ============================================================ */
[data-testid="stMetric"] {
    min-height: 112px;
    padding: 18px 20px;
    border: 1px solid #dbe3ef;
    border-radius: 17px;
    background: #fff;
    box-shadow: 0 9px 24px rgba(15,23,42,.055);
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 750 !important;
}

[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 27px !important;
    font-weight: 850 !important;
}

/* ============================================================
   INPUTS / FORMS
   ============================================================ */
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
[data-baseweb="select"] > div {
    border-color: #d5deeb !important;
    border-radius: 12px !important;
    background: #fff !important;
}

[data-baseweb="input"]:focus-within > div,
[data-baseweb="textarea"]:focus-within > div,
[data-baseweb="select"]:focus-within > div {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.10) !important;
}

textarea, input {
    border-radius: 12px !important;
}

label[data-testid="stWidgetLabel"] p {
    color: #334155;
    font-size: 13px;
    font-weight: 700;
}

/* ============================================================
   BUTTONS
   ============================================================ */
.stButton > button,
.stDownloadButton > button {
    min-height: 43px;
    padding: 8px 18px;
    border: 1px solid #dbe3ef;
    border-radius: 12px;
    color: #1e293b;
    background: #fff;
    font-weight: 750;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    border-color: #818cf8;
    box-shadow: 0 9px 22px rgba(79,70,229,.14);
}

.stButton > button[kind="primary"] {
    border: none !important;
    color: #fff !important;
    background: linear-gradient(135deg, #4338ca, #2563eb) !important;
    box-shadow: 0 8px 20px rgba(67,56,202,.20);
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 12px 26px rgba(67,56,202,.28);
}

/* ============================================================
   SCORE / SKILLS
   ============================================================ */
.score-container {
    padding: 24px;
    margin: 16px 0;
    border: 1px solid #c7d2fe;
    border-radius: 20px;
    background: linear-gradient(135deg, #eef2ff, #f8fafc 60%, #eff6ff);
    box-shadow: 0 12px 30px rgba(79,70,229,.08);
}

.skill {
    display: inline-block;
    padding: 7px 12px;
    margin: 3px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 750;
}

.skill-match {
    color: #166534;
    background: #dcfce7;
    border: 1px solid #bbf7d0;
}

.skill-missing {
    color: #991b1b;
    background: #fee2e2;
    border: 1px solid #fecaca;
}

/* ============================================================
   AGENT / AI SURFACES
   ============================================================ */
.agent-card {
    position: relative;
    overflow: hidden;
    padding: 28px 30px;
    margin-bottom: 22px;
    border-radius: 22px;
    color: #fff;
    background: linear-gradient(135deg, #090f1d, #1e1b4b 52%, #312e81);
    box-shadow: 0 18px 38px rgba(15,23,42,.20);
}

.agent-card::after {
    content: "AI";
    position: absolute;
    right: 24px;
    bottom: -25px;
    color: rgba(255,255,255,.05);
    font-size: 110px;
    font-weight: 900;
}

.agent-title {
    position: relative;
    z-index: 1;
    margin-bottom: 6px;
    font-size: 25px;
    font-weight: 850;
}

.agent-subtitle {
    position: relative;
    z-index: 1;
    color: rgba(255,255,255,.76);
    font-size: 13px;
    line-height: 1.55;
}

.ai-response {
    padding: 20px 21px;
    margin-top: 16px;
    border: 1px solid #dbe3ef;
    border-left: 5px solid #6366f1;
    border-radius: 16px;
    background: #fff;
    box-shadow: 0 9px 24px rgba(15,23,42,.06);
}

/* ============================================================
   RANKING CARDS
   ============================================================ */
.rank-card {
    padding: 20px;
    margin: 12px 0;
    border: 1px solid #dbe3ef;
    border-radius: 18px;
    background: #fff;
    box-shadow: 0 9px 24px rgba(15,23,42,.055);
}

.rank-number {
    color: #4f46e5;
    font-size: 25px;
    font-weight: 850;
}

.rank-name {
    color: #111827;
    font-size: 18px;
    font-weight: 800;
}

.rank-score {
    color: #15803d;
    font-size: 24px;
    font-weight: 850;
}

/* ============================================================
   FILE UPLOADER / TABLES / EXPANDERS
   ============================================================ */
[data-testid="stFileUploader"] {
    padding: 10px;
    border: 1px solid #dbe3ef;
    border-radius: 16px;
    background: #fff;
    box-shadow: 0 8px 22px rgba(15,23,42,.045);
}

[data-testid="stDataFrame"] {
    overflow: hidden;
    border: 1px solid #dbe3ef;
    border-radius: 15px;
    box-shadow: 0 8px 22px rgba(15,23,42,.045);
}

[data-testid="stExpander"] {
    overflow: hidden;
    border: 1px solid #dbe3ef;
    border-radius: 14px;
    background: rgba(255,255,255,.9);
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

hr {
    border: none;
    border-top: 1px solid #dbe3ef;
    margin: 25px 0;
}

/* ============================================================
   FOOTER
   ============================================================ */
.footer {
    padding: 28px 20px;
    margin-top: 32px;
    border-top: 1px solid #dbe3ef;
    color: #64748b;
    font-size: 12px;
    text-align: center;
}

/* ============================================================
   RESPONSIVE
   ============================================================ */
@media (max-width: 900px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        min-height: auto;
        padding: 28px 25px;
        border-radius: 21px;
    }

    .hero h1 {
        font-size: 31px;
    }

    [data-baseweb="tab-list"] {
        overflow-x: auto;
    }
}

@media (max-width: 600px) {
    .block-container {
        padding-top: .8rem;
    }

    .hero {
        padding: 24px 20px;
    }

    .hero h1 {
        font-size: 26px;
        letter-spacing: -.6px;
    }

    .hero p {
        font-size: 13px;
    }

    .hero-meta {
        gap: 6px;
    }

    .hero-chip {
        font-size: 11px;
    }

    .section-title {
        font-size: 21px;
    }
}

/* ============================================================
   AI HR HEADER IMAGE
   ============================================================ */

.hero {
    position: relative !important;
    overflow: hidden !important;

    min-height: 190px !important;

    padding: 38px 42px !important;

    border-radius: 26px !important;

    margin-bottom: 30px !important;

    color: white !important;

    background-image:
        linear-gradient(
            90deg,
            rgba(2, 6, 23, 0.94) 0%,
            rgba(15, 23, 42, 0.82) 38%,
            rgba(30, 27, 75, 0.62) 68%,
            rgba(37, 99, 235, 0.48) 100%
        ),
        url("data:image/png;base64,""" + HEADER_IMAGE_BASE64 + """") !important;

    background-size: cover !important;

    background-position: center !important;

    background-repeat: no-repeat !important;

    box-shadow:
        0 22px 50px rgba(30, 27, 75, 0.28) !important;
}


/* Keep all header content above the image */

.hero-content {
    position: relative !important;
    z-index: 10 !important;
}

.hero h1 {
    position: relative !important;
    z-index: 10 !important;

    color: #ffffff !important;

    text-shadow:
        0 3px 14px rgba(0, 0, 0, 0.40) !important;
}

.hero p {
    position: relative !important;
    z-index: 10 !important;

    color: rgba(255,255,255,0.90) !important;

    text-shadow:
        0 2px 8px rgba(0, 0, 0, 0.35) !important;
}


/* Image depth overlay */

.hero::before {
    content: "" !important;

    position: absolute !important;

    inset: 0 !important;

    z-index: 1 !important;

    pointer-events: none !important;

    background:
        linear-gradient(
            90deg,
            rgba(2, 6, 23, 0.25),
            transparent 60%
        ) !important;
}


/* Soft light effect */

.hero::after {
    content: "" !important;

    position: absolute !important;

    width: 320px !important;
    height: 320px !important;

    right: -110px !important;
    top: -150px !important;

    border-radius: 50% !important;

    z-index: 2 !important;

    pointer-events: none !important;

    background:
        radial-gradient(
            circle,
            rgba(255,255,255,0.13),
            transparent 68%
        ) !important;
}


/* ============================================================
   AI HR RECRUITMENT ASSISTANT
   PREMIUM GLASS DASHBOARD UI
   Keeps existing functionality + header + background image
   ============================================================ */


/* ============================================================
   1. APP / PAGE BACKGROUND
   ============================================================ */

[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}

.main .block-container {
    max-width: 1380px !important;

    padding-top: 28px !important;
    padding-bottom: 70px !important;
    padding-left: 32px !important;
    padding-right: 32px !important;
}


/* ============================================================
   2. SOFT DARK OVERLAY
   Makes the background image visible but keeps text readable
   ============================================================ */

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;

    inset: 0;

    pointer-events: none;
    z-index: 0;

    background:
        linear-gradient(
            180deg,
            rgba(5, 15, 45, 0.12) 0%,
            rgba(15, 23, 60, 0.16) 45%,
            rgba(240, 244, 255, 0.25) 100%
        );
}


/* Keep actual Streamlit content above overlay */

.main,
.block-container {
    position: relative;
    z-index: 1;
}


/* ============================================================
   3. HEADER
   Your existing .hero remains untouched
   ============================================================ */

.hero {
    position: relative !important;
    overflow: hidden !important;

    border-radius: 24px !important;

    border: 1px solid rgba(255,255,255,0.35) !important;

    box-shadow:
        0 25px 60px rgba(5, 15, 50, 0.35),
        0 0 45px rgba(79,70,229,0.18) !important;
}

.hero h1 {
    font-weight: 850 !important;
    letter-spacing: -1px !important;
}

.hero p {
    opacity: 0.95 !important;
}


/* ============================================================
   4. TABS — FLOATING GLASS NAVIGATION
   ============================================================ */

.stTabs {
    margin-top: 22px !important;
}


/* Main navigation container */

.stTabs [data-baseweb="tab-list"] {
    display: flex !important;
    align-items: center !important;

    gap: 8px !important;

    padding: 8px !important;

    border-radius: 20px !important;

    background:
        rgba(255,255,255,0.72) !important;

    border:
        1px solid rgba(255,255,255,0.75) !important;

    box-shadow:
        0 18px 45px rgba(20,35,80,0.16),
        inset 0 1px 0 rgba(255,255,255,0.9) !important;

    backdrop-filter: blur(20px) saturate(150%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(150%) !important;
}


/* Individual tabs */

.stTabs [data-baseweb="tab"] {
    min-height: 44px !important;

    padding:
        0 18px !important;

    border-radius: 14px !important;

    border: none !important;

    background: transparent !important;

    color: #172554 !important;

    font-weight: 700 !important;

    transition:
        all 0.2s ease !important;
}


/* Hover */

.stTabs [data-baseweb="tab"]:hover {
    background:
        rgba(99,102,241,0.10) !important;

    transform:
        translateY(-1px);
}


/* Active tab */

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: white !important;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        ) !important;

    box-shadow:
        0 8px 22px rgba(79,70,229,0.35) !important;
}


/* Remove default Streamlit underline */

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}


/* ============================================================
   5. TAB CONTENT — GLASS WORKSPACE
   ============================================================ */

.stTabs [data-baseweb="tab-panel"] {
    margin-top: 20px !important;

    padding: 26px !important;

    border-radius: 26px !important;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.74),
            rgba(245,248,255,0.58)
        ) !important;

    border:
        1px solid rgba(255,255,255,0.72) !important;

    box-shadow:
        0 25px 70px rgba(20,35,80,0.14),
        inset 0 1px 0 rgba(255,255,255,0.9) !important;

    backdrop-filter:
        blur(22px) saturate(140%) !important;

    -webkit-backdrop-filter:
        blur(22px) saturate(140%) !important;
}


/* ============================================================
   6. SECTION TITLES
   ============================================================ */

h1, h2, h3 {
    color: #101a3a !important;
}

h2 {
    font-weight: 850 !important;
    letter-spacing: -0.7px !important;
}

h3 {
    font-weight: 800 !important;
}


/* ============================================================
   7. YOUR EXISTING INFO CARDS
   ============================================================ */

.info-card {
    position: relative !important;

    min-height: 112px !important;

    padding: 23px 25px !important;

    border-radius: 20px !important;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.94),
            rgba(238,243,255,0.82)
        ) !important;

    border:
        1px solid rgba(255,255,255,0.95) !important;

    box-shadow:
        0 16px 38px rgba(20,30,80,0.18),
        inset 0 1px 0 rgba(255,255,255,0.95) !important;

    backdrop-filter:
        blur(18px) !important;

    -webkit-backdrop-filter:
        blur(18px) !important;
}

.info-card:hover {
    transform: translateY(-3px) !important;

    box-shadow:
        0 22px 48px rgba(20,35,80,0.18),
        0 0 25px rgba(99,102,241,0.08) !important;
}


/* ============================================================
   8. STREAMLIT COLUMNS
   Makes the two-column layout feel intentional
   ============================================================ */

[data-testid="column"] {
    position: relative !important;
}


/* ============================================================
   9. TEXT AREAS
   ============================================================ */

.stTextArea textarea {
    border-radius: 16px !important;

    border:
        1px solid rgba(148,163,184,0.20) !important;

    background:
        rgba(248,250,255,0.78) !important;

    color:
        #172554 !important;

    box-shadow:
        inset 0 2px 8px rgba(20,35,80,0.035) !important;

    backdrop-filter:
        blur(10px) !important;
}

.stTextArea textarea:focus {
    border-color:
        #6366f1 !important;

    box-shadow:
        0 0 0 3px rgba(99,102,241,0.13),
        0 8px 20px rgba(79,70,229,0.08) !important;
}


/* ============================================================
   10. FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    padding: 14px !important;

    border-radius: 19px !important;

    background:
        rgba(255,255,255,0.72) !important;

    border:
        1px solid rgba(255,255,255,0.90) !important;

    box-shadow:
        0 14px 32px rgba(20,35,80,0.10) !important;

    backdrop-filter:
        blur(16px) !important;
}

[data-testid="stFileUploaderDropzone"] {
    border-radius: 15px !important;

    border:
        1px dashed rgba(79,70,229,0.35) !important;

    background:
        rgba(238,242,255,0.62) !important;
}


/* ============================================================
   11. BUTTONS
   ============================================================ */

.stButton > button {
    min-height: 45px !important;

    padding:
        0 23px !important;

    border:
        none !important;

    border-radius:
        14px !important;

    color:
        white !important;

    font-weight:
        750 !important;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        ) !important;

    box-shadow:
        0 9px 24px rgba(79,70,229,0.28) !important;

    transition:
        all 0.2s ease !important;
}

.stButton > button:hover {
    transform:
        translateY(-2px) !important;

    box-shadow:
        0 15px 32px rgba(79,70,229,0.38) !important;
}


/* ============================================================
   12. METRIC CARDS
   ============================================================ */

[data-testid="stMetric"] {
    padding: 20px !important;

    border-radius: 19px !important;

    background:
        rgba(255,255,255,0.78) !important;

    border:
        1px solid rgba(255,255,255,0.90) !important;

    box-shadow:
        0 15px 35px rgba(20,35,80,0.11) !important;

    backdrop-filter:
        blur(16px) !important;
}

[data-testid="stMetricLabel"] {
    color:
        #64748b !important;

    font-weight:
        650 !important;
}

[data-testid="stMetricValue"] {
    color:
        #4338ca !important;

    font-weight:
        850 !important;
}


/* ============================================================
   13. RANKING CARDS
   ============================================================ */

.rank-card {
    padding: 22px !important;

    margin: 15px 0 !important;

    border-radius: 20px !important;

    background:
        rgba(255,255,255,0.80) !important;

    border:
        1px solid rgba(255,255,255,0.90) !important;

    box-shadow:
        0 16px 38px rgba(20,35,80,0.11) !important;

    backdrop-filter:
        blur(16px) !important;
}


/* ============================================================
   14. EXPANDERS
   ============================================================ */

[data-testid="stExpander"] {
    border-radius:
        18px !important;

    border:
        1px solid rgba(255,255,255,0.82) !important;

    background:
        rgba(255,255,255,0.72) !important;

    box-shadow:
        0 12px 30px rgba(20,35,80,0.08) !important;

    backdrop-filter:
        blur(14px) !important;
}


/* ============================================================
   15. DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {
    border-radius:
        18px !important;

    overflow:
        hidden !important;

    border:
        1px solid rgba(255,255,255,0.75) !important;

    box-shadow:
        0 15px 35px rgba(20,35,80,0.11) !important;
}


/* ============================================================
   16. ALERTS
   ============================================================ */

[data-testid="stAlert"] {
    border-radius:
        16px !important;

    border:
        1px solid rgba(99,102,241,0.16) !important;

    box-shadow:
        0 10px 25px rgba(20,35,80,0.07) !important;

    backdrop-filter:
        blur(12px) !important;
}


/* ============================================================
   17. DOWNLOAD BUTTON
   ============================================================ */

.stDownloadButton > button {
    border-radius:
        13px !important;

    border:
        1px solid rgba(79,70,229,0.25) !important;

    background:
        rgba(255,255,255,0.82) !important;

    color:
        #4338ca !important;

    font-weight:
        700 !important;
}

.stDownloadButton > button:hover {
    background:
        rgba(238,242,255,0.95) !important;

    border-color:
        #6366f1 !important;
}


/* ============================================================
   18. LABELS
   ============================================================ */

label {
    color:
        #172554 !important;

    font-weight:
        700 !important;
}


/* ============================================================
   19. DIVIDERS
   ============================================================ */

hr {
    border:
        none !important;

    height:
        1px !important;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(99,102,241,0.30),
            transparent
        ) !important;

    margin:
        24px 0 !important;
}


/* ============================================================
   20. SCROLLBAR
   ============================================================ */

::-webkit-scrollbar {
    width: 9px;
}

::-webkit-scrollbar-track {
    background:
        rgba(226,232,240,0.35);
}

::-webkit-scrollbar-thumb {
    border-radius:
        10px;

    background:
        linear-gradient(
            #6366f1,
            #7c3aed
        );
}


/* ============================================================
   21. MOBILE
   ============================================================ */

@media (max-width: 900px) {

    .main .block-container {
        padding-left:
            15px !important;

        padding-right:
            15px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;

        padding: 8px !important;

        border-radius: 18px !important;

        background: rgba(255, 255, 255, 0.82) !important;

        border: 1px solid rgba(255, 255, 255, 0.95) !important;

        box-shadow:
            0 10px 30px rgba(20, 30, 80, 0.18),
            inset 0 1px 0 rgba(255,255,255,0.95) !important;

        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
    }

    .stTabs [data-baseweb="tab"] {
        padding:
            0 12px !important;

        font-size:
            13px !important;
    }

    /* ============================================================
    GLASS MAIN WORKSPACE
    ============================================================ */

    .stTabs [data-baseweb="tab-panel"] {
        margin-top: 18px !important;

        padding: 28px !important;

        border-radius: 26px !important;

        background:
            linear-gradient(
                135deg,
                rgba(35, 65, 150, 0.34),
                rgba(255, 255, 255, 0.12)
            ) !important;

        border:
            1px solid rgba(255,255,255,0.48) !important;

        box-shadow:
            0 20px 55px rgba(15, 30, 80, 0.20),
            inset 0 1px 0 rgba(255,255,255,0.65) !important;

        backdrop-filter:
            blur(18px) saturate(135%) !important;

        -webkit-backdrop-filter:
            blur(18px) saturate(135%) !important;
    }

    .hero {
        border-radius:
            20px !important;
    }
}


/* ============================================================
   PREMIUM AI HR DASHBOARD
   FINAL GLASS UI
   ============================================================ */


/* ------------------------------------------------------------
   1. MAIN CONTENT
   ------------------------------------------------------------ */

.block-container {
    max-width: 1380px !important;

    padding-top: 5.5rem !important;
    padding-bottom: 4rem !important;
}


/* ------------------------------------------------------------
   2. BACKGROUND
   ------------------------------------------------------------ */

.stApp {
    background-color: transparent !important;
}


/* REMOVE DARK SIDE OVERLAY */

.stApp::before {
    display: none !important;
}


/* Keep content above background */

.stApp > div {
    position: relative !important;
    z-index: 1 !important;
}


/* ------------------------------------------------------------
   3. HEADER
   ------------------------------------------------------------ */

.hero {
    position: relative !important;

    overflow: hidden !important;

    border-radius: 24px !important;

    box-shadow:
        0 22px 55px rgba(15, 23, 60, 0.22) !important;
}

.hero h1 {
    font-weight: 850 !important;

    letter-spacing: -1px !important;
}

.hero p {
    opacity: 0.94 !important;
}


/* ------------------------------------------------------------
   4. TAB NAVIGATION
   ------------------------------------------------------------ */

.stTabs {
    position: relative !important;

    z-index: 10 !important;

    margin-top: 18px !important;

    margin-bottom: 18px !important;
}


/* GLASS TAB BAR */

.stTabs [data-baseweb="tab-list"] {
    display: flex !important;

    align-items: center !important;

    gap: 7px !important;

    padding: 8px !important;

    border-radius: 20px !important;

    background: rgba(205, 220, 255, 0.82) !important;

    border:
        1px solid rgba(255, 255, 255, 0.96) !important;

    box-shadow:
        0 12px 32px rgba(30, 41, 90, 0.16) !important;

    backdrop-filter: blur(20px) !important;

    -webkit-backdrop-filter: blur(20px) !important;
}


/* INDIVIDUAL TABS */

.stTabs [data-baseweb="tab"] {
    min-height: 42px !important;

    padding: 0 17px !important;

    border-radius: 13px !important;

    border: none !important;

    background: transparent !important;

    color: #172554 !important;

    font-weight: 700 !important;

    transition:
        background 0.2s ease,
        color 0.2s ease,
        transform 0.2s ease !important;
}


/* TAB HOVER */

.stTabs [data-baseweb="tab"]:hover {
    background:
        rgba(79, 70, 229, 0.10) !important;

    color: #312e81 !important;
}


/* ACTIVE TAB */

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        ) !important;

    box-shadow:
        0 7px 18px rgba(79, 70, 229, 0.30) !important;
}


/* ACTIVE TAB TEXT */

.stTabs [data-baseweb="tab"][aria-selected="true"] * {
    color: #ffffff !important;
}


/* REMOVE DEFAULT STREAMLIT LINE */

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}


/* ------------------------------------------------------------
   5. MAIN GLASS WORKSPACE
   ------------------------------------------------------------ */

.stTabs > div:nth-child(2) {
    position: relative !important;

    padding: 18px !important;

    margin-top: 8px !important;

    border-radius: 24px !important;

    background:
        linear-gradient(
            135deg,
            rgba(65, 110, 205, 0.32),
            rgba(91, 70, 190, 0.24)
        ) !important;

    border:
        1px solid rgba(255, 255, 255, 0.38) !important;

    box-shadow:
        0 20px 48px rgba(20, 35, 90, 0.14) !important;

    backdrop-filter: blur(16px) !important;

    -webkit-backdrop-filter: blur(16px) !important;
}


/* ------------------------------------------------------------
   6. SECTION HEADINGS
   ------------------------------------------------------------ */

h2 {
    color: #101a3a !important;

    font-weight: 850 !important;

    letter-spacing: -0.5px !important;
}

h3 {
    color: #172554 !important;

    font-weight: 800 !important;
}


/* ------------------------------------------------------------
   7. INFORMATION CARDS
   ------------------------------------------------------------ */

.info-card {
    position: relative !important;

    min-height: 112px !important;

    padding: 23px 25px !important;

    margin-bottom: 16px !important;

    border-radius: 21px !important;

    border:
        1px solid rgba(255, 255, 255, 0.97) !important;

    background:
        rgba(255, 255, 255, 0.94) !important;

    box-shadow:
        0 15px 35px rgba(30, 41, 59, 0.14) !important;

    backdrop-filter: blur(18px) !important;

    -webkit-backdrop-filter: blur(18px) !important;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease !important;
}


.info-card:hover {
    transform: translateY(-3px) !important;

    box-shadow:
        0 20px 42px rgba(30, 41, 59, 0.18) !important;
}


/* CARD TITLE */

.info-card h3 {
    color: #172554 !important;

    font-size: 18px !important;

    font-weight: 800 !important;
}


/* CARD DESCRIPTION */

.info-card p {
    color: #60708f !important;

    line-height: 1.55 !important;
}


/* ------------------------------------------------------------
   8. TEXT AREAS
   ------------------------------------------------------------ */

.stTextArea textarea,
.stTextInput input {
    border-radius: 15px !important;

    border:
        1px solid rgba(148, 163, 184, 0.20) !important;

    background:
        rgba(248, 250, 255, 0.94) !important;

    color: #172554 !important;

    box-shadow:
        inset 0 1px 4px rgba(30, 41, 59, 0.035) !important;
}


.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color:
        #6366f1 !important;

    box-shadow:
        0 0 0 3px rgba(99, 102, 241, 0.12) !important;
}


/* ------------------------------------------------------------
   9. FILE UPLOADER
   ------------------------------------------------------------ */

[data-testid="stFileUploader"] {
    padding: 12px !important;

    border-radius: 18px !important;

    background:
        rgba(255, 255, 255, 0.94) !important;

    border:
        1px solid rgba(255, 255, 255, 0.97) !important;

    box-shadow:
        0 12px 30px rgba(30, 41, 59, 0.12) !important;

    backdrop-filter: blur(15px) !important;

    -webkit-backdrop-filter: blur(15px) !important;
}


[data-testid="stFileUploaderDropzone"] {
    border-radius: 14px !important;

    border:
        1px dashed rgba(79, 70, 229, 0.30) !important;

    background:
        rgba(238, 242, 255, 0.68) !important;
}


/* ------------------------------------------------------------
   10. BUTTONS
   ------------------------------------------------------------ */

.stButton > button {
    min-height: 44px !important;

    padding: 0 22px !important;

    border: none !important;

    border-radius: 13px !important;

    color: #ffffff !important;

    font-weight: 750 !important;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        ) !important;

    box-shadow:
        0 8px 20px rgba(79, 70, 229, 0.25) !important;

    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease !important;
}


.stButton > button:hover {
    transform: translateY(-2px) !important;

    box-shadow:
        0 13px 28px rgba(79, 70, 229, 0.34) !important;
}


/* ------------------------------------------------------------
   11. METRIC CARDS
   ------------------------------------------------------------ */

[data-testid="stMetric"] {
    padding: 18px !important;

    border-radius: 18px !important;

    background:
        rgba(255, 255, 255, 0.93) !important;

    border:
        1px solid rgba(255, 255, 255, 0.97) !important;

    box-shadow:
        0 12px 30px rgba(30, 41, 59, 0.10) !important;

    backdrop-filter: blur(14px) !important;
}


[data-testid="stMetricLabel"] {
    color: #64748b !important;

    font-weight: 650 !important;
}


[data-testid="stMetricValue"] {
    color: #4338ca !important;

    font-weight: 850 !important;
}


/* ------------------------------------------------------------
   12. RANKING CARDS
   ------------------------------------------------------------ */

.rank-card {
    padding: 21px !important;

    margin: 14px 0 !important;

    border-radius: 19px !important;

    border:
        1px solid rgba(255, 255, 255, 0.96) !important;

    background:
        rgba(255, 255, 255, 0.93) !important;

    box-shadow:
        0 14px 32px rgba(30, 41, 59, 0.10) !important;

    backdrop-filter: blur(14px) !important;
}


/* ------------------------------------------------------------
   13. DATAFRAME
   ------------------------------------------------------------ */

[data-testid="stDataFrame"] {
    border-radius: 17px !important;

    overflow: hidden !important;

    border:
        1px solid rgba(255, 255, 255, 0.80) !important;

    box-shadow:
        0 12px 30px rgba(30, 41, 59, 0.10) !important;
}


/* ------------------------------------------------------------
   14. EXPANDERS
   ------------------------------------------------------------ */

[data-testid="stExpander"] {
    border-radius: 17px !important;

    background:
        rgba(255, 255, 255, 0.91) !important;

    border:
        1px solid rgba(255, 255, 255, 0.96) !important;

    box-shadow:
        0 10px 25px rgba(30, 41, 59, 0.07) !important;
}


/* ------------------------------------------------------------
   15. LABELS
   ------------------------------------------------------------ */

label {
    color: #172554 !important;

    font-weight: 700 !important;
}


/* ------------------------------------------------------------
   16. ALERTS
   ------------------------------------------------------------ */

[data-testid="stAlert"] {
    border-radius: 15px !important;

    box-shadow:
        0 8px 22px rgba(30, 41, 59, 0.06) !important;
}


/* ------------------------------------------------------------
   17. DOWNLOAD BUTTON
   ------------------------------------------------------------ */

.stDownloadButton > button {
    border-radius: 12px !important;

    border:
        1px solid rgba(79, 70, 229, 0.25) !important;

    background:
        rgba(255, 255, 255, 0.92) !important;

    color: #4338ca !important;

    font-weight: 700 !important;
}


.stDownloadButton > button:hover {
    background:
        #eef2ff !important;

    border-color:
        #6366f1 !important;
}


/* ------------------------------------------------------------
   18. DIVIDERS
   ------------------------------------------------------------ */

hr {
    border: none !important;

    height: 1px !important;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(99, 102, 241, 0.28),
            transparent
        ) !important;

    margin: 22px 0 !important;
}


/* ------------------------------------------------------------
   19. MOBILE
   ------------------------------------------------------------ */

@media (max-width: 900px) {

    .block-container {
        padding-left: 1rem !important;

        padding-right: 1rem !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0 12px !important;

        font-size: 13px !important;
    }

    .stTabs > div:nth-child(2) {
        padding: 10px !important;

        border-radius: 18px !important;
    }
}


/* ============================================================
   SINGLE CANDIDATE TAB - SPECIFIC GLASS PANEL
   ============================================================ */

.stTabs > div:nth-child(2) > div:nth-child(1) {
    background: rgba(255, 255, 255, 0.85) !important;
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.28) !important;
    box-shadow: none !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
}


button[kind="primary"] {
    border-radius: 12px !important;
}

div[data-testid="stVerticalBlock"]:has(
    div[data-testid="stHeadingWithActionElements"]
):has(
    [data-testid="stMetric"]
) {
    background: rgba(255, 255, 255, 0.85) !important;
    border-radius: 18px !important;
    padding: 20px !important;
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
    """Create Gemini when configured, otherwise use local Ollama."""

    import os

    gemini_key = os.getenv("GEMINI_API_KEY")

    if gemini_key:
        return ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0.2,
            max_output_tokens=450,
            google_api_key=gemini_key
        )

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

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
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

    # ========================================================
    # OPTIONAL — UPLOAD UP TO 5 RESUMES
    # ========================================================

    uploaded_resumes = st.file_uploader(
        "📄 Upload up to 5 candidate resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key="ranking_resume_uploads"
    )

    if len(uploaded_resumes) > 5:

        st.error(
            "Please upload a maximum of 5 resumes."
        )

        uploaded_resumes = uploaded_resumes[:5]

    # --------------------------------------------------------
    # Convert uploaded PDFs into the EXISTING candidate format
    # --------------------------------------------------------

    uploaded_candidate_text = []

    if uploaded_resumes:

        for index, resume_file in enumerate(
            uploaded_resumes,
            start=1
        ):

            try:

                resume_text = readpdf(
                    resume_file
                )

                if not resume_text.strip():

                    st.warning(
                        f"Could not extract text from "
                        f"{resume_file.name}."
                    )

                    continue

                # --------------------------------------------
                # IMPORTANT:
                # Create headings that match the existing
                # Candidate A / Candidate B parser.
                # --------------------------------------------

                candidate_label = (
                    f"Candidate {chr(64 + index)}"
                )

                uploaded_candidate_text.append(
                    candidate_label
                    + "\n"
                    + "Name: "
                    + resume_file.name.rsplit(".", 1)[0]
                    + "\n"
                    + resume_text.replace(
                        "\r\n",
                        "\n"
                    ).replace(
                        "\r",
                        "\n"
                    ).strip()
                )

            except Exception as error:

                st.warning(
                    f"Could not read "
                    f"{resume_file.name}: {error}"
                )

        if uploaded_candidate_text:

            st.success(
                f"Loaded "
                f"{len(uploaded_candidate_text)} "
                f"candidate resume(s) from PDF."
            )

            st.caption(
                "The uploaded resumes will be analyzed "
                "using the existing candidate ranking system."
            )

        else:

            st.warning(
                "No valid resume text was extracted "
                "from the uploaded PDFs."
            )

    # ========================================================
    # MANUAL CANDIDATE INPUT
    # ========================================================

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
    # IMPORTANT FIX
    # If PDFs were uploaded, use their extracted text
    # instead of the manual candidate text.
    #
    # This MUST be BEFORE the Rank Candidates button.
    # ========================================================

    if uploaded_resumes and uploaded_candidate_text:

        candidate_blob = "\n\n".join(
            uploaded_candidate_text
        )

        st.info(
            f"Using {len(uploaded_candidate_text)} "
            f"uploaded PDF resume(s) for ranking."
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

                    # ----------------------------------------
                    # Extract Education
                    # ----------------------------------------

                    education_match = re.search(
                        r"(?im)^Education\s*:\s*(.+)$",
                        candidate_text
                    )

                    education = (
                        education_match.group(1).strip()
                        if education_match
                        else "Not provided"
                    )

                    # ----------------------------------------
                    # Extract Experience
                    # ----------------------------------------

                    experience_match = re.search(
                        r"(?im)^Experience\s*:\s*(.+)$",
                        candidate_text
                    )

                    experience = (
                        experience_match.group(1).strip()
                        if experience_match
                        else "Not provided"
                    )

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

                                "Education": education,

                                "Experience": experience,

                                "Score": candidate_result[
                                    "score"
                                ],

                                "Skill Score":
                                f"{candidate_result['skill_score']}/70",

                                "Experience Score":
                                    f"{candidate_result['experience_score']}/20",

                                "Education Score":
                                    f"{candidate_result['education_score']}/10",

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
                            "Education",
                            "Experience",
                            "Score",
                            "Skill Score",
                            "Experience Score",
                            "Education Score",
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
