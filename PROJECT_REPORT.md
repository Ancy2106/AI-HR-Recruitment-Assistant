# Project Report — AI HR Recruitment Assistant

## Problem
Recruiters spend time reading resumes, comparing job requirements, finding skill gaps, preparing interviews, and checking policy.

## Solution
A local-first AI recruitment decision-support application built with Streamlit, LangChain and Ollama.

## Scoring
70 points skill match + 20 experience + 10 education evidence. The numeric score is deterministic and explainable rather than LLM-generated.

## Agent
The LangChain Recruitment Agent can call candidate analysis, matched skills, missing skills, and HR policy tools before responding.

## RAG
HR policy is embedded with OllamaEmbeddings and retrieved through LangChain InMemoryVectorStore before Ollama generates an answer.

## Responsible AI
Use job-related evidence only, exclude protected characteristics, show evidence and gaps, and require a human recruiter to make the final hiring decision.

## Future work
Persistent vector database, authentication, audit logs, richer resume parsing, evaluation datasets, recruiter feedback, enterprise integrations.
