# StudyMate – AI-Powered Exam Preparation Assistant

StudyMate is a Generative AI-based tool designed to help students during exam preparation.  
It allows users to upload study materials and generates concise summaries to improve learning efficiency.

## 🚀 Features
- Upload PDF or text study material  
- AI-powered summarization  
- Concise, exam-focused notes  
- Fast content condensation for quick revision  

## 🏗️ Tech Stack
- Python  
- Streamlit (Interactive Web Application)  
- PyMuPDF (PDF text extraction)  
- SentenceTransformers (all-MiniLM-L6-v2)  
- FAISS (Vector similarity search)  
- PyTorch (Model execution backend)  
- Hugging Face Transformers  
- IBM Granite 3.3 2B Instruct (LLM)  

## 🧠 How It Works
1. User uploads study material  
2. Text is extracted and preprocessed  
3. Content embeddings are generated  
4. Relevant chunks are retrieved using FAISS  
5. LLM generates a structured summary  
6. Summary is displayed to the user  

## 🎯 Motivation
Built during a Generative AI Hackathon to address the problem of time-consuming revision during exams.
