# AI Resume Screener with BERT

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellowgreen)
![Gradio](https://img.shields.io/badge/Interface-Gradio-FF6B6B)

An AI-powered resume screening system that uses BERT to automatically rank and classify resumes based on job descriptions, with both Gradio and Streamlit web interfaces.

## Features

- Extracts text from PDF/DOCX resumes
- Generates BERT embeddings for semantic matching
- Ranks resumes by job description relevance
- Dual web interfaces: Gradio (fast prototyping) + Streamlit (production)
- Precision@K evaluation metrics

## Installation

git clone https://github.com/benny-daniel6/ai-resume-screener.git
cd ai-resume-screener
pip install -r requirements.txt

## Usage
1. Command Line Version

python scripts/rank_resumes.py --job_desc "job_descriptions/data_scientist.txt" --resumes "resumes/"

2. Gradio Interface (Fast Prototyping)

python app_gradio.py
Access at http://localhost:7860
Gradio Interface

3. Streamlit Interface (Production)
streamlit run app_streamlit.py

## Project Structure

├── data/
│   ├── resumedataset.csv        
├── scripts/
│   ├── AI_Resume_Screener_with_BERT.py     # Text extraction
|
├── requirements.txt
└── README.md

## License
MIT License
