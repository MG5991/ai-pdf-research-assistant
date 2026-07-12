# AI PDF Research Assistant

A local AI-powered research assistant that allows users to upload a PDF, search its contents, and ask grounded questions about the document.

The application uses a Retrieval-Augmented Generation workflow with TF-IDF retrieval and a local Llama model running through Ollama.

## Features

- Upload a PDF
- Extract text page by page
- Split PDF text into overlapping chunks
- Search chunks using TF-IDF and cosine similarity
- Retrieve the most relevant document sections
- Ask custom questions about the PDF
- Summarize the document
- Explain the document in simple language
- Extract key points
- Display retrieved source pages
- Show similarity scores and RAG debugging information
- Run the language model locally with Ollama
- No cloud API key required

## How It Works

```text
PDF upload
    ↓
Text extraction
    ↓
Page-based text chunking
    ↓
TF-IDF vectorization
    ↓
Cosine-similarity retrieval
    ↓
Relevant PDF chunks
    ↓
Local Ollama language model
    ↓
Grounded answer with source pages
```

## Tech Stack

- Python
- Streamlit
- PyPDF
- scikit-learn
- Ollama
- Llama 3.2

## Project Structure

```text
ai-pdf-research-assistant/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

Before running the application, install Ollama and download the local model.

### Install Ollama on Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama:

```bash
sudo systemctl start ollama
sudo systemctl enable ollama
```

Download the model:

```bash
ollama pull llama3.2:3b
```

Confirm that the model is available:

```bash
ollama list
```

## Python Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

### 2. Activate it

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example Questions

- What is the main contribution of this paper?
- What methodology did the authors use?
- What are the main results?
- What limitations are discussed?
- Explain this paper in simple language.
- Summarize the main findings and conclusions.

## Current RAG Approach

The current version uses:

- Character-based overlapping chunks
- TF-IDF text vectors
- Cosine similarity
- Top-k chunk retrieval
- A local Llama 3.2 model for answer generation

This approach is lightweight and easy to run locally.

## Current Limitations

- Works primarily with text-based PDFs
- Scanned PDFs require OCR, which is not currently included
- TF-IDF depends mainly on keyword similarity
- The local 3B model may occasionally produce awkward wording
- The application currently processes one PDF at a time
- The search index is rebuilt whenever the PDF is uploaded
- Public deployment requires hosting Ollama or connecting to a remote model service

## Planned Improvements

- Semantic embedding-based retrieval
- Vector database integration
- Multiple PDF support
- Chat history
- Follow-up questions
- OCR for scanned PDFs
- Improved interface
- Docker support
- Public deployment

## Privacy

PDF text and questions are processed locally when using Ollama. The application does not need to send document contents to a cloud language-model provider.