# AI PDF Research Assistant

## Live Demo

Try the deployed application:

[Open AI PDF Research Assistant](https://mg5991-ai-pdf-assistant.streamlit.app/)

An AI-powered research assistant that lets users upload a PDF, create a semantic vector index, retrieve relevant document sections, and ask grounded questions about the document.

The application uses Retrieval-Augmented Generation (RAG), SentenceTransformers embeddings, ChromaDB vector search, and Ollama for answer generation.

## Features

- Upload a text-based PDF
- Extract text page by page
- Split PDF content into overlapping chunks
- Convert document chunks into semantic embeddings
- Store embeddings, text, page numbers, and metadata in ChromaDB
- Retrieve relevant sections by semantic meaning
- Reuse previously created local document indexes
- Identify PDFs using SHA-256 content hashes
- Ask custom questions about a PDF
- Summarize the document
- Explain the document in simple language
- Extract key points, findings, and conclusions
- Display retrieved source pages
- Display semantic similarity scores
- Inspect the retrieved chunks used to generate an answer
- Rebuild the current PDF index when needed
- Run with either local Ollama or Ollama Cloud

## How It Works

```text
PDF upload
    ↓
SHA-256 content hash
    ↓
Check for an existing ChromaDB index
    ↓
Page-by-page text extraction
    ↓
Overlapping text chunks
    ↓
SentenceTransformer embeddings
    ↓
ChromaDB vector index
    ↓
Question embedding
    ↓
Semantic vector search
    ↓
Relevant PDF chunks
    ↓
Local or cloud Ollama model
    ↓
Grounded answer with source pages
```

## RAG Architecture

The application follows a semantic vector-RAG pipeline:

1. The uploaded PDF is converted into bytes.
2. A SHA-256 hash is calculated from the PDF contents.
3. The hash is used to create a document-specific ChromaDB collection.
4. The application checks whether a compatible index already exists.
5. If no reusable index exists, the PDF is read page by page.
6. Extracted text is cleaned and divided into overlapping chunks.
7. Each chunk is converted into a 384-dimensional semantic embedding.
8. The embeddings, chunk text, page numbers, filename, and metadata are stored in ChromaDB.
9. The user's question is converted into an embedding using the same model.
10. ChromaDB returns the document chunks with the closest vectors.
11. Only the retrieved chunks are sent to the selected Ollama language model.
12. The generated answer is displayed together with the retrieved source pages.

## Tech Stack

- Python
- Streamlit
- PyPDF
- SentenceTransformers
- `all-MiniLM-L6-v2` embedding model
- ChromaDB
- Ollama
- Llama 3.2 3B for local generation
- GPT-OSS through Ollama Cloud for deployed generation
- Git and GitHub
- Streamlit Community Cloud

## Model Modes

The application automatically supports two generation modes.

### Local mode

When `OLLAMA_API_KEY` is not configured, the application connects to Ollama running locally:

```text
http://localhost:11434
```

The local language model is:

```text
llama3.2:3b
```

In local mode, PDF context and questions are processed on the user's computer.

### Cloud mode

When `OLLAMA_API_KEY` is configured, the application connects to Ollama Cloud.

The deployed Streamlit version uses:

```text
gpt-oss:120b
```

The API key is stored securely as a Streamlit deployment secret and is not included in the GitHub repository.

## Vector Database Modes

### Persistent local ChromaDB

When the application runs locally without an Ollama Cloud API key, it uses a persistent ChromaDB database stored in:

```text
chroma_db/
```

Each PDF is identified by its SHA-256 content hash.

When the same PDF is uploaded again, the application checks whether its ChromaDB collection already contains the expected number of chunks. If it does, the existing index is reused instead of regenerating every embedding.

The `chroma_db/` folder is excluded from Git and must not be committed to the repository.

### Temporary public ChromaDB

The Streamlit Community Cloud version uses an in-memory ChromaDB client.

Indexes may be reused while the application process remains active, but they are not guaranteed to survive:

- application restarts
- platform reboots
- inactivity shutdowns
- redeployments
- infrastructure changes

A hosted vector database would be required for durable public persistence.

## Project Structure

```text
ai-pdf-research-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── chroma_db/          # Generated locally and ignored by Git
```

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/MG5991/ai-pdf-research-assistant.git
cd ai-pdf-research-assistant
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The SentenceTransformers embedding model is downloaded automatically the first time the application runs.

## Install Ollama for Local Mode

### Linux installation

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start and enable the Ollama service:

```bash
sudo systemctl start ollama
sudo systemctl enable ollama
```

Download the local language model:

```bash
ollama pull llama3.2:3b
```

Confirm that it is installed:

```bash
ollama list
```

You should see:

```text
llama3.2:3b
```

## Run the Application Locally

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

The sidebar should display:

```text
Generation mode: Local Ollama
Language model: llama3.2:3b
Retriever: Chroma semantic search
Embedding model: all-Mini