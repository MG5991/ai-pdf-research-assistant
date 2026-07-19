# AI PDF Research Assistant

## Live Demo

Try the deployed application:

[Open AI PDF Research Assistant](https://mg5991-ai-pdf-assistant.streamlit.app/)

An AI-powered research assistant that lets users upload a PDF, retrieve relevant sections by semantic meaning, and ask grounded questions about the document.

The application uses Retrieval-Augmented Generation (RAG), SentenceTransformers embeddings, and Ollama for answer generation.

## Features

- Upload a text-based PDF
- Extract text page by page
- Split PDF content into overlapping chunks
- Convert document chunks into semantic embeddings
- Retrieve relevant sections by meaning rather than keyword matching
- Ask custom questions about a PDF
- Summarize the document
- Explain the document in simple language
- Extract key points, findings, and conclusions
- Display retrieved source pages
- Display semantic similarity scores
- Inspect the retrieved chunks used to generate an answer
- Run with either local Ollama or Ollama Cloud
- Cache the embedding model and document embeddings for faster reruns

## How It Works

```text
PDF upload
    ↓
Page-by-page text extraction
    ↓
Overlapping text chunks
    ↓
SentenceTransformer embeddings
    ↓
Question embedding
    ↓
Semantic similarity retrieval
    ↓
Most relevant PDF chunks
    ↓
Local or cloud Ollama model
    ↓
Grounded answer with source pages
```

## RAG Architecture

The application follows a semantic RAG pipeline:

1. The uploaded PDF is read page by page.
2. Extracted text is cleaned and divided into overlapping chunks.
3. Each chunk is converted into a 384-dimensional semantic embedding.
4. The user's question is converted into an embedding using the same model.
5. The question embedding is compared with all document embeddings.
6. The most relevant chunks are selected.
7. Only the retrieved chunks are sent to the selected Ollama language model.
8. The generated answer is displayed together with the retrieved source pages.

## Tech Stack

- Python
- Streamlit
- PyPDF
- SentenceTransformers
- `all-MiniLM-L6-v2` embedding model
- Ollama
- Llama 3.2 3B for local generation
- GPT-OSS through Ollama Cloud for deployed generation
- Git and GitHub
- Streamlit Community Cloud

## Model Modes

The application automatically supports two execution modes.

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

## Project Structure

```text
ai-pdf-research-assistant/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
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
Retriever: Semantic embeddings
Embedding model: all-MiniLM-L6-v2
```

## Run with Ollama Cloud

Set the Ollama API key as an environment variable:

```bash
read -s -p "Paste Ollama API key: " OLLAMA_API_KEY
export OLLAMA_API_KEY
echo
```

Then run:

```bash
python -m streamlit run app.py
```

Do not place the real API key inside `app.py`, `README.md`, or any committed file.

## Deployment

The public version is deployed using Streamlit Community Cloud.

Deployment configuration:

```text
Repository: MG5991/ai-pdf-research-assistant
Branch: main
Main file: app.py
```

The following secret is configured through Streamlit's deployment settings:

```toml
OLLAMA_API_KEY = "your_private_ollama_api_key"
```

The real key must never be committed to GitHub.

## Example Questions

- What is the main contribution of this paper?
- What research gap does this study address?
- What methodology did the authors use?
- What are the main findings?
- What limitations are discussed?
- Which approach performed best?
- Explain this paper in simple language.
- Summarize the methods, results, and conclusions.
- Extract the most important key points.

## Semantic Retrieval

The current version uses:

- Character-based overlapping chunks
- SentenceTransformers document embeddings
- SentenceTransformers query embeddings
- Normalized 384-dimensional vectors
- Semantic similarity ranking
- Adjustable top-k retrieval
- Source-page metadata
- Streamlit caching for the model and document embeddings

Unlike keyword-based TF-IDF retrieval, semantic embeddings can connect phrases with similar meanings even when they do not share the same words.

For example:

```text
Question:
Which model performed best?

Document:
Shrinkage-LDA achieved the highest classification accuracy.
```

Semantic retrieval can recognize that these statements are related.

## Current Limitations

- The application currently processes one PDF at a time
- It works primarily with text-based PDFs
- Scanned or image-based PDFs require OCR, which is not yet included
- The document index is recreated when a new PDF is processed
- Embeddings are currently stored in application memory rather than a persistent vector database
- The local 3B language model may occasionally produce awkward wording
- Source-page references are based on retrieved chunks and should still be checked against the original document
- Large PDFs may require more processing time and memory
- Public users consume the application's Ollama Cloud allowance
- The public deployment is a portfolio demonstration rather than a high-scale production service

## Privacy

### Local mode

When local Ollama is used, PDF context and questions are processed on the user's computer.

### Cloud mode

When Ollama Cloud is used, the retrieved PDF chunks and the user's question are sent to the configured cloud language model to generate an answer.

Users should avoid uploading confidential, private, or sensitive documents to the public demonstration application.

## Planned Improvements

- Persistent vector database integration
- Reuse previously generated document indexes
- Multiple PDF support
- Chat history
- Follow-up questions
- Document comparison
- OCR for scanned PDFs
- Improved source citations
- Better interface and mobile layout
- File-size and page-count limits
- Usage and abuse protection
- Docker support
- Production-style deployment configuration

## Development Roadmap

- Session 1: Basic PDF upload and AI question answering
- Session 2: Clean project structure and documentation
- Session 3: TF-IDF RAG prototype
- Session 4: GitHub preparation and dependency testing
- Session 5: GitHub repository setup
- Session 6: Public Streamlit deployment
- Session 7: Semantic embedding retrieval
- Session 8: Persistent vector database
- Session 9: Multiple PDFs and chat history
- Session 10: Interface and portfolio polish
- Session 11: Docker and production-style packaging