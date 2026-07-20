# AI PDF Research Assistant

## Live Demo

Try the deployed application:

[Open AI PDF Research Assistant](https://mg5991-ai-pdf-assistant.streamlit.app/)

An AI-powered multi-document research assistant that allows users to upload PDF documents, create semantic vector indexes, retrieve relevant sections, compare documents, and ask grounded follow-up questions through a conversational interface.

The application uses Retrieval-Augmented Generation (RAG), SentenceTransformers embeddings, ChromaDB vector search, and Ollama for answer generation.

## Features

- Upload up to five PDF documents
- Process multiple documents in one research session
- Extract text page by page
- Split PDF content into overlapping text chunks
- Convert document chunks into semantic embeddings
- Store embeddings, text, filenames, page numbers, and metadata in ChromaDB
- Search across all uploaded documents
- Retrieve relevant sections by semantic meaning
- Reuse previously created indexes in local mode
- Identify PDFs using SHA-256 content hashes
- Detect and skip duplicate PDF content
- Limit uploaded file size to 20 MB per PDF
- Ask custom questions about the uploaded documents
- Ask contextual follow-up questions
- Keep conversation history during the current session
- Compare methods, results, and conclusions across documents
- Summarize each uploaded document
- Extract important findings and key points
- Display filename and page references
- Display semantic similarity scores
- Inspect retrieved chunks and retrieval details
- Clear the current conversation
- Rebuild uploaded document indexes
- Run with either local Ollama or Ollama Cloud

## How It Works

```text
Multiple PDF uploads
        ↓
File validation and duplicate detection
        ↓
SHA-256 content hash for each document
        ↓
Check for existing ChromaDB collections
        ↓
Page-by-page text extraction
        ↓
Overlapping text chunks
        ↓
SentenceTransformer embeddings
        ↓
Document-specific ChromaDB indexes
        ↓
User question and recent chat context
        ↓
Semantic search across all uploaded documents
        ↓
Highest-ranking chunks from each document
        ↓
Local or cloud Ollama model
        ↓
Grounded conversational answer
        ↓
Filename and page references
```

## RAG Architecture

The application follows a multi-document conversational RAG pipeline:

1. The user uploads one or more PDF documents.
2. Each file is validated against the file-count and file-size limits.
3. A SHA-256 hash is calculated from the contents of each PDF.
4. Duplicate PDFs are detected using their content hashes.
5. Each valid PDF is read page by page.
6. Extracted text is cleaned and divided into overlapping chunks.
7. Each chunk is converted into a 384-dimensional semantic embedding.
8. The embeddings, text, filename, page number, document hash, and chunk metadata are stored in ChromaDB.
9. Each PDF uses its own document-specific ChromaDB collection.
10. The user's question is combined with recent user questions when additional follow-up context is needed.
11. The retrieval query is converted into a semantic embedding.
12. ChromaDB searches each uploaded document for relevant chunks.
13. The strongest result from each document is included before the remaining retrieval positions are filled.
14. The selected PDF sections and recent conversation are sent to the Ollama language model.
15. The generated answer is displayed with filename and page references.
16. The question, answer, and sources remain visible in the current Streamlit session.

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

## Generation Modes

The application automatically supports two generation modes.

### Local Ollama mode

When `OLLAMA_API_KEY` is not configured, the application connects to Ollama running locally:

```text
http://localhost:11434
```

The local language model is:

```text
llama3.2:3b
```

In local mode:

- PDF text is extracted locally
- embeddings are generated locally
- vector indexes are stored locally
- questions and retrieved context are processed by the local Ollama model

### Ollama Cloud mode

When `OLLAMA_API_KEY` is configured, the application connects to Ollama Cloud.

The deployed Streamlit version uses:

```text
gpt-oss:120b
```

The API key is stored securely as a Streamlit deployment secret and is not included in the GitHub repository.

In cloud mode, the retrieved PDF sections and the user's question are sent to the configured Ollama Cloud model for answer generation.

## Vector Database Modes

### Persistent local ChromaDB

When the application runs without an Ollama Cloud API key, it uses a persistent ChromaDB database stored in:

```text
chroma_db/
```

Each PDF is identified by its SHA-256 content hash.

When the same PDF is uploaded again, the application checks whether its ChromaDB collection already contains the expected number of chunks. When the stored index is complete, it is reused instead of regenerating every embedding.

The `chroma_db/` directory is excluded from Git and must not be committed to the repository.

### Temporary public ChromaDB

When Ollama Cloud mode is active, the application uses an in-memory ChromaDB client.

Public vector indexes may be reused while the Streamlit process remains active, but they are not guaranteed to survive:

- application restarts
- inactivity shutdowns
- platform reboots
- redeployments
- infrastructure changes

A hosted vector database would be required for durable public persistence.

## Multi-Document Retrieval

The application searches every indexed PDF separately.

For each question:

1. The question is embedded using SentenceTransformers.
2. Every uploaded document collection is searched.
3. The highest-ranking result from each document is retained.
4. Remaining retrieval positions are filled using the strongest results across all documents.
5. Results are sorted by semantic similarity.
6. The selected chunks are sent to the language model.

This approach helps prevent one large document from completely dominating the retrieved context.

## Conversational Follow-Up Questions

The application keeps recent questions and answers in Streamlit Session State.

For short follow-up questions such as:

```text
Which one performed better?
```

the retrieval query can include recent user questions, helping the system understand what “which one” refers to.

Recent assistant answers are included for conversational understanding, but they are not treated as factual evidence. The uploaded PDF context remains the only factual source.

Conversation history is temporary and belongs only to the current Streamlit session.

## Source References

Generated answers are instructed to cite important claims using the following format:

```text
(filename.pdf, page 4)
```

The interface also displays a separate source list containing the filenames and page numbers of the retrieved chunks.

Source references are based on retrieval results and should still be checked against the original PDF when accuracy is critical.

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

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The SentenceTransformers embedding model is downloaded automatically the first time the application runs.

## Python Dependencies

The current `requirements.txt` contains:

```text
streamlit
pypdf
ollama>=0.6.2
sentence-transformers
chromadb
```

## Install Ollama for Local Mode

### Linux installation

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama:

```bash
sudo systemctl start ollama
```

Enable Ollama at system startup:

```bash
sudo systemctl enable ollama
```

Download the local language model:

```bash
ollama pull llama3.2:3b
```

Confirm that the model is available:

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
Retriever: Multi-document Chroma search
Embedding model: all-MiniLM-L6-v2
Vector database: Persistent local Chroma
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

The sidebar should display:

```text
Generation mode: Ollama Cloud
Language model: gpt-oss:120b
Retriever: Multi-document Chroma search
Embedding model: all-MiniLM-L6-v2
Vector database: Temporary Chroma
```

Do not place the real API key inside:

- `app.py`
- `README.md`
- `requirements.txt`
- Git commits
- public screenshots
- any committed configuration file

## Deployment

The public version is deployed using Streamlit Community Cloud.

Deployment configuration:

```text
Repository: MG5991/ai-pdf-research-assistant
Branch: main
Main file: app.py
```

The Ollama Cloud API key is configured through Streamlit deployment secrets:

```toml
OLLAMA_API_KEY = "your_private_ollama_api_key"
```

The real API key must never be committed to GitHub.

## Using the Application

### Upload documents

Upload between one and five text-based PDF documents.

Each file must be no larger than:

```text
20 MB
```

Duplicate PDF content is automatically detected and skipped.

### Ask a custom question

Examples:

- What is the main topic of each uploaded document?
- What research gap does each paper address?
- Which documents use machine learning?
- What methods were used in each study?
- Which model achieved the best result?
- What are the main findings?
- What limitations are discussed?
- How do the conclusions differ?
- Which document provides stronger experimental evidence?

### Ask follow-up questions

Examples:

- Which one performed better?
- What dataset did it use?
- How was that method evaluated?
- Did the other paper use the same approach?
- What limitations did the authors mention?

### Use quick actions

The interface provides three quick actions:

- **Summarize documents**
- **Compare documents**
- **Extract key points**

### Manage the session

The sidebar includes controls to:

- clear the current conversation
- rebuild all uploaded document indexes
- adjust the number of retrieved chunks
- display retrieved text chunks
- inspect document and index statistics

## Example Multi-Document Workflow

```text
Upload paper_a.pdf and paper_b.pdf

Question:
What is the main topic of each document?

Follow-up:
Which one uses machine learning?

Follow-up:
What model does it use?

Comparison:
Compare their methods, findings, and limitations.
```

## Semantic Vector Retrieval

The current version uses:

- page-based PDF extraction
- character-based overlapping chunks
- SentenceTransformers document embeddings
- SentenceTransformers query embeddings
- normalized 384-dimensional vectors
- ChromaDB cosine-distance search
- adjustable top-k retrieval
- document-specific vector collections
- filename and page metadata
- SHA-256 document identification
- duplicate-document detection
- persistent local index reuse
- temporary cloud indexes
- multi-document result merging
- follow-up query enrichment
- retrieval similarity scores
- index rebuilding controls

Unlike keyword-based retrieval, semantic embeddings can connect phrases with similar meanings even when they do not contain the same words.

For example:

```text
Question:
Which model performed best?

Document:
Shrinkage-LDA achieved the highest classification accuracy.
```

Semantic vector retrieval can recognize that these statements are related.

## Index Reuse

Each local ChromaDB collection name is generated using:

```text
PDF content hash + index version
```

An existing index is reused when:

- the same PDF content is uploaded again
- a compatible collection already exists
- the stored chunk count matches the expected chunk count

An index is rebuilt when:

- no collection exists
- the stored collection is incomplete
- the user selects the rebuild option
- the index version changes
- the chunking strategy changes
- the embedding strategy changes

## File Protection

The application currently applies the following upload rules:

```text
Maximum files per session: 5
Maximum size per PDF: 20 MB
Accepted format: PDF
Duplicate content: skipped
Scanned PDFs without extractable text: skipped
```

These restrictions help reduce memory usage, processing time, and accidental resource abuse.

## Current Limitations

- The application supports a maximum of five PDFs per session
- Each PDF is limited to 20 MB
- It works primarily with text-based PDFs
- Scanned or image-based PDFs require OCR, which is not currently included
- Chat history exists only in the current Streamlit session
- Chat history disappears after the session is closed or reset
- Public vector indexes are temporary
- Local vector indexes persist only on the machine where they were created
- Follow-up context uses recent conversation turns rather than permanent memory
- Retrieval searches document text but does not directly analyze images, charts, or diagrams
- Source references depend on the quality of extracted text and retrieved chunks
- The local 3B language model may occasionally produce awkward wording
- Large or complex PDFs may require additional processing time and memory
- The current retrieval system does not include a cross-encoder reranker
- The application does not currently provide user accounts or private document libraries
- Public users consume the application's Ollama Cloud allowance
- The public deployment is a portfolio demonstration rather than a high-scale production service

## Privacy

### Local mode

When local Ollama mode is used:

- PDFs are processed on the user's computer
- text extraction happens locally
- embeddings are generated locally
- vectors are stored in the local `chroma_db/` directory
- questions and retrieved PDF context are processed by the local Ollama model

### Cloud mode

When Ollama Cloud mode is used:

- PDF extraction and embedding generation happen inside the running application
- retrieved PDF chunks and user questions are sent to the configured Ollama Cloud model
- vector indexes are temporary
- chat history is stored only in the current Streamlit session

Users should not upload confidential, private, legally restricted, or sensitive documents to the public demonstration application.

## Planned Improvements

- Hosted vector database integration
- Durable public vector-index persistence
- Automated RAG evaluation
- Retrieval-quality testing
- Answer-faithfulness evaluation
- Unit and integration tests
- GitHub Actions CI/CD
- Cross-encoder reranking
- Hybrid keyword and semantic retrieval
- Query rewriting
- Improved citation verification
- OCR for scanned PDFs
- Image, chart, and table extraction
- User authentication
- Private document libraries
- Saved conversations
- Export answers to Markdown or Word
- Improved interface and mobile layout
- Usage analytics and monitoring
- Rate limiting and abuse protection
- Docker support
- Production-style deployment configuration

## Disclaimer

AI-generated answers may contain mistakes or incomplete interpretations.

Users should verify important findings, numerical results, citations, and conclusions against the original uploaded documents.