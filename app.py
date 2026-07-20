import hashlib
import os
import re
from io import BytesIO

import chromadb
import streamlit as st
from ollama import Client
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Model configuration
# -----------------------------

LOCAL_MODEL = "llama3.2:3b"
CLOUD_MODEL = "gpt-oss:120b"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_DIRECTORY = "chroma_db"

# Change this value in the future if the chunking
# or embedding strategy changes substantially.
INDEX_VERSION = "v1"

ollama_api_key = os.getenv(
    "OLLAMA_API_KEY",
    "",
).strip()

if ollama_api_key:
    MODEL_MODE = "Ollama Cloud"
    ACTIVE_MODEL = CLOUD_MODEL

    ollama_client = Client(
        host="https://ollama.com",
        headers={
            "Authorization": (
                f"Bearer {ollama_api_key}"
            ),
        },
    )

    # Public deployment uses temporary storage.
    USE_PERSISTENT_CHROMA = False
    VECTOR_DB_MODE = "Temporary Chroma"

else:
    MODEL_MODE = "Local Ollama"
    ACTIVE_MODEL = LOCAL_MODEL

    ollama_client = Client(
        host="http://localhost:11434",
    )

    # Local development persists indexes to disk.
    USE_PERSISTENT_CHROMA = True
    VECTOR_DB_MODE = "Persistent local Chroma"


# -----------------------------
# 2. Streamlit page setup
# -----------------------------

st.set_page_config(
    page_title="AI PDF Research Assistant",
    page_icon="📄",
    layout="wide",
)

st.title(
    "AI PDF Research Assistant — Vector RAG"
)

st.write(
    "Upload a PDF and ask questions about it. "
    "The app stores semantic document vectors in "
    "ChromaDB and retrieves the most relevant sections "
    "before generating an answer."
)


# -----------------------------
# 3. Load embedding model
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Load and cache the SentenceTransformers model.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# -----------------------------
# 4. Create Chroma client
# -----------------------------

@st.cache_resource(show_spinner=False)
def get_chroma_client(
    use_persistent_storage: bool,
):
    """
    Create and cache the Chroma client.

    Local mode:
        Persistent database stored in chroma_db/.

    Cloud mode:
        Temporary in-memory database.
    """

    if use_persistent_storage:
        return chromadb.PersistentClient(
            path=CHROMA_DIRECTORY
        )

    return chromadb.EphemeralClient()


# -----------------------------
# 5. Calculate PDF hash
# -----------------------------

def calculate_pdf_hash(
    pdf_bytes: bytes,
) -> str:
    """
    Generate a unique SHA-256 identifier
    based on the PDF file contents.
    """

    return hashlib.sha256(
        pdf_bytes
    ).hexdigest()


def create_collection_name(
    document_hash: str,
) -> str:
    """
    Create a valid Chroma collection name.

    The hash identifies the document.
    The index version identifies the retrieval setup.
    """

    return (
        f"pdf_{document_hash[:24]}_{INDEX_VERSION}"
    )


# -----------------------------
# 6. Clean extracted text
# -----------------------------

def clean_text(
    text: str,
) -> str:
    """
    Remove repeated spaces, tabs, and line breaks.
    """

    cleaned_text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return cleaned_text.strip()


# -----------------------------
# 7. Read PDF page by page
# -----------------------------

@st.cache_data(show_spinner=False)
def read_pdf_with_pages(
    pdf_bytes: bytes,
) -> list[dict]:
    """
    Extract readable text from each PDF page.
    """

    pdf_reader = PdfReader(
        BytesIO(pdf_bytes)
    )

    pages = []

    for page_number, page in enumerate(
        pdf_reader.pages,
        start=1,
    ):
        page_text = page.extract_text()

        if page_text and page_text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": clean_text(page_text),
                }
            )

    return pages


# -----------------------------
# 8. Split pages into chunks
# -----------------------------

@st.cache_data(show_spinner=False)
def split_pages_into_chunks(
    pages_as_tuple: tuple,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """
    Split each page into overlapping text chunks.

    A tuple is used as input so Streamlit can cache
    the result reliably.
    """

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if overlap < 0:
        raise ValueError(
            "Overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    pages = [
        {
            "page": page_number,
            "text": page_text,
        }
        for page_number, page_text in pages_as_tuple
    ]

    chunks = []
    step = chunk_size - overlap
    chunk_number = 0

    for page in pages:
        page_number = page["page"]
        page_text = page["text"]

        start = 0

        while start < len(page_text):
            end = start + chunk_size

            chunk_text = (
                page_text[start:end].strip()
            )

            if chunk_text:
                chunks.append(
                    {
                        "chunk_number": chunk_number,
                        "page": page_number,
                        "text": chunk_text,
                    }
                )

                chunk_number += 1

            start += step

    return chunks


# -----------------------------
# 9. Create chunk embeddings
# -----------------------------

def create_chunk_embeddings(
    chunks: list[dict],
):
    """
    Convert all document chunks into normalized
    semantic vectors.
    """

    embedding_model = load_embedding_model()

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    return embedding_model.encode_document(
        chunk_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


# -----------------------------
# 10. Create or reuse Chroma index
# -----------------------------

def create_or_reuse_document_index(
    document_hash: str,
    filename: str,
    chunks: list[dict],
):
    """
    Create a Chroma collection for the PDF.

    If the collection already contains the expected
    number of chunks, reuse it without regenerating
    embeddings.
    """

    chroma_client = get_chroma_client(
        USE_PERSISTENT_CHROMA
    )

    collection_name = create_collection_name(
        document_hash
    )

    collection = (
        chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={
                "document_hash": document_hash,
                "filename": filename,
                "embedding_model": (
                    EMBEDDING_MODEL_NAME
                ),
                "index_version": INDEX_VERSION,
            },
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )
    )

    existing_count = collection.count()
    expected_count = len(chunks)

    if (
        existing_count > 0
        and existing_count == expected_count
    ):
        return (
            collection,
            collection_name,
            True,
        )

    # A partial or incompatible index should not
    # be reused.
    if existing_count > 0:
        chroma_client.delete_collection(
            name=collection_name
        )

        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={
                "document_hash": document_hash,
                "filename": filename,
                "embedding_model": (
                    EMBEDDING_MODEL_NAME
                ),
                "index_version": INDEX_VERSION,
            },
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )

    chunk_embeddings = (
        create_chunk_embeddings(chunks)
    )

    chunk_ids = [
        (
            f"{document_hash[:12]}_"
            f"chunk_{chunk['chunk_number']:05d}"
        )
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "document_hash": document_hash,
            "filename": filename,
            "page": chunk["page"],
            "chunk_number": (
                chunk["chunk_number"]
            ),
        }
        for chunk in chunks
    ]

    collection.add(
        ids=chunk_ids,
        embeddings=chunk_embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    return (
        collection,
        collection_name,
        False,
    )


# -----------------------------
# 11. Query Chroma
# -----------------------------

def retrieve_relevant_chunks(
    question: str,
    collection,
    top_k: int = 5,
) -> list[dict]:
    """
    Embed the question and query the Chroma
    collection for the closest document chunks.
    """

    if not question.strip():
        return []

    record_count = collection.count()

    if record_count == 0:
        return []

    embedding_model = load_embedding_model()

    question_embedding = (
        embedding_model.encode_query(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )

    number_to_retrieve = min(
        top_k,
        record_count,
    )

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=number_to_retrieve,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (
        results.get("documents") or [[]]
    )[0]

    metadatas = (
        results.get("metadatas") or [[]]
    )[0]

    distances = (
        results.get("distances") or [[]]
    )[0]

    retrieved_chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        # For cosine distance:
        # similarity = 1 - distance
        similarity_score = (
            1.0 - float(distance)
        )

        retrieved_chunks.append(
            {
                "page": metadata["page"],
                "chunk_number": (
                    metadata["chunk_number"]
                ),
                "filename": metadata["filename"],
                "text": document,
                "score": similarity_score,
            }
        )

    return retrieved_chunks


# -----------------------------
# 12. Create grounded prompt
# -----------------------------

def create_prompt(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Build a prompt using only retrieved PDF text.
    """

    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Document: {chunk['filename']} | "
            f"Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a precise academic research assistant.

Answer the user's question using only the PDF context below.

Instructions:
- Give the direct answer first.
- Use clear and natural English.
- Paraphrase the source accurately.
- Do not use outside knowledge.
- Do not invent facts, methods, datasets, results, or conclusions.
- Avoid unnecessary repetition.
- Cite the relevant page number after important claims,
  using the format: (page 2).
- If the retrieved context is insufficient, say:
  "I could not find enough information in the retrieved PDF sections."
- Keep the answer concise unless the user requests detail.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    return prompt.strip()


# -----------------------------
# 13. Ask Ollama
# -----------------------------

def ask_ollama_model(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Generate an answer using either local Ollama
    or Ollama Cloud.
    """

    prompt = create_prompt(
        question,
        retrieved_chunks,
    )

    response = ollama_client.chat(
        model=ACTIVE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful academic "
                    "document-analysis assistant. "
                    "Use only the supplied PDF context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.message.content


# -----------------------------
# 14. Sidebar settings
# -----------------------------

st.sidebar.header("RAG Settings")

st.sidebar.caption(
    f"Generation mode: {MODEL_MODE}"
)

st.sidebar.caption(
    f"Language model: {ACTIVE_MODEL}"
)

st.sidebar.caption(
    "Retriever: Chroma semantic search"
)

st.sidebar.caption(
    "Embedding model: all-MiniLM-L6-v2"
)

st.sidebar.caption(
    f"Vector database: {VECTOR_DB_MODE}"
)

top_k = st.sidebar.slider(
    "Number of chunks to retrieve",
    min_value=3,
    max_value=10,
    value=5,
)

show_chunks = st.sidebar.checkbox(
    "Show retrieved chunks",
    value=True,
)


# -----------------------------
# 15. PDF upload
# -----------------------------

uploaded_pdf = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
)

if uploaded_pdf is None:
    st.warning(
        "Upload a PDF to begin."
    )

    st.stop()

pdf_bytes = uploaded_pdf.getvalue()

document_hash = calculate_pdf_hash(
    pdf_bytes
)

st.success(
    f"Uploaded: {uploaded_pdf.name}"
)


# -----------------------------
# 16. Process PDF and build index
# -----------------------------

try:
    with st.spinner("Reading PDF..."):
        pages = read_pdf_with_pages(
            pdf_bytes
        )

    if not pages:
        st.error(
            "No readable text was found. "
            "The PDF may contain scanned images."
        )

        st.stop()

    pages_as_tuple = tuple(
        (
            page["page"],
            page["text"],
        )
        for page in pages
    )

    with st.spinner(
        "Splitting PDF into chunks..."
    ):
        chunks = split_pages_into_chunks(
            pages_as_tuple
        )

    if not chunks:
        st.error(
            "The app could not create "
            "searchable text chunks."
        )

        st.stop()

    with st.spinner(
        "Loading or creating Chroma index..."
    ):
        (
            collection,
            collection_name,
            index_reused,
        ) = create_or_reuse_document_index(
            document_hash=document_hash,
            filename=uploaded_pdf.name,
            chunks=chunks,
        )

except Exception as error:
    st.error(
        "The PDF could not be processed or indexed."
    )

    st.code(
        f"{type(error).__name__}: {error}"
    )

    st.stop()


if index_reused:
    st.info(
        f"Reused an existing Chroma index containing "
        f"{collection.count()} document chunks."
    )

else:
    st.info(
        f"Created a new Chroma index containing "
        f"{collection.count()} document chunks."
    )


# -----------------------------
# 17. Index controls
# -----------------------------

st.sidebar.metric(
    "Indexed chunks",
    collection.count(),
)

if st.sidebar.button(
    "Rebuild current PDF index",
    use_container_width=True,
):
    chroma_client = get_chroma_client(
        USE_PERSISTENT_CHROMA
    )

    chroma_client.delete_collection(
        name=collection_name
    )

    st.cache_data.clear()

    st.rerun()


# -----------------------------
# 18. Quick actions
# -----------------------------

st.subheader("Quick actions")

col1, col2, col3 = st.columns(3)

with col1:
    summarize_button = st.button(
        "Summarize",
        use_container_width=True,
    )

with col2:
    beginner_button = st.button(
        "Explain simply",
        use_container_width=True,
    )

with col3:
    key_points_button = st.button(
        "Key points",
        use_container_width=True,
    )


# -----------------------------
# 19. Custom question
# -----------------------------

question = st.text_input(
    "Or ask your own question about the PDF:",
    placeholder=(
        "For example: "
        "What research gap does this paper address?"
    ),
)


# -----------------------------
# 20. Decide final question
# -----------------------------

final_question = None

if summarize_button:
    final_question = (
        "Summarize the main ideas, research "
        "approach, results, and conclusions "
        "of this PDF."
    )

elif beginner_button:
    final_question = (
        "Explain the main ideas of this PDF "
        "in simple beginner-friendly language."
    )

elif key_points_button:
    final_question = (
        "Extract the most important key points, "
        "findings, and conclusions from this PDF."
    )

elif question.strip():
    final_question = question.strip()


# -----------------------------
# 21. Retrieve context and answer
# -----------------------------

if final_question:
    with st.spinner(
        "Querying the Chroma vector database..."
    ):
        retrieved_chunks = (
            retrieve_relevant_chunks(
                question=final_question,
                collection=collection,
                top_k=top_k,
            )
        )

    if not retrieved_chunks:
        st.error(
            "No relevant PDF chunks were found."
        )

        st.stop()

    source_pages = sorted(
        {
            chunk["page"]
            for chunk in retrieved_chunks
        }
    )

    source_pages_text = ", ".join(
        str(page)
        for page in source_pages
    )

    best_score = retrieved_chunks[0]["score"]

    st.caption(
        f"Using {len(retrieved_chunks)} "
        f"Chroma results from page(s): "
        f"{source_pages_text}."
    )

    if best_score < 0.15:
        st.warning(
            "Semantic retrieval confidence is low. "
            "The retrieved sections may not strongly "
            "match the question."
        )

    with st.expander(
        "How vector RAG worked"
    ):
        st.markdown(
            """
1. The app calculated a unique hash for the PDF.
2. It checked whether a Chroma collection already existed.
3. New PDFs were split into overlapping chunks.
4. Each chunk was converted into a semantic embedding.
5. The vectors, text, page numbers, and metadata were stored in Chroma.
6. The question was converted into an embedding.
7. Chroma returned the nearest document vectors.
8. The retrieved chunks were sent to Ollama.
9. Ollama generated an answer using the retrieved context.
            """
        )

        st.write(
            f"Document hash: "
            f"{document_hash[:16]}..."
        )

        st.write(
            f"Collection name: "
            f"{collection_name}"
        )

        st.write(
            f"Index reused: "
            f"{index_reused}"
        )

        st.write(
            f"Retrieved source pages: "
            f"{source_pages_text}"
        )

        st.write(
            f"Best similarity score: "
            f"{best_score:.3f}"
        )

        st.write(
            f"Stored chunks: "
            f"{collection.count()}"
        )

        st.write(
            f"Vector database mode: "
            f"{VECTOR_DB_MODE}"
        )

        st.write(
            f"Embedding model: "
            f"{EMBEDDING_MODEL_NAME}"
        )

        st.write(
            f"Generation mode: "
            f"{MODEL_MODE}"
        )

        st.write(
            f"Language model: "
            f"{ACTIVE_MODEL}"
        )

    if show_chunks:
        with st.expander(
            "Show retrieved chunks"
        ):
            for index, chunk in enumerate(
                retrieved_chunks,
                start=1,
            ):
                st.markdown(
                    f"**Chunk {index} — "
                    f"Page {chunk['page']} — "
                    f"Similarity: "
                    f"{chunk['score']:.3f}**"
                )

                st.write(
                    chunk["text"]
                )

    with st.spinner(
        f"Generating answer with "
        f"{ACTIVE_MODEL}..."
    ):
        try:
            answer = ask_ollama_model(
                final_question,
                retrieved_chunks,
            )

        except Exception as error:
            st.error(
                f"The app could not connect using "
                f"{MODEL_MODE}. Check the local "
                "Ollama service or cloud credentials."
            )

            st.code(
                f"{type(error).__name__}: {error}"
            )

            st.stop()

    if not answer or not answer.strip():
        st.error(
            "The selected model returned "
            "an empty answer."
        )

        st.stop()

    st.subheader("Answer")

    st.write(answer)

    st.markdown(
        f"**Retrieved source pages:** "
        f"{source_pages_text}"
    )