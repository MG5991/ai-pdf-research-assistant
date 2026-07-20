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
# 1. Configuration
# -----------------------------

LOCAL_MODEL = "llama3.2:3b"
CLOUD_MODEL = "gpt-oss:120b"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_DIRECTORY = "chroma_db"
INDEX_VERSION = "v1"

MAX_PDF_FILES = 5
MAX_FILE_SIZE_MB = 20
MAX_CHAT_HISTORY_MESSAGES = 8


# -----------------------------
# 2. Ollama configuration
# -----------------------------

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

    USE_PERSISTENT_CHROMA = False
    VECTOR_DB_MODE = "Temporary Chroma"

else:
    MODEL_MODE = "Local Ollama"
    ACTIVE_MODEL = LOCAL_MODEL

    ollama_client = Client(
        host="http://localhost:11434"
    )

    USE_PERSISTENT_CHROMA = True
    VECTOR_DB_MODE = "Persistent local Chroma"


# -----------------------------
# 3. Streamlit page setup
# -----------------------------

st.set_page_config(
    page_title="AI PDF Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "AI PDF Research Assistant is a "
            "multi-document Retrieval-Augmented "
            "Generation portfolio project."
        ),
    },
)


with st.container(
    border=True,
):
    st.title(
        "AI PDF Research Assistant"
    )

    st.markdown(
        "### Multi-document research, grounded in your sources"
    )

    st.write(
        "Upload research papers or reports, search "
        "them using semantic vector retrieval, compare "
        "documents, and ask contextual follow-up questions."
    )

    feature_col1, feature_col2, feature_col3 = (
        st.columns(
            3,
            gap="medium",
        )
    )

    with feature_col1:
        st.markdown(
            "**Semantic retrieval**"
        )

        st.caption(
            "Find relevant sections by meaning, "
            "not only matching keywords."
        )

    with feature_col2:
        st.markdown(
            "**Multi-document analysis**"
        )

        st.caption(
            "Search, summarize, and compare up "
            "to five PDF documents."
        )

    with feature_col3:
        st.markdown(
            "**Grounded answers**"
        )

        st.caption(
            "Inspect filenames, page references, "
            "retrieved chunks, and similarity scores."
        )

st.caption(
    "Portfolio demonstration · "
    "SentenceTransformers · ChromaDB · Ollama"
)


# -----------------------------
# 4. Session state
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_set_signature" not in st.session_state:
    st.session_state.document_set_signature = None


# -----------------------------
# 5. Load embedding model
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
# 6. Chroma clients
# -----------------------------

@st.cache_resource(show_spinner=False)
def get_persistent_chroma_client():
    """
    Create a persistent Chroma client for local use.
    """

    return chromadb.PersistentClient(
        path=CHROMA_DIRECTORY
    )


def get_chroma_client(
    use_persistent_storage: bool,
):
    """
    Return a persistent local client or a temporary
    session-based client.
    """

    if use_persistent_storage:
        return get_persistent_chroma_client()

    if (
        "ephemeral_chroma_client"
        not in st.session_state
    ):
        st.session_state.ephemeral_chroma_client = (
            chromadb.EphemeralClient()
        )

    return (
        st.session_state.ephemeral_chroma_client
    )


# -----------------------------
# 7. Document identifiers
# -----------------------------

def calculate_pdf_hash(
    pdf_bytes: bytes,
) -> str:
    """
    Generate a SHA-256 hash from PDF contents.
    """

    return hashlib.sha256(
        pdf_bytes
    ).hexdigest()


def create_collection_name(
    document_hash: str,
) -> str:
    """
    Create a valid document-specific Chroma
    collection name.
    """

    return (
        f"pdf_{document_hash[:24]}_"
        f"{INDEX_VERSION}"
    )


def create_document_set_signature(
    document_hashes: list[str],
) -> str:
    """
    Create a signature representing the current
    combination of uploaded documents.
    """

    combined_hashes = "|".join(
        sorted(document_hashes)
    )

    return hashlib.sha256(
        combined_hashes.encode("utf-8")
    ).hexdigest()


# -----------------------------
# 8. Text cleaning
# -----------------------------

def clean_text(
    text: str,
) -> str:
    """
    Normalize spaces, tabs, and line breaks.
    """

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# -----------------------------
# 9. PDF extraction
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
                    "text": clean_text(
                        page_text
                    ),
                }
            )

    return pages


# -----------------------------
# 10. Chunking
# -----------------------------

@st.cache_data(show_spinner=False)
def split_pages_into_chunks(
    pages_as_tuple: tuple,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """
    Split each page into overlapping text chunks.
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

    chunks = []
    step = chunk_size - overlap
    chunk_number = 0

    for page_number, page_text in pages_as_tuple:
        start = 0

        while start < len(page_text):
            chunk_text = page_text[
                start:start + chunk_size
            ].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_number": (
                            chunk_number
                        ),
                        "page": page_number,
                        "text": chunk_text,
                    }
                )

                chunk_number += 1

            start += step

    return chunks


# -----------------------------
# 11. Document embeddings
# -----------------------------

def create_chunk_embeddings(
    chunks: list[dict],
):
    """
    Convert document chunks into normalized
    semantic embeddings.
    """

    embedding_model = (
        load_embedding_model()
    )

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
# 12. Create or reuse index
# -----------------------------

def create_or_reuse_document_index(
    document_hash: str,
    filename: str,
    chunks: list[dict],
):
    """
    Create a Chroma collection for one PDF or
    reuse a complete existing collection.
    """

    chroma_client = get_chroma_client(
        USE_PERSISTENT_CHROMA
    )

    collection_name = (
        create_collection_name(
            document_hash
        )
    )

    collection_metadata = {
        "document_hash": document_hash,
        "filename": filename,
        "embedding_model": (
            EMBEDDING_MODEL_NAME
        ),
        "index_version": INDEX_VERSION,
    }

    collection = (
        chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata=collection_metadata,
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

    if existing_count > 0:
        chroma_client.delete_collection(
            name=collection_name
        )

        collection = (
            chroma_client.create_collection(
                name=collection_name,
                embedding_function=None,
                metadata=collection_metadata,
                configuration={
                    "hnsw": {
                        "space": "cosine",
                    }
                },
            )
        )

    chunk_embeddings = (
        create_chunk_embeddings(
            chunks
        )
    )

    chunk_ids = [
        (
            f"{document_hash[:12]}_"
            f"chunk_"
            f"{chunk['chunk_number']:05d}"
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
        embeddings=(
            chunk_embeddings.tolist()
        ),
        documents=documents,
        metadatas=metadatas,
    )

    return (
        collection,
        collection_name,
        False,
    )


# -----------------------------
# 13. Process uploaded PDFs
# -----------------------------

def process_uploaded_pdfs(
    uploaded_files,
) -> tuple[list[dict], list[str]]:
    """
    Validate, extract, chunk, and index all uploaded
    PDF files.
    """

    indexed_documents = []
    warnings = []
    seen_hashes = set()

    for uploaded_file in uploaded_files:
        pdf_bytes = (
            uploaded_file.getvalue()
        )

        file_size_mb = (
            len(pdf_bytes)
            / (1024 * 1024)
        )

        if file_size_mb > MAX_FILE_SIZE_MB:
            warnings.append(
                f"{uploaded_file.name}: skipped "
                f"because it exceeds "
                f"{MAX_FILE_SIZE_MB} MB."
            )

            continue

        document_hash = (
            calculate_pdf_hash(
                pdf_bytes
            )
        )

        if document_hash in seen_hashes:
            warnings.append(
                f"{uploaded_file.name}: skipped "
                "because the same PDF content "
                "was uploaded more than once."
            )

            continue

        seen_hashes.add(
            document_hash
        )

        try:
            pages = read_pdf_with_pages(
                pdf_bytes
            )

            if not pages:
                warnings.append(
                    f"{uploaded_file.name}: "
                    "no readable text was found. "
                    "It may be a scanned PDF."
                )

                continue

            pages_as_tuple = tuple(
                (
                    page["page"],
                    page["text"],
                )
                for page in pages
            )

            chunks = (
                split_pages_into_chunks(
                    pages_as_tuple
                )
            )

            if not chunks:
                warnings.append(
                    f"{uploaded_file.name}: "
                    "no searchable chunks "
                    "were created."
                )

                continue

            (
                collection,
                collection_name,
                index_reused,
            ) = (
                create_or_reuse_document_index(
                    document_hash=(
                        document_hash
                    ),
                    filename=(
                        uploaded_file.name
                    ),
                    chunks=chunks,
                )
            )

            indexed_documents.append(
                {
                    "filename": (
                        uploaded_file.name
                    ),
                    "document_hash": (
                        document_hash
                    ),
                    "collection_name": (
                        collection_name
                    ),
                    "collection": (
                        collection
                    ),
                    "index_reused": (
                        index_reused
                    ),
                    "page_count": len(
                        pages
                    ),
                    "chunk_count": (
                        collection.count()
                    ),
                    "file_size_mb": (
                        file_size_mb
                    ),
                }
            )

        except Exception as error:
            warnings.append(
                f"{uploaded_file.name}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

    return (
        indexed_documents,
        warnings,
    )


# -----------------------------
# 14. Search one document
# -----------------------------

def retrieve_from_collection(
    retrieval_query: str,
    document: dict,
    candidates_per_document: int,
) -> list[dict]:
    """
    Search one document's Chroma collection.
    """

    collection = (
        document["collection"]
    )

    record_count = (
        collection.count()
    )

    if record_count == 0:
        return []

    embedding_model = (
        load_embedding_model()
    )

    query_embedding = (
        embedding_model.encode_query(
            retrieval_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=min(
            candidates_per_document,
            record_count,
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (
        results.get("documents")
        or [[]]
    )[0]

    metadatas = (
        results.get("metadatas")
        or [[]]
    )[0]

    distances = (
        results.get("distances")
        or [[]]
    )[0]

    retrieved_chunks = []

    for (
        text,
        metadata,
        distance,
    ) in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved_chunks.append(
            {
                "filename": (
                    metadata["filename"]
                ),
                "document_hash": (
                    metadata[
                        "document_hash"
                    ]
                ),
                "page": (
                    metadata["page"]
                ),
                "chunk_number": (
                    metadata[
                        "chunk_number"
                    ]
                ),
                "text": text,
                "score": (
                    1.0
                    - float(distance)
                ),
            }
        )

    return retrieved_chunks


# -----------------------------
# 15. Search all documents
# -----------------------------

def retrieve_across_documents(
    retrieval_query: str,
    indexed_documents: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Search all uploaded documents.

    The strongest result from each document is kept
    before filling the remaining retrieval positions.
    """

    if not retrieval_query.strip():
        return []

    candidates_per_document = min(
        max(top_k, 3),
        10,
    )

    results_by_document = []

    for document in indexed_documents:
        document_results = (
            retrieve_from_collection(
                retrieval_query=(
                    retrieval_query
                ),
                document=document,
                candidates_per_document=(
                    candidates_per_document
                ),
            )
        )

        if document_results:
            results_by_document.append(
                document_results
            )

    effective_top_k = max(
        top_k,
        len(results_by_document),
    )

    selected = []
    selected_keys = set()

    # Keep one strong result from every document.
    for document_results in (
        results_by_document
    ):
        best_chunk = (
            document_results[0]
        )

        key = (
            best_chunk[
                "document_hash"
            ],
            best_chunk[
                "chunk_number"
            ],
        )

        selected.append(
            best_chunk
        )

        selected_keys.add(
            key
        )

    remaining_candidates = [
        chunk
        for document_results
        in results_by_document
        for chunk
        in document_results[1:]
    ]

    remaining_candidates.sort(
        key=lambda chunk: (
            chunk["score"]
        ),
        reverse=True,
    )

    for chunk in remaining_candidates:
        if (
            len(selected)
            >= effective_top_k
        ):
            break

        key = (
            chunk["document_hash"],
            chunk["chunk_number"],
        )

        if key not in selected_keys:
            selected.append(
                chunk
            )

            selected_keys.add(
                key
            )

    selected.sort(
        key=lambda chunk: (
            chunk["score"]
        ),
        reverse=True,
    )

    return selected[
        :effective_top_k
    ]


# -----------------------------
# 16. Follow-up retrieval query
# -----------------------------

def build_retrieval_query(
    current_question: str,
    chat_history: list[dict],
    previous_user_turns: int = 2,
) -> str:
    """
    Add recent user questions to the retrieval query
    so short follow-up questions have context.
    """

    previous_questions = [
        message["content"]
        for message
        in chat_history
        if message["role"] == "user"
    ]

    if (
        previous_questions
        and previous_questions[-1]
        == current_question
    ):
        previous_questions = (
            previous_questions[:-1]
        )

    recent_questions = (
        previous_questions[
            -previous_user_turns:
        ]
    )

    if not recent_questions:
        return current_question

    history_text = "\n".join(
        (
            f"Previous question: "
            f"{question}"
        )
        for question
        in recent_questions
    )

    return (
        f"{history_text}\n"
        f"Current follow-up question: "
        f"{current_question}"
    )


# -----------------------------
# 17. Chat history formatting
# -----------------------------

def format_chat_history(
    chat_history: list[dict],
) -> str:
    """
    Format recent chat messages for the prompt.
    """

    recent_messages = (
        chat_history[
            -MAX_CHAT_HISTORY_MESSAGES:
        ]
    )

    formatted_messages = []

    for message in recent_messages:
        if message["role"] == "user":
            role = "User"
        else:
            role = "Assistant"

        formatted_messages.append(
            f"{role}: "
            f"{message['content']}"
        )

    return "\n".join(
        formatted_messages
    )


# -----------------------------
# 18. Prompt construction
# -----------------------------

def create_prompt(
    question: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict],
) -> str:
    """
    Build a grounded multi-document prompt.
    """

    context_parts = [
        (
            f"[Document: "
            f"{chunk['filename']} | "
            f"Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )
        for chunk
        in retrieved_chunks
    ]

    context = "\n\n".join(
        context_parts
    )

    conversation = (
        format_chat_history(
            chat_history
        )
    )

    prompt = f"""
You are a precise academic research assistant.

Answer the current question using only the retrieved PDF context below.
Use the recent conversation only to understand references and follow-up
questions. Do not treat earlier assistant statements as source evidence.

Instructions:
- Give the direct answer first.
- Use clear and natural English.
- Compare documents when the question asks for comparison.
- Paraphrase the PDF context accurately.
- Do not use outside knowledge.
- Do not invent facts, methods, datasets, results, or conclusions.
- Cite important claims using: (filename, page 2).
- If documents disagree, clearly describe the disagreement.
- If the retrieved context is insufficient, say:
  "I could not find enough information in the retrieved PDF sections."
- Keep the answer concise unless the user asks for detail.

RECENT CONVERSATION:
{conversation or "No previous conversation."}

RETRIEVED PDF CONTEXT:
{context}

CURRENT QUESTION:
{question}

ANSWER:
"""

    return prompt.strip()


# -----------------------------
# 19. Ollama generation
# -----------------------------

def ask_ollama_model(
    question: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict],
) -> str:
    """
    Generate an answer using local Ollama or
    Ollama Cloud.
    """

    prompt = create_prompt(
        question=question,
        retrieved_chunks=(
            retrieved_chunks
        ),
        chat_history=chat_history,
    )

    response = ollama_client.chat(
        model=ACTIVE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful "
                    "multi-document research "
                    "assistant. Use only the "
                    "supplied PDF context as "
                    "factual evidence."
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
# 20. Source labels
# -----------------------------

def build_source_labels(
    retrieved_chunks: list[dict],
) -> list[str]:
    """
    Build unique filename and page labels.
    """

    labels = []
    seen = set()

    for chunk in retrieved_chunks:
        label = (
            f"{chunk['filename']} "
            f"— page {chunk['page']}"
        )

        if label not in seen:
            labels.append(
                label
            )

            seen.add(
                label
            )

    return labels


# -----------------------------
# 21. Display saved message
# -----------------------------

def display_message(
    message: dict,
):
    """
    Display a chat message and its sources.
    """

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        sources = (
            message.get(
                "sources",
                [],
            )
        )

        if sources:
            with st.expander(
                    f"Sources used ({len(sources)})",
                    expanded=False,
            ):
                for source in sources:
                    st.markdown(
                        f"- `{source}`"
                    )


# -----------------------------
# 22. Sidebar settings
# -----------------------------

st.sidebar.title(
    "Research Workspace"
)

st.sidebar.caption(
    "Multi-document conversational RAG"
)

with st.sidebar.expander(
    "System status",
    expanded=True,
):
    st.markdown(
        f"**Generation mode**  \n"
        f"{MODEL_MODE}"
    )

    st.markdown(
        f"**Language model**  \n"
        f"`{ACTIVE_MODEL}`"
    )

    st.markdown(
        "**Retriever**  \n"
        "Multi-document Chroma search"
    )

    st.markdown(
        "**Embedding model**  \n"
        "`all-MiniLM-L6-v2`"
    )

    st.markdown(
        f"**Vector database**  \n"
        f"{VECTOR_DB_MODE}"
    )

st.sidebar.divider()

st.sidebar.subheader(
    "Retrieval settings"
)

top_k = st.sidebar.slider(
    "Total chunks to retrieve",
    min_value=3,
    max_value=12,
    value=6,
    help=(
        "Higher values provide more context but "
        "may also include less relevant sections."
    ),
)

show_chunks = st.sidebar.checkbox(
    "Show retrieved text chunks",
    value=False,
    help=(
        "Display the exact PDF sections selected "
        "by the retrieval system."
    ),
)

st.sidebar.divider()

st.sidebar.info(
    "Public demo: avoid uploading confidential, "
    "private, or sensitive documents."
)

# -----------------------------
# 23. Multiple PDF upload
# -----------------------------
st.subheader(
    "1. Add your research documents"
)

st.caption(
    "Upload between one and five text-based PDFs. "
    "Each file may be up to 20 MB."
)

uploaded_pdfs = st.file_uploader(
    "Choose PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help=(
        f"Maximum {MAX_PDF_FILES} files. "
        f"Maximum {MAX_FILE_SIZE_MB} MB "
        "per file."
    ),
)

if not uploaded_pdfs:
    st.info(
        "Upload one or more PDFs "
        "to begin."
    )

    st.stop()

if (
    len(uploaded_pdfs)
    > MAX_PDF_FILES
):
    st.error(
        f"Upload no more than "
        f"{MAX_PDF_FILES} PDFs at a time. "
        f"You selected "
        f"{len(uploaded_pdfs)}."
    )

    st.stop()


# -----------------------------
# 24. Process all PDFs
# -----------------------------

with st.status(
    "Preparing your research workspace...",
    expanded=True,
) as processing_status:
    st.write(
        "Validating file sizes and checking "
        "for duplicate documents..."
    )

    st.write(
        "Extracting readable PDF text and "
        "creating document chunks..."
    )

    st.write(
        "Creating or reusing semantic "
        "ChromaDB indexes..."
    )

    (
        indexed_documents,
        processing_warnings,
    ) = process_uploaded_pdfs(
        uploaded_pdfs
    )

    if indexed_documents:
        processing_status.update(
            label=(
                f"Workspace ready — "
                f"{len(indexed_documents)} "
                "document(s) indexed."
            ),
            state="complete",
            expanded=False,
        )

    else:
        processing_status.update(
            label=(
                "No uploaded documents "
                "could be indexed."
            ),
            state="error",
            expanded=True,
        )

for warning in processing_warnings:
    st.warning(
        warning
    )

if not indexed_documents:
    st.error(
        "None of the uploaded PDFs "
        "could be indexed."
    )

    st.stop()


# -----------------------------
# 25. Reset chat when PDFs change
# -----------------------------

document_hashes = [
    document["document_hash"]
    for document
    in indexed_documents
]

current_signature = (
    create_document_set_signature(
        document_hashes
    )
)

if (
    st.session_state.document_set_signature
    != current_signature
):
    st.session_state.document_set_signature = (
        current_signature
    )

    st.session_state.messages = []


# -----------------------------
# 26. Document status
# -----------------------------

total_indexed_chunks = sum(
    document["chunk_count"]
    for document
    in indexed_documents
)

total_readable_pages = sum(
    document["page_count"]
    for document
    in indexed_documents
)

st.success(
    f"Research workspace ready with "
    f"{len(indexed_documents)} document(s)."
)

metric_col1, metric_col2, metric_col3 = (
    st.columns(
        3,
        gap="medium",
    )
)

with metric_col1:
    st.metric(
        "Documents",
        len(indexed_documents),
    )

with metric_col2:
    st.metric(
        "Readable pages",
        total_readable_pages,
    )

with metric_col3:
    st.metric(
        "Indexed chunks",
        total_indexed_chunks,
    )

with st.expander(
    "Document details",
    expanded=True,
):
    for document in indexed_documents:
        with st.container(
            border=True,
        ):
            document_col, status_col = (
                st.columns(
                    [4, 1],
                    gap="medium",
                )
            )

            with document_col:
                st.markdown(
                    f"**{document['filename']}**"
                )

                st.caption(
                    f"{document['page_count']} "
                    f"readable page(s) · "
                    f"{document['chunk_count']} "
                    f"indexed chunks · "
                    f"{document['file_size_mb']:.2f} MB"
                )

            with status_col:
                if document["index_reused"]:
                    st.success(
                        "Index reused"
                    )

                else:
                    st.info(
                        "Index created"
                    )

st.sidebar.metric(
    "Documents",
    len(indexed_documents),
)

st.sidebar.metric(
    "Readable pages",
    total_readable_pages,
)

st.sidebar.metric(
    "Indexed chunks",
    total_indexed_chunks,
)

# -----------------------------
# 27. Index and chat controls
# -----------------------------

if st.sidebar.button(
    "Rebuild uploaded indexes",
    use_container_width=True,
):
    chroma_client = (
        get_chroma_client(
            USE_PERSISTENT_CHROMA
        )
    )

    for document in indexed_documents:
        try:
            chroma_client.delete_collection(
                name=(
                    document[
                        "collection_name"
                    ]
                )
            )
        except Exception:
            pass

    st.cache_data.clear()
    st.session_state.messages = []

    st.rerun()

if st.sidebar.button(
    "Clear conversation",
    use_container_width=True,
):
    st.session_state.messages = []

    st.rerun()


# -----------------------------
# 28. Quick actions
# -----------------------------

st.subheader(
    "2. Choose a research task"
)

st.caption(
    "Use a prepared action or write a custom "
    "question in the research chat below."
)

col1, col2, col3 = st.columns(
    3
)

with col1:
    summarize_button = st.button(
        "Summarize documents",
        use_container_width=True,
    )

with col2:
    compare_button = st.button(
        "Compare documents",
        use_container_width=True,
    )

with col3:
    key_points_button = st.button(
        "Extract key points",
        use_container_width=True,
    )

quick_question = None

if summarize_button:
    quick_question = (
        "Summarize the main ideas, methods, "
        "findings, and conclusions across the "
        "uploaded documents. Clearly separate "
        "the documents."
    )

elif compare_button:
    quick_question = (
        "Compare the uploaded documents. "
        "Identify their main similarities, "
        "differences, methods, findings, "
        "and conclusions."
    )

elif key_points_button:
    quick_question = (
        "Extract the most important key points "
        "and findings from each uploaded "
        "document. Clearly label each document."
    )


# -----------------------------
# 29. Chat interface
# -----------------------------

st.subheader(
    "3. Research chat"
)

st.caption(
    "Ask questions, compare documents, or continue "
    "with contextual follow-up questions."
)

for saved_message in (
    st.session_state.messages
):
    display_message(
        saved_message
    )

chat_question = st.chat_input(
    "Ask a question or a follow-up "
    "about the uploaded PDFs"
)

final_question = (
    quick_question
    or chat_question
)


# -----------------------------
# 30. Retrieve and answer
# -----------------------------

if final_question:
    user_message = {
        "role": "user",
        "content": final_question,
    }

    st.session_state.messages.append(
        user_message
    )

    display_message(
        user_message
    )

    retrieval_query = (
        build_retrieval_query(
            current_question=(
                final_question
            ),
            chat_history=(
                st.session_state.messages
            ),
        )
    )

    with st.spinner(
        "Searching across the "
        "uploaded PDFs..."
    ):
        retrieved_chunks = (
            retrieve_across_documents(
                retrieval_query=(
                    retrieval_query
                ),
                indexed_documents=(
                    indexed_documents
                ),
                top_k=top_k,
            )
        )

    if not retrieved_chunks:
        assistant_message = {
            "role": "assistant",
            "content": (
                "No relevant PDF sections "
                "were found."
            ),
            "sources": [],
        }

        st.session_state.messages.append(
            assistant_message
        )

        display_message(
            assistant_message
        )

        st.stop()

    source_labels = (
        build_source_labels(
            retrieved_chunks
        )
    )

    best_score = (
        retrieved_chunks[0]["score"]
    )

    with st.chat_message(
        "assistant"
    ):
        if best_score < 0.15:
            st.warning(
                "The semantic match for this question "
                "is relatively weak. Review the retrieved "
                "sources before relying on the answer."
            )
        with st.spinner(
            f"Generating answer with "
            f"{ACTIVE_MODEL}..."
        ):
            try:
                answer = (
                    ask_ollama_model(
                        question=(
                            final_question
                        ),
                        retrieved_chunks=(
                            retrieved_chunks
                        ),
                        chat_history=(
                            st.session_state.messages[
                                :-1
                            ]
                        ),
                    )
                )

            except Exception as error:
                st.error(
                    f"The app could not connect "
                    f"using {MODEL_MODE}. "
                    "Check the Ollama service "
                    "or cloud credentials."
                )

                st.code(
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                st.stop()

        if (
            not answer
            or not answer.strip()
        ):
            st.error(
                "The selected model returned "
                "an empty answer."
            )

            st.stop()

        st.markdown(
            answer
        )

        st.markdown(
            "**Retrieved sources**"
        )

        source_columns = st.columns(
            2,
            gap="small",
        )

        for source_index, source in enumerate(
                source_labels
        ):
            source_column = source_columns[
                source_index % 2
                ]

            with source_column:
                with st.container(
                        border=True,
                ):
                    st.caption(
                        "PDF source"
                    )

                    st.write(
                        source
                    )

        with st.expander(
            "Retrieval details"
        ):
            st.write(
                f"Retrieval query: "
                f"{retrieval_query}"
            )

            st.write(
                f"Documents searched: "
                f"{len(indexed_documents)}"
            )

            st.write(
                f"Chunks returned: "
                f"{len(retrieved_chunks)}"
            )

            st.write(
                f"Best similarity score: "
                f"{best_score:.3f}"
            )

            st.write(
                f"Vector database mode: "
                f"{VECTOR_DB_MODE}"
            )

            if show_chunks:
                for (
                    index,
                    chunk,
                ) in enumerate(
                    retrieved_chunks,
                    start=1,
                ):
                    st.markdown(
                        f"**Result {index} — "
                        f"{chunk['filename']} — "
                        f"page {chunk['page']} — "
                        f"similarity "
                        f"{chunk['score']:.3f}**"
                    )

                    st.write(
                        chunk["text"]
                    )

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": source_labels,
    }

    st.session_state.messages.append(
        assistant_message
    )

st.divider()

footer_col1, footer_col2 = st.columns(
    [3, 2],
    gap="medium",
)

with footer_col1:
    st.caption(
        "AI PDF Research Assistant · "
        "Multi-document conversational RAG"
    )

with footer_col2:
    st.caption(
        "Verify important findings against "
        "the original documents."
    )