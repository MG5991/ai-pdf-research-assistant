import os
import re

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

else:
    MODEL_MODE = "Local Ollama"
    ACTIVE_MODEL = LOCAL_MODEL

    ollama_client = Client(
        host="http://localhost:11434",
    )


# -----------------------------
# 2. Streamlit page setup
# -----------------------------

st.set_page_config(
    page_title="AI PDF Research Assistant",
    page_icon="📄",
    layout="wide",
)

st.title(
    "AI PDF Research Assistant — Semantic RAG"
)

st.write(
    "Upload a PDF and ask questions about it. "
    "The app uses semantic embeddings to retrieve "
    "relevant sections before generating an answer."
)


# -----------------------------
# 3. Load embedding model
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Load the SentenceTransformers embedding model.

    Streamlit caches the model so it is not reloaded
    every time the application reruns.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# -----------------------------
# 4. Clean extracted text
# -----------------------------

def clean_text(text: str) -> str:
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
# 5. Read PDF page by page
# -----------------------------

def read_pdf_with_pages(
    uploaded_file,
) -> list[dict]:
    """
    Extract readable text from each PDF page.

    Each page is stored with:
    - page number
    - extracted text
    """

    pdf_reader = PdfReader(
        uploaded_file
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
# 6. Split pages into chunks
# -----------------------------

def split_pages_into_chunks(
    pages: list[dict],
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
                        "page": page_number,
                        "text": chunk_text,
                    }
                )

            start += step

    return chunks


# -----------------------------
# 7. Create semantic embeddings
# -----------------------------

@st.cache_data(show_spinner=False)
def create_chunk_embeddings(
    chunk_texts: tuple[str, ...],
):
    """
    Convert document chunks into semantic vectors.

    Streamlit caches the vectors using the tuple of
    chunk texts as the cache key.
    """

    embedding_model = load_embedding_model()

    embeddings = (
        embedding_model.encode_document(
            list(chunk_texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )

    return embeddings


# -----------------------------
# 8. Retrieve semantic matches
# -----------------------------

def retrieve_relevant_chunks(
    question: str,
    chunks: list[dict],
    chunk_embeddings,
    top_k: int = 5,
) -> list[dict]:
    """
    Embed the question and compare it with every
    document-chunk embedding.

    Because the embeddings are normalized, their
    dot product is equivalent to cosine similarity.
    """

    if not question.strip():
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

    similarities = (
        chunk_embeddings @ question_embedding
    )

    number_to_retrieve = min(
        top_k,
        len(chunks),
    )

    top_indices = similarities.argsort()[
        -number_to_retrieve:
    ][::-1]

    retrieved_chunks = []

    for index in top_indices:
        chunk = chunks[index].copy()

        chunk["score"] = float(
            similarities[index]
        )

        retrieved_chunks.append(chunk)

    return retrieved_chunks


# -----------------------------
# 9. Create grounded prompt
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
            f"[Page {chunk['page']}]\n"
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
# 10. Ask Ollama
# -----------------------------

def ask_ollama_model(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Generate an answer through either local Ollama
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
# 11. Sidebar settings
# -----------------------------

st.sidebar.header("RAG Settings")

st.sidebar.caption(
    f"Generation mode: {MODEL_MODE}"
)

st.sidebar.caption(
    f"Language model: {ACTIVE_MODEL}"
)

st.sidebar.caption(
    "Retriever: Semantic embeddings"
)

st.sidebar.caption(
    "Embedding model: all-MiniLM-L6-v2"
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
# 12. PDF upload
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

st.success(
    f"Uploaded: {uploaded_pdf.name}"
)


# -----------------------------
# 13. Process and index PDF
# -----------------------------

try:
    with st.spinner("Reading PDF..."):
        pages = read_pdf_with_pages(
            uploaded_pdf
        )

    if not pages:
        st.error(
            "No readable text was found. "
            "The PDF may contain scanned images."
        )

        st.stop()

    with st.spinner(
        "Splitting PDF into chunks..."
    ):
        chunks = split_pages_into_chunks(
            pages
        )

    if not chunks:
        st.error(
            "The app could not create "
            "searchable text chunks."
        )

        st.stop()

    chunk_texts = tuple(
        chunk["text"]
        for chunk in chunks
    )

    with st.spinner(
        "Creating semantic embeddings..."
    ):
        chunk_embeddings = (
            create_chunk_embeddings(
                chunk_texts
            )
        )

except Exception as error:
    st.error(
        "The PDF could not be processed."
    )

    st.code(
        f"{type(error).__name__}: {error}"
    )

    st.stop()


st.info(
    f"Extracted {len(pages)} readable pages, "
    f"created {len(chunks)} chunks, and generated "
    f"{len(chunk_embeddings)} semantic embeddings."
)


# -----------------------------
# 14. Quick actions
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
# 15. Custom question
# -----------------------------

question = st.text_input(
    "Or ask your own question about the PDF:",
    placeholder=(
        "For example: "
        "What research gap does this paper address?"
    ),
)


# -----------------------------
# 16. Decide final question
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
# 17. Retrieve context and answer
# -----------------------------

if final_question:
    with st.spinner(
        "Searching by semantic meaning..."
    ):
        retrieved_chunks = (
            retrieve_relevant_chunks(
                question=final_question,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
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
        f"semantically retrieved chunks from "
        f"page(s): {source_pages_text}."
    )

    if best_score < 0.15:
        st.warning(
            "Semantic retrieval confidence is low. "
            "The retrieved sections may not strongly "
            "match the question."
        )

    with st.expander(
        "How semantic RAG worked"
    ):
        st.markdown(
            """
1. The app split the PDF into overlapping chunks.
2. It converted every chunk into a semantic embedding.
3. It converted your question into another embedding.
4. It compared the question vector with all chunk vectors.
5. It selected the chunks with the highest similarity.
6. It sent those chunks to the selected Ollama model.
7. Ollama generated an answer using that context.
            """
        )

        st.write(
            f"Retrieved source pages: "
            f"{source_pages_text}"
        )

        st.write(
            f"Best semantic similarity score: "
            f"{best_score:.3f}"
        )

        st.write(
            f"Embedding dimensions: "
            f"{chunk_embeddings.shape[1]}"
        )

        st.write(
            f"Embedding model: "
            f"{EMBEDDING_MODEL_NAME}"
        )

        st.write(
            f"Generation mode: {MODEL_MODE}"
        )

        st.write(
            f"Language model: {ACTIVE_MODEL}"
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
                    f"Semantic score: "
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