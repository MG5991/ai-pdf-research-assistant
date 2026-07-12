import re

import streamlit as st
from ollama import chat
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Local Ollama model installed on this computer
LOCAL_MODEL = "llama3.2:3b"


# -----------------------------
# 1. Streamlit page setup
# -----------------------------

st.set_page_config(
    page_title="AI PDF Research Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("AI PDF Research Assistant — Local RAG")
st.write(
    "Upload a PDF and ask questions about it. "
    "The app retrieves relevant sections and uses a local Ollama model to answer."
)


# -----------------------------
# 2. Clean extracted text
# -----------------------------

def clean_text(text: str) -> str:
    """
    Remove repeated spaces, tabs, and line breaks from extracted PDF text.
    """

    cleaned_text = re.sub(r"\s+", " ", text)
    return cleaned_text.strip()


# -----------------------------
# 3. Read PDF page by page
# -----------------------------

def read_pdf_with_pages(uploaded_file) -> list[dict]:
    """
    Read an uploaded PDF and return its readable pages.

    Each page is stored as a dictionary containing:
    - page: page number
    - text: extracted and cleaned page text
    """

    pdf_reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
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
# 4. Split pages into chunks
# -----------------------------

def split_pages_into_chunks(
    pages: list[dict],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """
    Split each PDF page into smaller overlapping text chunks.

    chunk_size:
        Approximate number of characters in each chunk.

    overlap:
        Number of characters repeated between neighboring chunks.
        This reduces the chance of losing context at chunk boundaries.
    """

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []

    for page in pages:
        page_number = page["page"]
        page_text = page["text"]

        start = 0
        step = chunk_size - overlap

        while start < len(page_text):
            end = start + chunk_size
            chunk_text = page_text[start:end].strip()

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
# 5. Build TF-IDF search index
# -----------------------------

def build_search_index(chunks: list[dict]):
    """
    Convert PDF chunks into TF-IDF vectors.

    The vectorizer learns important words and phrases from the PDF.
    The resulting vectors make the chunks searchable.
    """

    if not chunks:
        raise ValueError("Cannot build a search index because there are no chunks.")

    chunk_texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    chunk_vectors = vectorizer.fit_transform(chunk_texts)

    return vectorizer, chunk_vectors


# -----------------------------
# 6. Retrieve relevant chunks
# -----------------------------

def retrieve_relevant_chunks(
    question: str,
    chunks: list[dict],
    vectorizer,
    chunk_vectors,
    top_k: int = 5,
) -> list[dict]:
    """
    Compare the user's question with every PDF chunk.

    Return the chunks with the highest cosine-similarity scores.
    """

    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        chunk_vectors,
    ).flatten()

    number_to_retrieve = min(top_k, len(chunks))

    top_indices = similarities.argsort()[-number_to_retrieve:][::-1]

    retrieved_chunks = []

    for index in top_indices:
        chunk = chunks[index].copy()
        chunk["score"] = float(similarities[index])
        retrieved_chunks.append(chunk)

    return retrieved_chunks


# -----------------------------
# 7. Create prompt for local model
# -----------------------------

def create_prompt(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Build a grounded prompt using only retrieved PDF chunks.
    """

    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a precise academic research assistant.

Answer the user's question using only the PDF context below.

Instructions:
- Give the direct answer first.
- Use clear, natural English.
- Do not copy poorly written sentences from the PDF word for word.
- Paraphrase the information accurately.
- Do not use outside knowledge.
- Do not invent facts, methods, datasets, results, or conclusions.
- Avoid unnecessary repetition and vague phrases such as
  "it can be reasonably inferred."
- Cite the relevant page number after each important claim,
  using the format: (page 2).
- If the context is insufficient, say:
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
# 8. Ask local Ollama model
# -----------------------------

def ask_local_model(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Send the retrieved PDF context and question to Ollama.
    """

    prompt = create_prompt(question, retrieved_chunks)

    response = chat(
        model=LOCAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful document-analysis assistant. "
                    "Use only the PDF context supplied by the user."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0.1,
        },
    )

    return response.message.content


# -----------------------------
# 9. Sidebar settings
# -----------------------------

st.sidebar.header("RAG Settings")
st.sidebar.caption(f"Local model: {LOCAL_MODEL}")

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
# 10. PDF upload
# -----------------------------

uploaded_pdf = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
)

if uploaded_pdf is None:
    st.warning("Upload a PDF to begin.")
    st.stop()

st.success(f"Uploaded: {uploaded_pdf.name}")


# -----------------------------
# 11. Process PDF
# -----------------------------

try:
    with st.spinner("Reading PDF..."):
        pages = read_pdf_with_pages(uploaded_pdf)

    if not pages:
        st.error(
            "No readable text was found. "
            "The PDF may contain scanned images instead of selectable text."
        )
        st.stop()

    with st.spinner("Splitting PDF into chunks..."):
        chunks = split_pages_into_chunks(pages)

    if not chunks:
        st.error("The app could not create searchable text chunks.")
        st.stop()

    with st.spinner("Building local search index..."):
        vectorizer, chunk_vectors = build_search_index(chunks)

except Exception as error:
    st.error("The PDF could not be processed.")
    st.code(str(error))
    st.stop()


st.info(
    f"Extracted {len(pages)} readable pages "
    f"and created {len(chunks)} searchable chunks."
)


# -----------------------------
# 12. Quick actions
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
# 13. Custom question
# -----------------------------

question = st.text_input(
    "Or ask your own question about the PDF:",
    placeholder="For example: What methodology did the paper use?",
)


# -----------------------------
# 14. Decide final question
# -----------------------------

final_question = None

if summarize_button:
    final_question = (
        "Summarize the main ideas, research approach, results, "
        "and conclusions of this PDF."
    )

elif beginner_button:
    final_question = (
        "Explain the main ideas of this PDF in simple, "
        "beginner-friendly language."
    )

elif key_points_button:
    final_question = (
        "Extract the most important key points, findings, "
        "and conclusions from this PDF."
    )

elif question.strip():
    final_question = question.strip()


# -----------------------------
# 15. Retrieve context and answer
# -----------------------------

if final_question:
    with st.spinner("Searching relevant PDF chunks..."):
        retrieved_chunks = retrieve_relevant_chunks(
            question=final_question,
            chunks=chunks,
            vectorizer=vectorizer,
            chunk_vectors=chunk_vectors,
            top_k=top_k,
        )

    if not retrieved_chunks:
        st.error("No relevant PDF chunks were found.")
        st.stop()

    source_pages = sorted(
        {chunk["page"] for chunk in retrieved_chunks}
    )

    source_pages_text = ", ".join(
        str(page) for page in source_pages
    )

    best_score = retrieved_chunks[0]["score"]

    st.caption(
        f"Using {len(retrieved_chunks)} retrieved chunks "
        f"from page(s): {source_pages_text}."
    )

    if best_score < 0.05:
        st.warning(
            "Retrieval confidence is low. "
            "The search did not find a strong textual match for this question."
        )

    with st.expander("How RAG worked for this question"):
        st.markdown(
            """
1. The app took your question.
2. It compared the question with every PDF chunk using TF-IDF.
3. It ranked the chunks using cosine similarity.
4. It selected the most relevant chunks.
5. It sent those chunks to the local Ollama model.
6. The local model generated an answer using the retrieved context.
            """
        )

        st.write(f"Retrieved source pages: {source_pages_text}")
        st.write(f"Best similarity score: {best_score:.3f}")
        st.write(f"Local language model: {LOCAL_MODEL}")

    if show_chunks:
        with st.expander("Show retrieved chunks"):
            for index, chunk in enumerate(
                retrieved_chunks,
                start=1,
            ):
                st.markdown(
                    f"**Chunk {index} — Page {chunk['page']} — "
                    f"Similarity score: {chunk['score']:.3f}**"
                )
                st.write(chunk["text"])

    with st.spinner(
        f"Generating answer locally with {LOCAL_MODEL}..."
    ):
        try:
            answer = ask_local_model(
                final_question,
                retrieved_chunks,
            )

        except Exception as error:
            st.error(
                "The app could not connect to the local Ollama model. "
                "Make sure Ollama is running and the model is installed."
            )
            st.code(str(error))
            st.stop()

    if not answer or not answer.strip():
        st.error("The local model returned an empty answer.")
        st.stop()

    st.subheader("Answer")
    st.write(answer)
    st.markdown(f"**Retrieved source pages:** {source_pages_text}")