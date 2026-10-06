import os
import tempfile
import numpy as np
import streamlit as st
import ollama
import chromadb

from PIL import Image
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Automated Tumor Detection",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# 2. APPLICATION TITLE
# =========================================================

st.title("🧠 Automated Tumor Detection from Medical Scans")

st.markdown(
    """
    ### Multi-Model AI Medical Imaging System

    This project combines:

    - 🧠 Deep Learning
    - 👁️ Computer Vision
    - 🤖 Ollama Local AI
    - 🔎 Sentence Transformers
    - 🗄️ ChromaDB
    - 📄 Medical PDF Knowledge
    - 🌐 Streamlit
    """
)

st.warning(
    "⚠️ Educational/research prototype only. "
    "This system is NOT a medical diagnosis tool. "
    "Results must be reviewed by a qualified medical professional."
)


# =========================================================
# 3. LOAD SENTENCE TRANSFORMER
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# =========================================================
# 4. CHROMADB
# =========================================================

@st.cache_resource
def load_database():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="medical_knowledge"
    )

    return collection


collection = load_database()


# =========================================================
# 5. ADD DEFAULT MEDICAL KNOWLEDGE
# =========================================================

def add_default_knowledge():

    documents = [
        """
        MRI is commonly used for brain imaging because it provides
        detailed information about soft tissues.
        """,

        """
        CT scans use X-rays to create cross-sectional images of
        structures inside the body.
        """,

        """
        Tumor segmentation means identifying the pixels belonging
        to a suspected tumor region in a medical image.
        """,

        """
        U-Net is a popular deep learning architecture used for
        biomedical image segmentation.
        """,

        """
        Medical AI predictions should be validated using appropriate
        datasets and reviewed by qualified healthcare professionals.
        """,

        """
        Tumors can have different shapes, sizes, textures and
        intensities depending on tumor type and imaging modality.
        """
    ]

    ids = [
        "mri",
        "ct",
        "segmentation",
        "unet",
        "validation",
        "tumor_features"
    ]

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )


add_default_knowledge()


# =========================================================
# 6. PDF KNOWLEDGE UPLOAD
# =========================================================

def read_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def add_pdf_to_database(pdf_file):

    text = read_pdf(pdf_file)

    if not text.strip():

        return False

    # Break PDF into smaller chunks
    chunks = []

    chunk_size = 1000

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        if chunk.strip():

            chunks.append(chunk)

    for i, chunk in enumerate(chunks):

        embedding = embedding_model.encode(
            chunk
        ).tolist()

        collection.upsert(
            ids=[f"pdf_{pdf_file.name}_{i}"],
            documents=[chunk],
            embeddings=[embedding]
        )

    return True


# =========================================================
# 7. RETRIEVE KNOWLEDGE
# =========================================================

def retrieve_knowledge(question):

    embedding = embedding_model.encode(
        question
    ).tolist()

    result = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    documents = result.get(
        "documents",
        [[]]
    )

    if documents and documents[0]:

        return "\n\n".join(
            documents[0]
        )

    return "No relevant medical information found."


# =========================================================
# 8. IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    # Resize for consistent processing
    image = image.resize(
        (512, 512)
    )

    return image


# =========================================================
# 9. DEMONSTRATION HEATMAP
# =========================================================

def create_demo_heatmap(image):

    """
    IMPORTANT:
    This is only a visualization demonstration.
    It is NOT a tumor segmentation algorithm.
    A trained U-Net/nnU-Net model should replace this function
    for actual segmentation research.
    """

    img = np.array(image.convert("L"))

    # Normalize intensity
    normalized = (
        img - img.min()
    ) / (
        img.max() - img.min() + 1e-8
    )

    # Highlight unusually bright regions
    mask = normalized > 0.80

    output = np.array(
        image.convert("RGB")
    ).copy()

    # Overlay highlighted pixels
    output[mask] = [
        255,
        0,
        0
    ]

    return Image.fromarray(output)


# =========================================================
# 10. OLLAMA AI ANALYSIS
# =========================================================

def analyze_with_ollama(
    image_path,
    question,
    knowledge
):

    prompt = f"""
You are an AI assistant in an educational medical imaging project.

Analyze the uploaded medical scan as an AI research assistant.

DO NOT provide a medical diagnosis.

DO NOT claim that a tumor definitely exists.

Describe only visual observations and uncertainty.

Relevant knowledge:

{knowledge}

User question:

{question}

Return the response using exactly these sections:

SCAN OBSERVATION
Describe general visible characteristics.

SUSPICIOUS REGION
If an area appears potentially abnormal, describe its
approximate location and appearance without diagnosing it.

AI INTERPRETATION
Explain what the AI noticed.

CONFIDENCE AND LIMITATIONS
Explain why the result is uncertain.

RECOMMENDED NEXT STEP
Recommend review by an appropriate medical professional.
"""

    response = ollama.chat(

        model="llama3.2-vision",

        messages=[
            {
                "role": "user",

                "content": prompt,

                "images": [
                    image_path
                ]
            }
        ]
    )

    return response["message"]["content"]


# =========================================================
# 11. SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Project Configuration")

st.sidebar.success(
    "Streamlit: Active"
)

st.sidebar.success(
    "Ollama: Local AI"
)

st.sidebar.success(
    "ChromaDB: Knowledge Base"
)

st.sidebar.success(
    "Sentence Transformer: Embeddings"
)

st.sidebar.info(
    "Vision Model: llama3.2-vision"
)


# =========================================================
# 12. PDF UPLOAD
# =========================================================

st.sidebar.subheader(
    "📚 Medical Knowledge"
)

pdf_file = st.sidebar.file_uploader(
    "Upload a medical reference PDF",
    type=["pdf"]
)

if pdf_file:

    if st.sidebar.button(
        "Add PDF to Knowledge Base"
    ):

        with st.spinner(
            "Reading PDF..."
        ):

            success = add_pdf_to_database(
                pdf_file
            )

        if success:

            st.sidebar.success(
                "PDF added successfully!"
            )

        else:

            st.sidebar.error(
                "Could not read PDF."
            )


# =========================================================
# 13. SCAN UPLOAD
# =========================================================

st.header("📤 Upload Medical Scan")

uploaded_file = st.file_uploader(
    "Upload MRI / CT / medical image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ]
)


# =========================================================
# 14. QUESTION
# =========================================================

question = st.text_area(
    "Ask the AI about the scan",

    value=(
        "Analyze this scan and identify whether "
        "there is any suspicious region that "
        "requires further medical evaluation."
    )
)


# =========================================================
# 15. MAIN ANALYSIS
# =========================================================

if uploaded_file:

    original_image = Image.open(
        uploaded_file
    )

    image = preprocess_image(
        original_image
    )

    st.divider()

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # ORIGINAL IMAGE
    # -----------------------------------------------------

    with col1:

        st.subheader(
            "🖼️ Original Scan"
        )

        st.image(
            image,
            use_container_width=True
        )


    # -----------------------------------------------------
    # DEMO SEGMENTATION
    # -----------------------------------------------------

    with col2:

        st.subheader(
            "🎯 Region Visualization"
        )

        demo_image = create_demo_heatmap(
            image
        )

        st.image(
            demo_image,
            caption=(
                "Demonstration visualization — "
                "not a medical segmentation result"
            ),
            use_container_width=True
        )


    st.divider()

    # -----------------------------------------------------
    # ANALYZE BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔍 Analyze Scan",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "AI is analyzing the medical scan..."
        ):

            try:

                # Save uploaded image
                extension = os.path.splitext(
                    uploaded_file.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension
                ) as temp:

                    image.save(
                        temp.name
                    )

                    image_path = temp.name


                # Retrieve relevant knowledge
                knowledge = retrieve_knowledge(
                    question
                )


                # Ollama analysis
                result = analyze_with_ollama(
                    image_path,
                    question,
                    knowledge
                )


                # -------------------------------------------------
                # DISPLAY RESULT
                # -------------------------------------------------

                st.subheader(
                    "🤖 AI Analysis Report"
                )

                st.markdown(
                    result
                )


                # -------------------------------------------------
                # KNOWLEDGE USED
                # -------------------------------------------------

                with st.expander(
                    "📚 Medical Knowledge Used"
                ):

                    st.write(
                        knowledge
                    )


                # Delete temporary image
                os.remove(
                    image_path
                )


            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )

                st.info(
                    """
                    Check the following:

                    1. Ollama is installed.
                    2. Ollama is running.
                    3. The vision model is installed.
                    4. Run:
                       
                       ollama pull llama3.2-vision
                    """
                )


# =========================================================
# 16. PROJECT WORKFLOW
# =========================================================

st.divider()

st.header(
    "🔬 Project Workflow"
)

workflow = """

### Step 1 — Data Collection

Medical scan datasets containing healthy and abnormal cases
are collected from appropriate research sources.

↓

### Step 2 — Preprocessing

Images are resized and normalized before being given to the
deep learning model.

↓

### Step 3 — Deep Learning

A model such as **U-Net / nnU-Net** can be trained using
labeled medical scans.

The model learns image features such as:

- Shape
- Texture
- Intensity
- Boundaries
- Spatial patterns

↓

### Step 4 — Tumor Segmentation

The trained segmentation model produces a pixel-level mask
of the suspected region.

↓

### Step 5 — Ollama AI

The local Ollama vision model analyzes the scan and produces
a human-readable explanation.

↓

### Step 6 — ChromaDB + RAG

Relevant information from medical reference documents is
retrieved using Sentence Transformers and ChromaDB.

↓

### Step 7 — Streamlit Report

The final AI observations and segmentation visualization
are displayed through the Streamlit interface.
"""

st.markdown(
    workflow
)


# =========================================================
# 17. FOOTER
# =========================================================

st.divider()

st.caption(
    "Automated Tumor Detection from Medical Scans | "
    "Multi-Model AI Research Prototype"
)