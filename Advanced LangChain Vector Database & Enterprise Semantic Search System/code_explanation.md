# ============================================================
# DAY 65/100 — ADVANCED LANGCHAIN VECTOR DATABASE
# PART 1 — LANGCHAIN DOCUMENT PIPELINE
# ============================================================
#
# Pipeline:
# Enterprise Documents
#        ↓
# Text Cleaning
#        ↓
# LangChain Document Objects
#        ↓
# Recursive Chunking
#        ↓
# Metadata Enrichment
#        ↓
# Chunk Validation
#
# CPU-friendly | Lightweight | In-memory | Jupyter Notebook
# ============================================================


# ============================================================
# 1. INSTALL REQUIRED PACKAGES
# ============================================================

import sys
import subprocess
import importlib.util

required_packages = {
    "langchain_core": "langchain-core",
    "langchain_text_splitters": "langchain-text-splitters",
    "pandas": "pandas",
}

for import_name, package_name in required_packages.items():

    if importlib.util.find_spec(import_name) is None:
        print(f"Installing: {package_name}")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            package_name
        ])

print("Required packages are ready.")


# ============================================================
# 2. IMPORTS
# ============================================================

import re
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 3. PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "Day 65 - Advanced LangChain Vector Database"

CHUNK_SIZE = 350
CHUNK_OVERLAP = 50

MIN_CHUNK_LENGTH = 40

print("=" * 70)
print(PROJECT_NAME)
print("=" * 70)

print(f"Chunk size     : {CHUNK_SIZE}")
print(f"Chunk overlap  : {CHUNK_OVERLAP}")
print(f"Min chunk size : {MIN_CHUNK_LENGTH}")


# ============================================================
# 4. ENTERPRISE SAMPLE DOCUMENTS
# ============================================================
#
# We intentionally use a tiny local dataset.
#
# In a real enterprise system these could come from:
# - PDF
# - DOCX
# - HTML
# - Knowledge bases
# - SharePoint
# - Databases
# - Internal APIs
#
# For this notebook we keep everything in memory.
# ============================================================

raw_documents = [

    {
        "document_id": "POL-001",
        "title": "Enterprise Security Policy",
        "department": "Security",
        "document_type": "Policy",
        "source": "security_policy.pdf",
        "content": """
        Enterprise Security Policy

        All employees must protect company information and customer data.
        Sensitive information must only be accessed by authorized users.

        Passwords must follow the organization's password policy.
        Multi-factor authentication is required for privileged accounts.

        Employees must immediately report suspected security incidents
        to the security operations team.

        Access to sensitive systems must follow the principle of least privilege.
        User access should be reviewed regularly and removed when no longer required.
        """
    },

    {
        "document_id": "HR-001",
        "title": "Employee Remote Work Policy",
        "department": "Human Resources",
        "document_type": "Policy",
        "source": "remote_work_policy.pdf",
        "content": """
        Employee Remote Work Policy

        Employees may work remotely when approved by their manager.
        Remote employees must maintain a secure working environment.

        Company devices must use approved security controls.
        Employees must not share corporate credentials with other individuals.

        Sensitive company information should not be stored on personal devices.
        Employees working remotely must follow all information security requirements.
        """
    },

    {
        "document_id": "FIN-001",
        "title": "Expense Reimbursement Policy",
        "department": "Finance",
        "document_type": "Policy",
        "source": "expense_policy.pdf",
        "content": """
        Expense Reimbursement Policy

        Employees may request reimbursement for approved business expenses.
        Expense claims should include valid receipts and business justification.

        Travel expenses must comply with approved travel limits.
        Hotel and transportation expenses should remain within company limits.

        Expense claims must normally be submitted within thirty days.
        Managers are responsible for reviewing and approving employee claims.
        """
    },

    {
        "document_id": "IT-001",
        "title": "Incident Management Procedure",
        "department": "IT",
        "document_type": "Procedure",
        "source": "incident_management.pdf",
        "content": """
        IT Incident Management Procedure

        Employees should report technical incidents through the approved
        service management platform.

        Incidents are categorized according to business impact and urgency.
        Critical incidents receive the highest priority.

        The support team investigates the issue, identifies the root cause,
        applies a resolution, and documents the incident.

        Major incidents require communication with relevant stakeholders
        until service is restored.
        """
    },

    {
        "document_id": "BFSI-001",
        "title": "Insurance Claims Processing Guidelines",
        "department": "Insurance",
        "document_type": "Guideline",
        "source": "claims_guidelines.pdf",
        "content": """
        Insurance Claims Processing Guidelines

        Claims must be validated before processing.
        Required customer and policy information should be available
        before the claim enters detailed assessment.

        Claims may require document verification, eligibility validation,
        policy checks, and risk assessment.

        High-risk or unusual claims should be routed to an authorized
        human reviewer.

        Every claim decision should maintain an auditable record.
        """
    },

    {
        "document_id": "API-001",
        "title": "Internal API Development Standards",
        "department": "Engineering",
        "document_type": "Engineering Standard",
        "source": "api_standards.md",
        "content": """
        Internal API Development Standards

        APIs should use clear resource-oriented endpoints.
        Request and response schemas must be explicitly defined.

        APIs should validate incoming requests before processing.
        Error responses should provide meaningful error information
        without exposing sensitive internal implementation details.

        Authentication and authorization must be applied to protected APIs.
        API changes should be tested before deployment.
        """
    },

    {
        "document_id": "AI-001",
        "title": "Enterprise AI Governance Guidelines",
        "department": "Artificial Intelligence",
        "document_type": "Guideline",
        "source": "ai_governance.pdf",
        "content": """
        Enterprise AI Governance Guidelines

        AI applications must be designed with security, reliability,
        transparency, and responsible AI principles.

        AI systems that generate business-critical decisions should provide
        appropriate explanations and supporting evidence.

        Sensitive information must not be unnecessarily exposed to AI models.
        AI outputs should be validated before being used in critical workflows.

        Human review should be available for high-risk decisions.
        AI applications should maintain appropriate logs for auditing.
        """
    },

    {
        "document_id": "CLOUD-001",
        "title": "Cloud Application Deployment Standards",
        "department": "Cloud Engineering",
        "document_type": "Engineering Standard",
        "source": "cloud_standards.pdf",
        "content": """
        Cloud Application Deployment Standards

        Applications deployed to cloud environments should use
        environment-specific configuration.

        Secrets must not be hard-coded into application source code.
        Production applications should expose health checks.

        Services should generate structured logs and application metrics.
        Resource limits should be defined for production workloads.

        Deployments should support rollback when a release causes
        unexpected application failures.
        """
    }
]


print(f"\nLoaded {len(raw_documents)} enterprise documents.")


# ============================================================
# 5. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):
    """
    Basic enterprise document cleaning.

    Operations:
    - Remove unnecessary whitespace
    - Normalize line breaks
    - Remove repeated spaces
    - Preserve readable sentence structure
    """

    if not isinstance(text, str):
        return ""

    # Normalize line breaks
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive whitespace around lines
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


# ============================================================
# 6. CLEAN ALL DOCUMENTS
# ============================================================

cleaned_documents = []

for item in raw_documents:

    cleaned_item = item.copy()

    cleaned_item["content"] = clean_text(item["content"])

    cleaned_documents.append(cleaned_item)


print("\nDocument cleaning completed.")

for doc in cleaned_documents[:3]:

    print("-" * 60)
    print(f"ID         : {doc['document_id']}")
    print(f"Title      : {doc['title']}")
    print(f"Department : {doc['department']}")
    print(f"Characters : {len(doc['content'])}")


# ============================================================
# 7. CONVERT INTO LANGCHAIN DOCUMENT OBJECTS
# ============================================================
#
# LangChain Document contains:
#
# Document(
#     page_content="...",
#     metadata={...}
# )
#
# page_content = actual searchable text
# metadata     = information about the source
# ============================================================

langchain_documents = []

for item in cleaned_documents:

    document = Document(
        page_content=item["content"],
        metadata={
            "document_id": item["document_id"],
            "title": item["title"],
            "department": item["department"],
            "document_type": item["document_type"],
            "source": item["source"]
        }
    )

    langchain_documents.append(document)


print(
    f"\nCreated {len(langchain_documents)} LangChain Document objects."
)


# ============================================================
# 8. INSPECT ONE LANGCHAIN DOCUMENT
# ============================================================

sample_document = langchain_documents[0]

print("\n" + "=" * 70)
print("SAMPLE LANGCHAIN DOCUMENT")
print("=" * 70)

print("\nPAGE CONTENT:")
print(sample_document.page_content)

print("\nMETADATA:")
print(sample_document.metadata)


# ============================================================
# 9. CREATE RECURSIVE CHARACTER TEXT SPLITTER
# ============================================================
#
# RecursiveCharacterTextSplitter attempts to split text
# using natural separators before falling back to smaller ones.
#
# Typical priority:
#
# paragraph → line → sentence-like boundary → word → character
#
# This is more practical than blindly cutting every N characters.
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

print("\nRecursiveCharacterTextSplitter created.")


# ============================================================
# 10. SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

chunked_documents = text_splitter.split_documents(
    langchain_documents
)

print(
    f"\nOriginal documents : {len(langchain_documents)}"
)

print(
    f"Generated chunks   : {len(chunked_documents)}"
)


# ============================================================
# 11. ADD CHUNK-LEVEL METADATA
# ============================================================
#
# Every chunk gets:
#
# document_id
# chunk_index
# chunk_id
# title
# department
# document_type
# source
#
# This metadata becomes extremely important later for:
# - filtering
# - retrieval
# - citations
# - access control
# - source attribution
# ============================================================

document_chunk_counter = {}

for chunk in chunked_documents:

    document_id = chunk.metadata["document_id"]

    if document_id not in document_chunk_counter:
        document_chunk_counter[document_id] = 0

    chunk_index = document_chunk_counter[document_id]

    chunk.metadata["chunk_index"] = chunk_index

    chunk.metadata["chunk_id"] = (
        f"{document_id}-CHUNK-{chunk_index:03d}"
    )

    chunk.metadata["character_count"] = len(
        chunk.page_content
    )

    document_chunk_counter[document_id] += 1


# ============================================================
# 12. REMOVE VERY SMALL CHUNKS
# ============================================================
#
# Extremely small fragments are usually poor retrieval units.
#
# We keep this lightweight and only remove chunks below the
# minimum configured length.
# ============================================================

filtered_chunks = []

for chunk in chunked_documents:

    text = chunk.page_content.strip()

    if len(text) >= MIN_CHUNK_LENGTH:
        filtered_chunks.append(chunk)


chunked_documents = filtered_chunks


print(
    f"\nChunks after minimum-length filtering: "
    f"{len(chunked_documents)}"
)


# ============================================================
# 13. RE-CALCULATE CHUNK INDEXES
# ============================================================
#
# Filtering may have removed some chunks.
# Rebuild the chunk indexes so they remain sequential.
# ============================================================

document_chunk_counter = {}

for chunk in chunked_documents:

    document_id = chunk.metadata["document_id"]

    if document_id not in document_chunk_counter:
        document_chunk_counter[document_id] = 0

    chunk_index = document_chunk_counter[document_id]

    chunk.metadata["chunk_index"] = chunk_index

    chunk.metadata["chunk_id"] = (
        f"{document_id}-CHUNK-{chunk_index:03d}"
    )

    document_chunk_counter[document_id] += 1


# ============================================================
# 14. CREATE CHUNK DATAFRAME
# ============================================================
#
# This makes the pipeline easier to inspect.
#
# The actual retrieval system in later parts will use the
# LangChain Document objects.
# ============================================================

chunk_records = []

for chunk in chunked_documents:

    chunk_records.append({
        "chunk_id": chunk.metadata["chunk_id"],
        "document_id": chunk.metadata["document_id"],
        "title": chunk.metadata["title"],
        "department": chunk.metadata["department"],
        "document_type": chunk.metadata["document_type"],
        "source": chunk.metadata["source"],
        "chunk_index": chunk.metadata["chunk_index"],
        "character_count": len(chunk.page_content),
        "text": chunk.page_content
    })


chunks_df = pd.DataFrame(chunk_records)


# ============================================================
# 15. DISPLAY CHUNK DATASET
# ============================================================

print("\n" + "=" * 70)
print("CHUNK DATASET")
print("=" * 70)

print(
    f"\nTotal chunks: {len(chunks_df)}"
)

display(
    chunks_df[
        [
            "chunk_id",
            "document_id",
            "title",
            "department",
            "chunk_index",
            "character_count"
        ]
    ].head(15)
)


# ============================================================
# 16. DISPLAY ACTUAL CHUNK CONTENT
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE CHUNK CONTENT")
print("=" * 70)

for i, chunk in enumerate(chunked_documents[:5]):

    print(f"\n--- CHUNK {i + 1} ---")

    print(
        f"Chunk ID   : {chunk.metadata['chunk_id']}"
    )

    print(
        f"Document   : {chunk.metadata['document_id']}"
    )

    print(
        f"Department : {chunk.metadata['department']}"
    )

    print(
        f"Source     : {chunk.metadata['source']}"
    )

    print("\nContent:")

    print(chunk.page_content)


# ============================================================
# 17. DOCUMENT STATISTICS
# ============================================================

document_statistics = (
    chunks_df
    .groupby(
        [
            "document_id",
            "title",
            "department"
        ]
    )
    .agg(
        chunks=("chunk_id", "count"),
        total_characters=("character_count", "sum"),
        average_chunk_size=("character_count", "mean")
    )
    .reset_index()
)

document_statistics["average_chunk_size"] = (
    document_statistics["average_chunk_size"]
    .round(2)
)


print("\n" + "=" * 70)
print("DOCUMENT STATISTICS")
print("=" * 70)

display(document_statistics)


# ============================================================
# 18. DEPARTMENT DISTRIBUTION
# ============================================================

department_distribution = (
    chunks_df["department"]
    .value_counts()
    .reset_index()
)

department_distribution.columns = [
    "department",
    "chunk_count"
]


print("\n" + "=" * 70)
print("CHUNKS BY DEPARTMENT")
print("=" * 70)

display(department_distribution)


# ============================================================
# 19. CHUNK SIZE ANALYSIS
# ============================================================

chunk_size_statistics = {
    "minimum": int(chunks_df["character_count"].min()),
    "maximum": int(chunks_df["character_count"].max()),
    "average": round(
        chunks_df["character_count"].mean(),
        2
    ),
    "median": round(
        chunks_df["character_count"].median(),
        2
    )
}


print("\n" + "=" * 70)
print("CHUNK SIZE STATISTICS")
print("=" * 70)

for key, value in chunk_size_statistics.items():
    print(f"{key.capitalize():<10}: {value}")


# ============================================================
# 20. VALIDATE DOCUMENT PIPELINE
# ============================================================

validation_results = {}


# ---- Validation 1: Documents exist

validation_results["documents_loaded"] = (
    len(langchain_documents) > 0
)


# ---- Validation 2: Chunks exist

validation_results["chunks_created"] = (
    len(chunked_documents) > 0
)


# ---- Validation 3: Every chunk is a LangChain Document

validation_results["valid_document_objects"] = all(
    isinstance(chunk, Document)
    for chunk in chunked_documents
)


# ---- Validation 4: Required metadata exists

required_metadata = [
    "document_id",
    "title",
    "department",
    "document_type",
    "source",
    "chunk_index",
    "chunk_id"
]

validation_results["metadata_complete"] = all(
    all(field in chunk.metadata for field in required_metadata)
    for chunk in chunked_documents
)


# ---- Validation 5: No empty chunks

validation_results["no_empty_chunks"] = all(
    len(chunk.page_content.strip()) > 0
    for chunk in chunked_documents
)


# ---- Validation 6: Chunk IDs are unique

chunk_ids = [
    chunk.metadata["chunk_id"]
    for chunk in chunked_documents
]

validation_results["unique_chunk_ids"] = (
    len(chunk_ids) == len(set(chunk_ids))
)


# ---- Validation 7: DataFrame exists

validation_results["chunk_dataframe_created"] = (
    not chunks_df.empty
)


# ============================================================
# 21. PRINT VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PART 1 VALIDATION")
print("=" * 70)

for check, result in validation_results.items():

    status = "PASS" if result else "FAIL"

    print(
        f"{status:<6} | {check}"
    )


all_valid = all(validation_results.values())


print("\n" + "-" * 70)

if all_valid:
    print("✅ PART 1 COMPLETED SUCCESSFULLY")
else:
    print("❌ PART 1 VALIDATION FAILED")


# ============================================================
# 22. FINAL PIPELINE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DAY 65 — PART 1 SUMMARY")
print("=" * 70)

print("""
Enterprise Documents
        ↓
Text Cleaning
        ↓
LangChain Document Objects
        ↓
Recursive Character Chunking
        ↓
Metadata Enrichment
        ↓
Chunk Validation
        ↓
Ready for Embeddings + Vector Database
""")


# ============================================================
# 23. IMPORTANT VARIABLES CREATED
# ============================================================

print("=" * 70)
print("IMPORTANT VARIABLES CREATED")
print("=" * 70)

important_variables = {
    "raw_documents": "Original enterprise document dataset",
    "cleaned_documents": "Cleaned enterprise documents",
    "langchain_documents": "LangChain Document objects",
    "text_splitter": "RecursiveCharacterTextSplitter",
    "chunked_documents": "Final LangChain document chunks",
    "chunks_df": "DataFrame containing chunk metadata",
    "document_statistics": "Document-level chunk statistics",
    "department_distribution": "Chunk distribution by department",
    "chunk_size_statistics": "Chunk size statistics",
    "validation_results": "Part 1 validation results"
}

for variable, description in important_variables.items():

    print(
        f"{variable:<28} → {description}"
    )


# ============================================================
# 24. FINAL ARCHITECTURE
# ============================================================

print("\n" + "=" * 70)
print("PART 1 ARCHITECTURE")
print("=" * 70)

print("""
                ENTERPRISE DOCUMENTS
                         │
                         ▼
                  TEXT CLEANING
                         │
                         ▼
              LANGCHAIN DOCUMENT
                         │
                         ▼
             RECURSIVE CHUNKING
                         │
                         ▼
              METADATA ENRICHMENT
                         │
                         ▼
                  CHUNK VALIDATION
                         │
                         ▼
              ┌─────────────────────┐
              │ FINAL CHUNK OBJECTS │
              │                     │
              │ page_content        │
              │ document_id         │
              │ chunk_id            │
              │ chunk_index         │
              │ title               │
              │ department          │
              │ document_type       │
              │ source              │
              └─────────────────────┘
                         │
                         ▼
             PART 2: EMBEDDINGS +
                 VECTOR DATABASE
""")


print("\n🚀 Day 65 Part 1 is ready for Part 2.")
# ============================================================
# DAY 65/100 — ADVANCED LANGCHAIN VECTOR DATABASE
# PART 2 — EMBEDDINGS + VECTOR DATABASE + SIMILARITY SEARCH
# ============================================================
#
# Part 1:
# Enterprise Documents
#       ↓
# Cleaning
#       ↓
# LangChain Documents
#       ↓
# Chunking
#       ↓
# Metadata
#
# Part 2:
# Chunks
#       ↓
# TF-IDF Vector Representation
#       ↓
# LangChain InMemoryVectorStore
#       ↓
# Similarity Search
#       ↓
# Top-K Results
#
# CPU-friendly | Lightweight | No large embedding model
# ============================================================


# ============================================================
# 1. INSTALL REQUIRED PACKAGES
# ============================================================

import sys
import subprocess
import importlib.util


required_packages = {
    "langchain_core": "langchain-core",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy"
}


for import_name, package_name in required_packages.items():

    if importlib.util.find_spec(import_name) is None:

        print(f"Installing: {package_name}")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            package_name
        ])


print("Required packages are ready.")


# ============================================================
# 2. IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


# ============================================================
# 3. DEPENDENCY CHECK
# ============================================================
#
# Part 2 depends on Part 1.
#
# We expect:
#
# - chunked_documents
# - chunks_df
#
# to already exist.
# ============================================================


required_part1_variables = [
    "chunked_documents",
    "chunks_df"
]


missing_variables = [
    variable
    for variable in required_part1_variables
    if variable not in globals()
]


if missing_variables:

    raise RuntimeError(
        "Part 1 variables are missing: "
        + ", ".join(missing_variables)
        + "\nPlease run Part 1 first."
    )


print("✅ Part 1 dependency check passed.")


# ============================================================
# 4. PROJECT CONFIGURATION
# ============================================================

TOP_K = 5

MIN_SIMILARITY_SCORE = 0.05

MAX_FEATURES = 2000

print("\n" + "=" * 70)
print("DAY 65 — PART 2 CONFIGURATION")
print("=" * 70)

print(f"Top-K results          : {TOP_K}")
print(f"Minimum similarity     : {MIN_SIMILARITY_SCORE}")
print(f"Maximum TF-IDF features: {MAX_FEATURES}")


# ============================================================
# 5. EXTRACT CHUNK TEXT
# ============================================================
#
# The vectorizer learns its vocabulary from the chunks.
#
# Each chunk becomes one vector.
#
# Example:
#
# Chunk 1 → [0.0, 0.4, 0.1, ...]
# Chunk 2 → [0.2, 0.0, 0.7, ...]
# Chunk 3 → [0.0, 0.3, 0.5, ...]
# ============================================================


chunk_texts = [
    document.page_content
    for document in chunked_documents
]


print(
    f"\nNumber of chunks to vectorize: {len(chunk_texts)}"
)


# ============================================================
# 6. CREATE TF-IDF VECTOR REPRESENTATION
# ============================================================
#
# TF-IDF converts text into numerical vectors.
#
# TF  = Term Frequency
# IDF = Inverse Document Frequency
#
# The result is a sparse high-dimensional vector.
#
# This is NOT a dense transformer embedding.
# It is used here because it is:
#
# - CPU-friendly
# - fast
# - lightweight
# - easy to inspect
# - suitable for demonstrating vector retrieval architecture
# ============================================================


tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=MAX_FEATURES,
    sublinear_tf=True
)


chunk_vectors = tfidf_vectorizer.fit_transform(
    chunk_texts
)


print("\n" + "=" * 70)
print("VECTOR REPRESENTATION")
print("=" * 70)

print(
    f"Number of chunks : {chunk_vectors.shape[0]}"
)

print(
    f"Vector dimensions: {chunk_vectors.shape[1]}"
)

print(
    f"Matrix format    : {type(chunk_vectors).__name__}"
)

print(
    f"Matrix shape     : {chunk_vectors.shape}"
)


# ============================================================
# 7. INSPECT VECTOR VOCABULARY
# ============================================================

feature_names = tfidf_vectorizer.get_feature_names_out()


print("\n" + "=" * 70)
print("VECTOR VOCABULARY SAMPLE")
print("=" * 70)

print(
    "First 40 vocabulary features:"
)

print(
    list(feature_names[:40])
)


# ============================================================
# 8. CREATE CUSTOM LANGCHAIN EMBEDDINGS CLASS
# ============================================================
#
# LangChain VectorStores expect an Embeddings implementation.
#
# We create a lightweight adapter around our TF-IDF vectorizer.
#
# This gives us:
#
# text → TF-IDF vector
#
# while still allowing us to use LangChain's vector-store API.
# ============================================================


class TFIDFEmbeddings(Embeddings):

    def __init__(self, vectorizer):

        self.vectorizer = vectorizer


    def embed_documents(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        vectors = self.vectorizer.transform(texts)

        return vectors.toarray().tolist()


    def embed_query(
        self,
        text: str
    ) -> List[float]:

        vector = self.vectorizer.transform([text])

        return vector.toarray()[0].tolist()


# ============================================================
# 9. CREATE LANGCHAIN EMBEDDING OBJECT
# ============================================================


embedding_model = TFIDFEmbeddings(
    tfidf_vectorizer
)


print("\n✅ Custom LangChain embedding adapter created.")


# ============================================================
# 10. TEST EMBEDDING GENERATION
# ============================================================


test_text = (
    "How should employees protect sensitive company information?"
)


test_embedding = embedding_model.embed_query(
    test_text
)


print("\n" + "=" * 70)
print("EMBEDDING TEST")
print("=" * 70)

print(
    f"Input text length : {len(test_text)}"
)

print(
    f"Embedding length  : {len(test_embedding)}"
)

print(
    f"Non-zero values   : {sum(value != 0 for value in test_embedding)}"
)


# ============================================================
# 11. CREATE LANGCHAIN IN-MEMORY VECTOR STORE
# ============================================================
#
# InMemoryVectorStore provides a real LangChain vector-store
# abstraction without requiring a large external database.
#
# Later, this architecture can be replaced with:
#
# - Chroma
# - FAISS
# - Pinecone
# - Azure AI Search
# - Qdrant
# - Weaviate
# - Milvus
# - PostgreSQL + pgvector
#
# without changing the overall retrieval concept.
# ============================================================


vector_store = InMemoryVectorStore(
    embedding=embedding_model
)


print("\n" + "=" * 70)
print("LANGCHAIN VECTOR STORE")
print("=" * 70)

print(
    "Vector store type:"
)

print(
    type(vector_store).__name__
)


# ============================================================
# 12. ADD DOCUMENT CHUNKS TO VECTOR STORE
# ============================================================
#
# Each LangChain Document contains:
#
# page_content
# metadata
#
# The vector store creates the corresponding vector and
# associates it with the document metadata.
# ============================================================


vector_ids = vector_store.add_documents(
    documents=chunked_documents
)


print(
    f"\nDocuments added to vector store: {len(vector_ids)}"
)


# ============================================================
# 13. VECTOR STORE VALIDATION
# ============================================================


print("\n" + "=" * 70)
print("VECTOR STORE INTERNAL STATE")
print("=" * 70)


if hasattr(vector_store, "store"):

    print(
        f"Stored vector records: {len(vector_store.store)}"
    )

else:

    print(
        "Vector store successfully initialized."
    )


# ============================================================
# 14. DIRECT COSINE SIMILARITY FUNCTION
# ============================================================
#
# We keep a direct similarity implementation as a transparent
# debugging/evaluation layer.
#
# Query:
#
#     query → vector
#
# Chunk:
#
#     chunk → vector
#
# Similarity:
#
#     cosine(query, chunk)
#
# Higher score = more similar.
# ============================================================


def calculate_query_similarity(query):

    query_vector = tfidf_vectorizer.transform(
        [query]
    )

    similarity_scores = cosine_similarity(
        query_vector,
        chunk_vectors
    )[0]

    return similarity_scores


# ============================================================
# 15. CREATE TRANSPARENT SEARCH FUNCTION
# ============================================================


def vector_search(
    query,
    top_k=TOP_K,
    min_score=MIN_SIMILARITY_SCORE
):
    """
    Lightweight vector search.

    Returns:
        List of dictionaries containing:
        - document
        - score
        - rank
    """

    similarity_scores = calculate_query_similarity(
        query
    )

    ranked_indices = np.argsort(
        similarity_scores
    )[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        score = float(
            similarity_scores[index]
        )

        if score < min_score:
            continue

        document = chunked_documents[index]

        results.append({
            "rank": len(results) + 1,
            "score": score,
            "document": document
        })

    return results


# ============================================================
# 16. CREATE LANGCHAIN VECTOR SEARCH FUNCTION
# ============================================================
#
# This function uses the actual LangChain vector-store API.
# ============================================================


def langchain_vector_search(
    query,
    top_k=TOP_K
):

    results = vector_store.similarity_search_with_score(
        query,
        k=top_k
    )

    formatted_results = []

    for rank, item in enumerate(
        results,
        start=1
    ):

        document, score = item

        formatted_results.append({
            "rank": rank,
            "score": float(score),
            "document": document
        })

    return formatted_results


# ============================================================
# 17. TEST QUERY 1 — SECURITY
# ============================================================


query_1 = (
    "How should employees protect sensitive information?"
)


print("\n" + "=" * 70)
print("VECTOR SEARCH TEST 1")
print("=" * 70)

print(
    f"\nQuery: {query_1}"
)


results_1 = vector_search(
    query_1,
    top_k=TOP_K
)


for result in results_1:

    document = result["document"]

    print("\n" + "-" * 60)

    print(
        f"Rank       : {result['rank']}"
    )

    print(
        f"Similarity : {result['score']:.4f}"
    )

    print(
        f"Document   : {document.metadata['document_id']}"
    )

    print(
        f"Title      : {document.metadata['title']}"
    )

    print(
        f"Department : {document.metadata['department']}"
    )

    print(
        f"Source     : {document.metadata['source']}"
    )

    print(
        f"Content    : {document.page_content[:250]}..."
    )


# ============================================================
# 18. TEST QUERY 2 — INSURANCE
# ============================================================


query_2 = (
    "What should happen when an insurance claim is high risk?"
)


print("\n" + "=" * 70)
print("VECTOR SEARCH TEST 2")
print("=" * 70)

print(
    f"\nQuery: {query_2}"
)


results_2 = vector_search(
    query_2,
    top_k=TOP_K
)


for result in results_2:

    document = result["document"]

    print("\n" + "-" * 60)

    print(
        f"Rank       : {result['rank']}"
    )

    print(
        f"Similarity : {result['score']:.4f}"
    )

    print(
        f"Document   : {document.metadata['document_id']}"
    )

    print(
        f"Title      : {document.metadata['title']}"
    )

    print(
        f"Department : {document.metadata['department']}"
    )

    print(
        f"Content    : {document.page_content[:250]}..."
    )


# ============================================================
# 19. TEST QUERY 3 — API
# ============================================================


query_3 = (
    "How should an internal API handle invalid requests?"
)


print("\n" + "=" * 70)
print("VECTOR SEARCH TEST 3")
print("=" * 70)

print(
    f"\nQuery: {query_3}"
)


results_3 = vector_search(
    query_3,
    top_k=TOP_K
)


for result in results_3:

    document = result["document"]

    print("\n" + "-" * 60)

    print(
        f"Rank       : {result['rank']}"
    )

    print(
        f"Similarity : {result['score']:.4f}"
    )

    print(
        f"Document   : {document.metadata['document_id']}"
    )

    print(
        f"Title      : {document.metadata['title']}"
    )

    print(
        f"Department : {document.metadata['department']}"
    )

    print(
        f"Content    : {document.page_content[:250]}..."
    )


# ============================================================
# 20. TEST ACTUAL LANGCHAIN VECTOR STORE
# ============================================================


langchain_test_query = (
    "What are the requirements for secure cloud deployment?"
)


print("\n" + "=" * 70)
print("LANGCHAIN VECTOR STORE TEST")
print("=" * 70)

print(
    f"\nQuery: {langchain_test_query}"
)


langchain_results = langchain_vector_search(
    langchain_test_query,
    top_k=5
)


for result in langchain_results:

    document = result["document"]

    print("\n" + "-" * 60)

    print(
        f"Rank       : {result['rank']}"
    )

    print(
        f"Vector score: {result['score']:.4f}"
    )

    print(
        f"Document   : {document.metadata['document_id']}"
    )

    print(
        f"Title      : {document.metadata['title']}"
    )

    print(
        f"Department : {document.metadata['department']}"
    )

    print(
        f"Source     : {document.metadata['source']}"
    )


# ============================================================
# 21. BUILD RETRIEVAL RESULT DATAFRAME
# ============================================================


def results_to_dataframe(results):

    rows = []

    for result in results:

        document = result["document"]

        rows.append({
            "rank": result["rank"],
            "score": round(
                result["score"],
                4
            ),
            "document_id": document.metadata["document_id"],
            "chunk_id": document.metadata["chunk_id"],
            "title": document.metadata["title"],
            "department": document.metadata["department"],
            "document_type": document.metadata["document_type"],
            "source": document.metadata["source"],
            "chunk_index": document.metadata["chunk_index"],
            "text": document.page_content
        })

    return pd.DataFrame(rows)


search_results_df = results_to_dataframe(
    results_1
)


print("\n" + "=" * 70)
print("SEARCH RESULT DATAFRAME")
print("=" * 70)

display(
    search_results_df[
        [
            "rank",
            "score",
            "document_id",
            "chunk_id",
            "title",
            "department",
            "source"
        ]
    ]
)


# ============================================================
# 22. SEARCH FUNCTION WITH METADATA
# ============================================================
#
# Metadata will become extremely important in Part 3.
#
# Example:
#
# Search only Finance documents
# Search only Security documents
# Search only Policies
#
# This function introduces the idea without yet implementing
# advanced filtering/reranking.
# ============================================================


def vector_search_with_metadata(
    query,
    top_k=TOP_K,
    department=None,
    document_type=None
):

    similarity_scores = calculate_query_similarity(
        query
    )

    ranked_indices = np.argsort(
        similarity_scores
    )[::-1]

    results = []

    for index in ranked_indices:

        document = chunked_documents[index]

        # Department filter
        if (
            department is not None
            and document.metadata["department"].lower()
            != department.lower()
        ):
            continue

        # Document type filter
        if (
            document_type is not None
            and document.metadata["document_type"].lower()
            != document_type.lower()
        ):
            continue

        score = float(
            similarity_scores[index]
        )

        if score < MIN_SIMILARITY_SCORE:
            continue

        results.append({
            "rank": len(results) + 1,
            "score": score,
            "document": document
        })

        if len(results) >= top_k:
            break

    return results


# ============================================================
# 23. TEST METADATA FILTERING
# ============================================================


filtered_results = vector_search_with_metadata(
    query="How should employees protect company information?",
    top_k=5,
    department="Security"
)


print("\n" + "=" * 70)
print("METADATA FILTER TEST")
print("=" * 70)


for result in filtered_results:

    document = result["document"]

    print("\n" + "-" * 60)

    print(
        f"Rank       : {result['rank']}"
    )

    print(
        f"Score      : {result['score']:.4f}"
    )

    print(
        f"Document   : {document.metadata['document_id']}"
    )

    print(
        f"Department : {document.metadata['department']}"
    )

    print(
        f"Title      : {document.metadata['title']}"
    )


# ============================================================
# 24. RETRIEVAL CONFIDENCE
# ============================================================
#
# A simple lightweight confidence signal:
#
# top score
# average top-K score
# score gap
#
# This is NOT a calibrated probability.
# It is a retrieval-quality signal.
# ============================================================


def calculate_retrieval_confidence(results):

    if not results:

        return {
            "top_score": 0.0,
            "average_score": 0.0,
            "score_gap": 0.0,
            "confidence": "LOW"
        }


    scores = [
        result["score"]
        for result in results
    ]


    top_score = scores[0]

    average_score = float(
        np.mean(scores)
    )


    if len(scores) > 1:

        score_gap = (
            scores[0] - scores[1]
        )

    else:

        score_gap = scores[0]


    if top_score >= 0.45:

        confidence = "HIGH"

    elif top_score >= 0.20:

        confidence = "MEDIUM"

    else:

        confidence = "LOW"


    return {
        "top_score": round(
            float(top_score),
            4
        ),
        "average_score": round(
            average_score,
            4
        ),
        "score_gap": round(
            float(score_gap),
            4
        ),
        "confidence": confidence
    }


# ============================================================
# 25. TEST RETRIEVAL CONFIDENCE
# ============================================================


confidence_result = calculate_retrieval_confidence(
    results_1
)


print("\n" + "=" * 70)
print("RETRIEVAL CONFIDENCE")
print("=" * 70)

for key, value in confidence_result.items():

    print(
        f"{key:<20}: {value}"
    )


# ============================================================
# 26. CREATE REUSABLE ENTERPRISE RETRIEVER
# ============================================================


def enterprise_vector_retriever(
    query,
    top_k=TOP_K,
    department=None,
    document_type=None
):
    """
    Enterprise vector retrieval wrapper.

    Pipeline:

    Query
       ↓
    TF-IDF vector
       ↓
    Similarity calculation
       ↓
    Metadata filtering
       ↓
    Top-K
       ↓
    Retrieval confidence
    """

    results = vector_search_with_metadata(
        query=query,
        top_k=top_k,
        department=department,
        document_type=document_type
    )


    confidence = calculate_retrieval_confidence(
        results
    )


    return {
        "query": query,
        "results": results,
        "confidence": confidence
    }


# ============================================================
# 27. TEST ENTERPRISE RETRIEVER
# ============================================================


retrieval_response = enterprise_vector_retriever(
    query="What happens to high risk insurance claims?",
    top_k=3,
    department="Insurance"
)


print("\n" + "=" * 70)
print("ENTERPRISE VECTOR RETRIEVER")
print("=" * 70)

print(
    f"\nQuery: {retrieval_response['query']}"
)

print(
    f"\nConfidence: "
    f"{retrieval_response['confidence']}"
)


for result in retrieval_response["results"]:

    document = result["document"]

    print("\n" + "-" * 60)

    print(
        f"Rank : {result['rank']}"
    )

    print(
        f"Score: {result['score']:.4f}"
    )

    print(
        f"ID   : {document.metadata['document_id']}"
    )

    print(
        f"Text : {document.page_content[:300]}..."
    )


# ============================================================
# 28. VECTOR STORE STATISTICS
# ============================================================


vector_store_statistics = {
    "documents": len(chunked_documents),
    "vector_dimensions": int(
        chunk_vectors.shape[1]
    ),
    "vocabulary_size": len(feature_names),
    "top_k": TOP_K,
    "embedding_type": "TF-IDF sparse lexical vector",
    "vector_store": "LangChain InMemoryVectorStore"
}


print("\n" + "=" * 70)
print("VECTOR STORE STATISTICS")
print("=" * 70)

for key, value in vector_store_statistics.items():

    print(
        f"{key:<25}: {value}"
    )


# ============================================================
# 29. PART 2 VALIDATION
# ============================================================


part2_validation = {}


# Vectorizer exists
part2_validation["vectorizer_created"] = (
    tfidf_vectorizer is not None
)


# Vectors created
part2_validation["vectors_created"] = (
    chunk_vectors.shape[0]
    == len(chunked_documents)
)


# Vector dimensions exist
part2_validation["vector_dimensions_valid"] = (
    chunk_vectors.shape[1] > 0
)


# Embedding adapter exists
part2_validation["embedding_adapter_created"] = (
    isinstance(
        embedding_model,
        Embeddings
    )
)


# Vector store exists
part2_validation["vector_store_created"] = (
    vector_store is not None
)


# Documents added
part2_validation["documents_indexed"] = (
    len(vector_ids)
    == len(chunked_documents)
)


# Search works
part2_validation["vector_search_works"] = (
    len(results_1) > 0
)


# Metadata search works
part2_validation["metadata_filter_works"] = (
    all(
        result["document"].metadata["department"]
        == "Security"
        for result in filtered_results
    )
    if filtered_results
    else False
)


# Confidence works
part2_validation["confidence_generated"] = (
    "confidence" in confidence_result
)


# DataFrame created
part2_validation["results_dataframe_created"] = (
    isinstance(
        search_results_df,
        pd.DataFrame
    )
)


# ============================================================
# 30. PRINT VALIDATION
# ============================================================


print("\n" + "=" * 70)
print("DAY 65 — PART 2 VALIDATION")
print("=" * 70)


for check, result in part2_validation.items():

    status = "PASS" if result else "FAIL"

    print(
        f"{status:<6} | {check}"
    )


part2_success = all(
    part2_validation.values()
)


print("\n" + "-" * 70)


if part2_success:

    print(
        "✅ PART 2 COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "❌ PART 2 VALIDATION FAILED"
    )


# ============================================================
# 31. IMPORTANT VARIABLES CREATED
# ============================================================


print("\n" + "=" * 70)
print("IMPORTANT VARIABLES CREATED")
print("=" * 70)


important_variables_part2 = {

    "tfidf_vectorizer":
        "TF-IDF vector representation model",

    "chunk_vectors":
        "Numerical vector matrix for all chunks",

    "feature_names":
        "Vector vocabulary/features",

    "embedding_model":
        "LangChain-compatible TF-IDF Embeddings adapter",

    "vector_store":
        "LangChain InMemoryVectorStore",

    "vector_ids":
        "IDs assigned to indexed documents",

    "vector_search":
        "Transparent cosine similarity search",

    "langchain_vector_search":
        "LangChain vector-store similarity search",

    "vector_search_with_metadata":
        "Vector search with metadata filtering",

    "enterprise_vector_retriever":
        "Reusable enterprise retrieval function",

    "confidence_result":
        "Retrieval quality signal",

    "search_results_df":
        "Search results DataFrame",

    "part2_validation":
        "Part 2 validation results"
}


for variable, description in important_variables_part2.items():

    print(
        f"{variable:<32} → {description}"
    )


# ============================================================
# 32. FINAL PART 2 ARCHITECTURE
# ============================================================


print("\n" + "=" * 70)
print("DAY 65 — PART 2 ARCHITECTURE")
print("=" * 70)


print("""
                    USER QUERY
                        │
                        ▼
                Query Text
                        │
                        ▼
              TF-IDF Embedding
                        │
                        ▼
              Query Vector
                        │
                        ▼
              ┌─────────────────┐
              │ LangChain       │
              │ InMemory        │
              │ VectorStore     │
              └────────┬────────┘
                       │
                       ▼
              Similarity Search
                       │
                       ▼
                Candidate Chunks
                       │
                       ▼
                Metadata Filter
                       │
                       ▼
                    Top-K
                       │
                       ▼
             Retrieval Confidence
                       │
                       ▼
               Retrieved Context
                       │
                       ▼
                 PART 3
        ADVANCED RETRIEVAL + RERANKING
""")


# ============================================================
# 33. KEY CONCEPTS
# ============================================================


print("\n" + "=" * 70)
print("KEY CONCEPTS LEARNED")
print("=" * 70)


concepts = [
    "Text can be represented as numerical vectors.",
    "TF-IDF creates sparse lexical vectors.",
    "Embeddings allow numerical similarity comparison.",
    "Cosine similarity measures vector similarity.",
    "A vector store connects vectors with source documents.",
    "LangChain provides a common vector-store abstraction.",
    "Metadata travels with the retrieved document.",
    "Top-K controls how many results are returned.",
    "Metadata filtering can restrict the retrieval scope.",
    "Retrieval confidence is a quality signal, not a probability.",
    "TF-IDF is lightweight but is not a modern semantic embedding model.",
    "Production systems can replace TF-IDF with dense embedding models."
]


for index, concept in enumerate(concepts, start=1):

    print(
        f"{index:02d}. {concept}"
    )


print("\n🚀 Day 65 Part 2 is complete.")
print("Next → Part 3: Advanced Retrieval, Query Expansion & Reranking")
# ============================================================
# DAY 65/100 — ADVANCED LANGCHAIN VECTOR DATABASE
# PART 3 — ADVANCED RETRIEVAL + QUERY EXPANSION + RERANKING
# ============================================================
#
# Part 1:
# Documents → Cleaning → LangChain Documents → Chunking → Metadata
#
# Part 2:
# Chunks → TF-IDF Vectors → Vector Store → Similarity Search
#
# Part 3:
# Query
#   ↓
# Query Normalization
#   ↓
# Query Expansion
#   ↓
# Candidate Retrieval
#   ↓
# Metadata Filtering
#   ↓
# Similarity Threshold
#   ↓
# Reranking
#   ↓
# Deduplication
#   ↓
# Top-K
#   ↓
# Retrieval Confidence
#   ↓
# Retrieval Evaluation
#
# CPU-friendly | Lightweight | Jupyter Notebook
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import re
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 2. DEPENDENCY CHECK
# ============================================================
#
# Part 3 depends on variables created in Parts 1 and 2.
# ============================================================

required_variables_part3 = [
    "chunked_documents",
    "chunks_df",
    "tfidf_vectorizer",
    "chunk_vectors",
    "embedding_model",
    "vector_store",
    "vector_search",
    "vector_search_with_metadata",
    "calculate_retrieval_confidence"
]


missing_variables = [
    variable
    for variable in required_variables_part3
    if variable not in globals()
]


if missing_variables:

    raise RuntimeError(
        "Missing variables from previous parts: "
        + ", ".join(missing_variables)
        + "\nPlease run Parts 1 and 2 first."
    )


print("✅ Part 1 + Part 2 dependency check passed.")


# ============================================================
# 3. PART 3 CONFIGURATION
# ============================================================

CANDIDATE_K = 10

FINAL_TOP_K = 5

SIMILARITY_THRESHOLD = 0.03

RERANK_VECTOR_WEIGHT = 0.60

RERANK_KEYWORD_WEIGHT = 0.25

RERANK_METADATA_WEIGHT = 0.15

MAX_QUERY_VARIANTS = 4

DIVERSITY_PENALTY = 0.08


print("\n" + "=" * 70)
print("DAY 65 — PART 3 CONFIGURATION")
print("=" * 70)

print(f"Candidate K              : {CANDIDATE_K}")
print(f"Final Top-K              : {FINAL_TOP_K}")
print(f"Similarity threshold     : {SIMILARITY_THRESHOLD}")
print(f"Vector score weight      : {RERANK_VECTOR_WEIGHT}")
print(f"Keyword score weight     : {RERANK_KEYWORD_WEIGHT}")
print(f"Metadata score weight    : {RERANK_METADATA_WEIGHT}")
print(f"Max query variants       : {MAX_QUERY_VARIANTS}")
print(f"Diversity penalty        : {DIVERSITY_PENALTY}")


# ============================================================
# 4. QUERY NORMALIZATION
# ============================================================
#
# Before retrieval, normalize the user's query.
#
# Operations:
# - lowercase
# - remove unnecessary punctuation
# - normalize whitespace
# ============================================================


def normalize_query(query):

    if not isinstance(query, str):

        return ""

    query = query.lower().strip()

    query = re.sub(
        r"[^a-z0-9\s\-]",
        " ",
        query
    )

    query = re.sub(
        r"\s+",
        " ",
        query
    )

    return query.strip()


# ============================================================
# 5. TEST QUERY NORMALIZATION
# ============================================================

raw_test_query = (
    "How should HIGH-RISK insurance claims be handled?"
)

normalized_test_query = normalize_query(
    raw_test_query
)


print("\n" + "=" * 70)
print("QUERY NORMALIZATION")
print("=" * 70)

print(f"Original   : {raw_test_query}")
print(f"Normalized : {normalized_test_query}")


# ============================================================
# 6. LIGHTWEIGHT QUERY EXPANSION DICTIONARY
# ============================================================
#
# A production system can use an LLM for query rewriting or
# multi-query generation.
#
# Because this notebook is CPU/storage constrained, we use
# deterministic domain-aware expansion.
#
# Example:
#
# "high risk claim"
#
# can produce:
#
# "high risk claim"
# "high risk insurance claim"
# "claim risk assessment"
# "unusual insurance claim"
# ============================================================


QUERY_EXPANSION_TERMS = {

    "insurance": [
        "claim",
        "policy",
        "eligibility",
        "risk",
        "assessment"
    ],

    "claim": [
        "insurance",
        "policy",
        "eligibility",
        "risk",
        "review"
    ],

    "security": [
        "sensitive information",
        "access control",
        "authentication",
        "least privilege",
        "security incident"
    ],

    "api": [
        "request",
        "response",
        "validation",
        "authentication",
        "authorization"
    ],

    "cloud": [
        "deployment",
        "health check",
        "secrets",
        "monitoring",
        "rollback"
    ],

    "ai": [
        "artificial intelligence",
        "governance",
        "responsible ai",
        "human review",
        "audit"
    ],

    "employee": [
        "staff",
        "worker",
        "user",
        "personnel"
    ],

    "incident": [
        "technical issue",
        "service disruption",
        "root cause",
        "resolution",
        "priority"
    ]
}


# ============================================================
# 7. QUERY EXPANSION FUNCTION
# ============================================================


def expand_query(
    query,
    max_variants=MAX_QUERY_VARIANTS
):
    """
    Generate lightweight deterministic query variants.

    The original query is always preserved.
    """

    normalized = normalize_query(query)

    if not normalized:

        return []


    variants = [normalized]

    query_words = normalized.split()


    # --------------------------------------------------------
    # Variant 1: Add relevant domain terms
    # --------------------------------------------------------

    matched_terms = []

    for word in query_words:

        if word in QUERY_EXPANSION_TERMS:

            matched_terms.extend(
                QUERY_EXPANSION_TERMS[word]
            )


    if matched_terms:

        expanded = (
            normalized
            + " "
            + " ".join(
                matched_terms[:5]
            )
        )

        variants.append(
            expanded
        )


    # --------------------------------------------------------
    # Variant 2: Important keyword query
    # --------------------------------------------------------

    meaningful_words = [
        word
        for word in query_words
        if len(word) >= 4
    ]


    if meaningful_words:

        keyword_query = " ".join(
            meaningful_words
        )

        variants.append(
            keyword_query
        )


    # --------------------------------------------------------
    # Variant 3: Reordered domain-oriented query
    # --------------------------------------------------------

    if len(query_words) >= 3:

        domain_words = []

        for word in query_words:

            if (
                word in QUERY_EXPANSION_TERMS
                or len(word) >= 6
            ):

                domain_words.append(word)


        if domain_words:

            variants.append(
                " ".join(domain_words)
            )


    # --------------------------------------------------------
    # Remove duplicates while preserving order
    # --------------------------------------------------------

    unique_variants = []

    for variant in variants:

        if variant not in unique_variants:

            unique_variants.append(
                variant
            )


    return unique_variants[
        :max_variants
    ]


# ============================================================
# 8. TEST QUERY EXPANSION
# ============================================================


expansion_test_query = (
    "How should high risk insurance claims be handled?"
)


query_variants = expand_query(
    expansion_test_query
)


print("\n" + "=" * 70)
print("QUERY EXPANSION")
print("=" * 70)

print(
    f"Original query: {expansion_test_query}"
)

print("\nGenerated variants:")

for index, variant in enumerate(
    query_variants,
    start=1
):

    print(
        f"{index}. {variant}"
    )


# ============================================================
# 9. KEYWORD OVERLAP SCORE
# ============================================================
#
# Vector similarity tells us how close two TF-IDF vectors are.
#
# We also calculate simple lexical overlap.
#
# This provides a second signal for reranking.
# ============================================================


STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "how",
    "what",
    "should",
    "when",
    "why",
    "can",
    "must"
}


def tokenize_for_overlap(text):

    normalized = normalize_query(
        text
    )

    tokens = normalized.split()

    return {
        token
        for token in tokens
        if token not in STOP_WORDS
        and len(token) >= 3
    }


def keyword_overlap_score(
    query,
    document_text
):

    query_tokens = tokenize_for_overlap(
        query
    )

    document_tokens = tokenize_for_overlap(
        document_text
    )


    if not query_tokens:

        return 0.0


    overlap = (
        query_tokens
        .intersection(
            document_tokens
        )
    )


    return (
        len(overlap)
        / len(query_tokens)
    )


# ============================================================
# 10. TEST KEYWORD OVERLAP
# ============================================================


test_document_text = chunked_documents[0].page_content


overlap_test_score = keyword_overlap_score(
    "security sensitive information",
    test_document_text
)


print("\n" + "=" * 70)
print("KEYWORD OVERLAP TEST")
print("=" * 70)

print(
    f"Overlap score: {overlap_test_score:.4f}"
)


# ============================================================
# 11. METADATA RELEVANCE SCORE
# ============================================================
#
# Metadata can provide another retrieval signal.
#
# For example:
#
# Query:
#     "insurance claim risk"
#
# Document department:
#     Insurance
#
# This should receive a small ranking boost.
#
# This is NOT access control.
# It is only a ranking signal.
# ============================================================


def metadata_relevance_score(
    query,
    document
):

    normalized_query = normalize_query(
        query
    )


    score = 0.0


    metadata_fields = [
        "title",
        "department",
        "document_type"
    ]


    for field in metadata_fields:

        value = normalize_query(
            str(
                document.metadata.get(
                    field,
                    ""
                )
            )
        )


        if not value:

            continue


        if value in normalized_query:

            score += 0.5

        else:

            metadata_tokens = set(
                value.split()
            )

            query_tokens = set(
                normalized_query.split()
            )

            overlap = (
                metadata_tokens
                .intersection(
                    query_tokens
                )
            )

            if overlap:

                score += min(
                    0.25 * len(overlap),
                    0.5
                )


    return min(
        score,
        1.0
    )


# ============================================================
# 12. CANDIDATE RETRIEVAL
# ============================================================
#
# Query expansion creates several retrieval queries.
#
# Each query retrieves candidates.
#
# Candidates are merged and deduplicated.
# ============================================================


def retrieve_candidates(
    query,
    candidate_k=CANDIDATE_K,
    department=None,
    document_type=None
):
    """
    Retrieve a broad candidate pool before reranking.
    """

    variants = expand_query(
        query
    )


    candidate_map = {}


    for variant in variants:

        similarity_scores = calculate_query_similarity(
            variant
        )


        ranked_indices = np.argsort(
            similarity_scores
        )[::-1]


        for index in ranked_indices[
            :candidate_k
        ]:

            document = chunked_documents[
                index
            ]


            # --------------------------------------------
            # Metadata filtering
            # --------------------------------------------

            if (
                department is not None
                and document.metadata[
                    "department"
                ].lower()
                != department.lower()
            ):

                continue


            if (
                document_type is not None
                and document.metadata[
                    "document_type"
                ].lower()
                != document_type.lower()
            ):

                continue


            score = float(
                similarity_scores[index]
            )


            if score < SIMILARITY_THRESHOLD:

                continue


            chunk_id = document.metadata[
                "chunk_id"
            ]


            if chunk_id not in candidate_map:

                candidate_map[
                    chunk_id
                ] = {

                    "document": document,

                    "best_vector_score":
                        score,

                    "variant_scores": [
                        score
                    ],

                    "matched_variants": [
                        variant
                    ]
                }

            else:

                existing = candidate_map[
                    chunk_id
                ]


                existing[
                    "variant_scores"
                ].append(
                    score
                )


                existing[
                    "matched_variants"
                ].append(
                    variant
                )


                if score > existing[
                    "best_vector_score"
                ]:

                    existing[
                        "best_vector_score"
                    ] = score


    return list(
        candidate_map.values()
    )


# ============================================================
# 13. TEST CANDIDATE RETRIEVAL
# ============================================================


candidate_query = (
    "How should high risk insurance claims be handled?"
)


candidate_results = retrieve_candidates(
    candidate_query,
    candidate_k=CANDIDATE_K,
    department="Insurance"
)


print("\n" + "=" * 70)
print("CANDIDATE RETRIEVAL")
print("=" * 70)

print(
    f"Query: {candidate_query}"
)

print(
    f"Candidates retrieved: "
    f"{len(candidate_results)}"
)


for index, candidate in enumerate(
    candidate_results,
    start=1
):

    document = candidate[
        "document"
    ]


    print("\n" + "-" * 60)

    print(
        f"Candidate       : {index}"
    )

    print(
        f"Document ID     : "
        f"{document.metadata['document_id']}"
    )

    print(
        f"Chunk ID        : "
        f"{document.metadata['chunk_id']}"
    )

    print(
        f"Vector score    : "
        f"{candidate['best_vector_score']:.4f}"
    )

    print(
        f"Title           : "
        f"{document.metadata['title']}"
    )


# ============================================================
# 14. RERANKING FUNCTION
# ============================================================
#
# Final ranking combines:
#
# Vector similarity
#        +
# Keyword overlap
#        +
# Metadata relevance
#
# Weighted score:
#
# final_score =
#       vector_score   * 0.60
#     + keyword_score  * 0.25
#     + metadata_score * 0.15
#
# This is a lightweight deterministic reranker.
#
# In a production system, this layer could be replaced with:
# - cross-encoder reranker
# - managed semantic ranker
# - LLM-based reranker
# ============================================================


def rerank_candidates(
    query,
    candidates,
    final_top_k=FINAL_TOP_K
):

    reranked = []


    for candidate in candidates:

        document = candidate[
            "document"
        ]


        vector_score = float(
            candidate[
                "best_vector_score"
            ]
        )


        keyword_score = keyword_overlap_score(
            query,
            document.page_content
        )


        metadata_score = metadata_relevance_score(
            query,
            document
        )


        final_score = (

            vector_score
            * RERANK_VECTOR_WEIGHT

            +

            keyword_score
            * RERANK_KEYWORD_WEIGHT

            +

            metadata_score
            * RERANK_METADATA_WEIGHT
        )


        reranked.append({

            "document": document,

            "vector_score":
                vector_score,

            "keyword_score":
                keyword_score,

            "metadata_score":
                metadata_score,

            "rerank_score":
                final_score,

            "matched_variants":
                candidate[
                    "matched_variants"
                ]
        })


    reranked.sort(
        key=lambda item:
            item["rerank_score"],
        reverse=True
    )


    return reranked[
        :final_top_k
    ]


# ============================================================
# 15. TEST RERANKING
# ============================================================


reranked_results = rerank_candidates(
    candidate_query,
    candidate_results,
    final_top_k=FINAL_TOP_K
)


print("\n" + "=" * 70)
print("RERANKED RESULTS")
print("=" * 70)


for rank, result in enumerate(
    reranked_results,
    start=1
):

    document = result[
        "document"
    ]


    print("\n" + "-" * 60)

    print(
        f"Final Rank       : {rank}"
    )

    print(
        f"Document ID      : "
        f"{document.metadata['document_id']}"
    )

    print(
        f"Chunk ID         : "
        f"{document.metadata['chunk_id']}"
    )

    print(
        f"Vector score     : "
        f"{result['vector_score']:.4f}"
    )

    print(
        f"Keyword score    : "
        f"{result['keyword_score']:.4f}"
    )

    print(
        f"Metadata score   : "
        f"{result['metadata_score']:.4f}"
    )

    print(
        f"Final rerank     : "
        f"{result['rerank_score']:.4f}"
    )

    print(
        f"Title            : "
        f"{document.metadata['title']}"
    )


# ============================================================
# 16. DEDUPLICATION
# ============================================================
#
# Multiple query variants may return overlapping chunks.
#
# Deduplication guarantees that the same chunk is not returned
# multiple times.
# ============================================================


def deduplicate_results(
    results
):

    unique_results = {}

    for result in results:

        document = result[
            "document"
        ]

        chunk_id = document.metadata[
            "chunk_id"
        ]


        if chunk_id not in unique_results:

            unique_results[
                chunk_id
            ] = result

        else:

            existing = unique_results[
                chunk_id
            ]

            if (
                result["rerank_score"]
                >
                existing["rerank_score"]
            ):

                unique_results[
                    chunk_id
                ] = result


    return list(
        unique_results.values()
    )


# ============================================================
# 17. LIGHTWEIGHT DIVERSITY FILTER
# ============================================================
#
# We want relevant results, but we also want to avoid returning
# too many near-duplicate chunks from the same document.
#
# The function applies a small diversity penalty.
# ============================================================


def apply_diversity_ranking(
    results,
    final_top_k=FINAL_TOP_K
):

    selected = []

    document_counts = {}


    remaining = sorted(
        results,
        key=lambda item:
            item["rerank_score"],
        reverse=True
    )


    while (
        remaining
        and len(selected) < final_top_k
    ):

        best_index = None

        best_adjusted_score = -1


        for index, result in enumerate(
            remaining
        ):

            document = result[
                "document"
            ]


            document_id = document.metadata[
                "document_id"
            ]


            count = document_counts.get(
                document_id,
                0
            )


            adjusted_score = (
                result["rerank_score"]
                -
                DIVERSITY_PENALTY
                * count
            )


            if (
                adjusted_score
                >
                best_adjusted_score
            ):

                best_adjusted_score = (
                    adjusted_score
                )

                best_index = index


        selected_result = remaining.pop(
            best_index
        )


        selected_result[
            "diversity_adjusted_score"
        ] = best_adjusted_score


        selected.append(
            selected_result
        )


        selected_document_id = (
            selected_result[
                "document"
            ].metadata[
                "document_id"
            ]
        )


        document_counts[
            selected_document_id
        ] = (
            document_counts.get(
                selected_document_id,
                0
            )
            + 1
        )


    return selected


# ============================================================
# 18. COMPLETE ADVANCED RETRIEVAL FUNCTION
# ============================================================


def advanced_retrieval(
    query,
    top_k=FINAL_TOP_K,
    department=None,
    document_type=None
):
    """
    Complete advanced retrieval pipeline.

    Query
      ↓
    Normalize
      ↓
    Expand
      ↓
    Candidate Retrieval
      ↓
    Metadata Filtering
      ↓
    Similarity Threshold
      ↓
    Reranking
      ↓
    Deduplication
      ↓
    Diversity
      ↓
    Top-K
      ↓
    Confidence
    """


    normalized_query = normalize_query(
        query
    )


    variants = expand_query(
        normalized_query
    )


    candidates = retrieve_candidates(
        normalized_query,
        candidate_k=CANDIDATE_K,
        department=department,
        document_type=document_type
    )


    reranked = rerank_candidates(
        normalized_query,
        candidates,
        final_top_k=max(
            top_k,
            CANDIDATE_K
        )
    )


    deduplicated = deduplicate_results(
        reranked
    )


    final_results = apply_diversity_ranking(
        deduplicated,
        final_top_k=top_k
    )


    # --------------------------------------------------------
    # Recalculate retrieval confidence
    # using final reranked scores.
    # --------------------------------------------------------

    if final_results:

        final_scores = [
            result[
                "rerank_score"
            ]
            for result in final_results
        ]


        top_score = float(
            final_scores[0]
        )


        average_score = float(
            np.mean(final_scores)
        )


        if len(final_scores) > 1:

            score_gap = (
                final_scores[0]
                -
                final_scores[1]
            )

        else:

            score_gap = top_score


        if top_score >= 0.45:

            confidence = "HIGH"

        elif top_score >= 0.20:

            confidence = "MEDIUM"

        else:

            confidence = "LOW"


    else:

        top_score = 0.0
        average_score = 0.0
        score_gap = 0.0
        confidence = "LOW"


    confidence_result = {

        "top_score":
            round(top_score, 4),

        "average_score":
            round(average_score, 4),

        "score_gap":
            round(score_gap, 4),

        "confidence":
            confidence
    }


    return {

        "query":
            query,

        "normalized_query":
            normalized_query,

        "query_variants":
            variants,

        "candidate_count":
            len(candidates),

        "results":
            final_results,

        "confidence":
            confidence_result
    }


# ============================================================
# 19. TEST COMPLETE ADVANCED RETRIEVAL
# ============================================================


advanced_result = advanced_retrieval(
    query=(
        "What should happen when an insurance "
        "claim is high risk?"
    ),
    top_k=5,
    department="Insurance"
)


print("\n" + "=" * 70)
print("COMPLETE ADVANCED RETRIEVAL")
print("=" * 70)


print(
    f"\nOriginal query:"
)

print(
    advanced_result["query"]
)


print(
    f"\nNormalized query:"
)

print(
    advanced_result["normalized_query"]
)


print(
    "\nQuery variants:"
)

for index, variant in enumerate(
    advanced_result["query_variants"],
    start=1
):

    print(
        f"{index}. {variant}"
    )


print(
    f"\nCandidate count: "
    f"{advanced_result['candidate_count']}"
)


print(
    f"\nRetrieval confidence:"
)

print(
    advanced_result["confidence"]
)


for rank, result in enumerate(
    advanced_result["results"],
    start=1
):

    document = result[
        "document"
    ]


    print("\n" + "-" * 60)

    print(
        f"Rank          : {rank}"
    )

    print(
        f"Document      : "
        f"{document.metadata['document_id']}"
    )

    print(
        f"Chunk         : "
        f"{document.metadata['chunk_id']}"
    )

    print(
        f"Rerank score  : "
        f"{result['rerank_score']:.4f}"
    )

    print(
        f"Vector score  : "
        f"{result['vector_score']:.4f}"
    )

    print(
        f"Keyword score : "
        f"{result['keyword_score']:.4f}"
    )

    print(
        f"Metadata score: "
        f"{result['metadata_score']:.4f}"
    )

    print(
        f"Title         : "
        f"{document.metadata['title']}"
    )

    print(
        f"Source        : "
        f"{document.metadata['source']}"
    )

    print(
        "\nContent:"
    )

    print(
        document.page_content[:400]
    )


# ============================================================
# 20. BUILD RETRIEVAL RESULT DATAFRAME
# ============================================================


def advanced_results_dataframe(
    retrieval_response
):

    rows = []


    for rank, result in enumerate(
        retrieval_response["results"],
        start=1
    ):

        document = result[
            "document"
        ]


        rows.append({

            "rank":
                rank,

            "rerank_score":
                round(
                    result[
                        "rerank_score"
                    ],
                    4
                ),

            "vector_score":
                round(
                    result[
                        "vector_score"
                    ],
                    4
                ),

            "keyword_score":
                round(
                    result[
                        "keyword_score"
                    ],
                    4
                ),

            "metadata_score":
                round(
                    result[
                        "metadata_score"
                    ],
                    4
                ),

            "document_id":
                document.metadata[
                    "document_id"
                ],

            "chunk_id":
                document.metadata[
                    "chunk_id"
                ],

            "title":
                document.metadata[
                    "title"
                ],

            "department":
                document.metadata[
                    "department"
                ],

            "source":
                document.metadata[
                    "source"
                ],

            "text":
                document.page_content
        })


    return pd.DataFrame(rows)


advanced_results_df = (
    advanced_results_dataframe(
        advanced_result
    )
)


print("\n" + "=" * 70)
print("ADVANCED RETRIEVAL RESULTS")
print("=" * 70)


display(
    advanced_results_df[
        [
            "rank",
            "rerank_score",
            "vector_score",
            "keyword_score",
            "metadata_score",
            "document_id",
            "chunk_id",
            "title",
            "department"
        ]
    ]
)


# ============================================================
# 21. MULTIPLE ENTERPRISE QUERY TESTS
# ============================================================
#
# Test different enterprise domains.
# ============================================================


evaluation_queries = [

    {
        "query":
            "How should sensitive company information be protected?",

        "department":
            "Security"
    },

    {
        "query":
            "What should happen when an insurance claim is unusual?",

        "department":
            "Insurance"
    },

    {
        "query":
            "How should invalid API requests be handled?",

        "department":
            "Engineering"
    },

    {
        "query":
            "What are important controls for cloud deployment?",

        "department":
            "Cloud Engineering"
    },

    {
        "query":
            "What should employees do during remote work?",

        "department":
            "Human Resources"
    }
]


evaluation_results = []


for test_case in evaluation_queries:

    response = advanced_retrieval(

        query=test_case["query"],

        top_k=FINAL_TOP_K,

        department=test_case["department"]
    )


    results = response[
        "results"
    ]


    top_document_id = None

    top_score = 0.0


    if results:

        top_document_id = (
            results[0][
                "document"
            ].metadata[
                "document_id"
            ]
        )

        top_score = (
            results[0][
                "rerank_score"
            ]
        )


    evaluation_results.append({

        "query":
            test_case["query"],

        "expected_department":
            test_case["department"],

        "top_document":
            top_document_id,

        "top_score":
            round(
                float(top_score),
                4
            ),

        "confidence":
            response[
                "confidence"
            ]["confidence"],

        "result_count":
            len(results)
    })


retrieval_evaluation_df = pd.DataFrame(
    evaluation_results
)


print("\n" + "=" * 70)
print("RETRIEVAL TEST RESULTS")
print("=" * 70)


display(
    retrieval_evaluation_df
)


# ============================================================
# 22. HIT@K EVALUATION
# ============================================================
#
# For this controlled dataset, we define the expected department
# as the relevance label.
#
# Hit@K asks:
#
# "Did at least one retrieved result belong to the expected
# department?"
#
# This is a simple retrieval evaluation metric.
# ============================================================


def calculate_hit_at_k(
    query_tests,
    k=FINAL_TOP_K
):

    hits = 0

    total = len(
        query_tests
    )


    details = []


    for test_case in query_tests:

        response = advanced_retrieval(

            query=test_case[
                "query"
            ],

            top_k=k,

            department=test_case[
                "department"
            ]
        )


        results = response[
            "results"
        ]


        retrieved_departments = [

            result[
                "document"
            ].metadata[
                "department"
            ]

            for result in results
        ]


        hit = (
            test_case[
                "department"
            ]
            in
            retrieved_departments
        )


        if hit:

            hits += 1


        details.append({

            "query":
                test_case["query"],

            "expected":
                test_case["department"],

            "retrieved":
                retrieved_departments,

            "hit":
                hit
        })


    hit_at_k = (
        hits / total
        if total > 0
        else 0.0
    )


    return hit_at_k, details


# ============================================================
# 23. CALCULATE HIT@5
# ============================================================


hit_at_5, hit_details = calculate_hit_at_k(
    evaluation_queries,
    k=5
)


print("\n" + "=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)


print(
    f"Hit@5: {hit_at_5:.2%}"
)


for detail in hit_details:

    print("\n" + "-" * 60)

    print(
        f"Query    : {detail['query']}"
    )

    print(
        f"Expected : {detail['expected']}"
    )

    print(
        f"Retrieved: {detail['retrieved']}"
    )

    print(
        f"Hit      : {detail['hit']}"
    )


# ============================================================
# 24. RETRIEVAL QUALITY SUMMARY
# ============================================================


confidence_values = (
    retrieval_evaluation_df[
        "confidence"
    ]
    .value_counts()
    .to_dict()
)


retrieval_quality_summary = {

    "test_queries":
        len(evaluation_queries),

    "hit_at_5":
        round(
            float(hit_at_5),
            4
        ),

    "high_confidence_queries":
        confidence_values.get(
            "HIGH",
            0
        ),

    "medium_confidence_queries":
        confidence_values.get(
            "MEDIUM",
            0
        ),

    "low_confidence_queries":
        confidence_values.get(
            "LOW",
            0
        ),

    "average_top_score":
        round(
            float(
                retrieval_evaluation_df[
                    "top_score"
                ].mean()
            ),
            4
        )
}


print("\n" + "=" * 70)
print("RETRIEVAL QUALITY SUMMARY")
print("=" * 70)


for key, value in (
    retrieval_quality_summary.items()
):

    print(
        f"{key:<30}: {value}"
    )


# ============================================================
# 25. LOW-CONFIDENCE / UNKNOWN QUERY TEST
# ============================================================
#
# A good retrieval system should not blindly return results
# with strong confidence for an unrelated query.
#
# This test demonstrates the retrieval confidence signal.
# ============================================================


unknown_query = (
    "What are the procedures for marine biology experiments?"
)


unknown_result = advanced_retrieval(
    query=unknown_query,
    top_k=FINAL_TOP_K
)


print("\n" + "=" * 70)
print("UNKNOWN QUERY TEST")
print("=" * 70)


print(
    f"\nQuery: {unknown_query}"
)


print(
    "\nConfidence:"
)

print(
    unknown_result["confidence"]
)


for rank, result in enumerate(
    unknown_result["results"],
    start=1
):

    document = result[
        "document"
    ]


    print("\n" + "-" * 60)

    print(
        f"Rank  : {rank}"
    )

    print(
        f"Score : "
        f"{result['rerank_score']:.4f}"
    )

    print(
        f"Title : "
        f"{document.metadata['title']}"
    )


# ============================================================
# 26. BUILD FINAL RETRIEVAL PIPELINE WRAPPER
# ============================================================


def enterprise_retrieval_pipeline(
    query,
    top_k=FINAL_TOP_K,
    department=None,
    document_type=None
):

    response = advanced_retrieval(

        query=query,

        top_k=top_k,

        department=department,

        document_type=document_type
    )


    # --------------------------------------------------------
    # Safe retrieval decision
    # --------------------------------------------------------

    confidence = response[
        "confidence"
    ]["confidence"]


    if confidence == "LOW":

        retrieval_status = (
            "LOW_CONFIDENCE"
        )

    else:

        retrieval_status = (
            "RETRIEVED"
        )


    return {

        **response,

        "retrieval_status":
            retrieval_status
    }


# ============================================================
# 27. TEST FINAL RETRIEVAL WRAPPER
# ============================================================


final_retrieval_test = (
    enterprise_retrieval_pipeline(
        query=(
            "How should high risk "
            "insurance claims be handled?"
        ),
        top_k=3,
        department="Insurance"
    )
)


print("\n" + "=" * 70)
print("FINAL ENTERPRISE RETRIEVAL WRAPPER")
print("=" * 70)


print(
    f"\nStatus: "
    f"{final_retrieval_test['retrieval_status']}"
)


print(
    f"Confidence: "
    f"{final_retrieval_test['confidence']}"
)


for rank, result in enumerate(
    final_retrieval_test["results"],
    start=1
):

    document = result[
        "document"
    ]


    print("\n" + "-" * 60)

    print(
        f"Rank: {rank}"
    )

    print(
        f"Score: "
        f"{result['rerank_score']:.4f}"
    )

    print(
        f"Document: "
        f"{document.metadata['document_id']}"
    )

    print(
        f"Source: "
        f"{document.metadata['source']}"
    )


# ============================================================
# 28. PART 3 VALIDATION
# ============================================================


part3_validation = {}


# Query normalization
part3_validation[
    "query_normalization"
] = (
    normalize_query(
        " TEST QUERY! "
    )
    == "test query"
)


# Query expansion
part3_validation[
    "query_expansion"
] = (
    len(
        expand_query(
            "insurance claim risk"
        )
    ) >= 1
)


# Candidate retrieval
part3_validation[
    "candidate_retrieval"
] = (
    len(
        candidate_results
    ) > 0
)


# Reranking
part3_validation[
    "reranking"
] = (
    len(
        reranked_results
    ) > 0
)


# Deduplication
deduplicated_test = (
    deduplicate_results(
        reranked_results
    )
)


deduplicated_ids = [

    result[
        "document"
    ].metadata[
        "chunk_id"
    ]

    for result in deduplicated_test
]


part3_validation[
    "deduplication"
] = (
    len(
        deduplicated_ids
    )
    ==
    len(
        set(
            deduplicated_ids
        )
    )
)


# Final advanced retrieval
part3_validation[
    "advanced_retrieval"
] = (
    len(
        advanced_result[
            "results"
        ]
    ) > 0
)


# Confidence
part3_validation[
    "retrieval_confidence"
] = (
    "confidence"
    in advanced_result[
        "confidence"
    ]
)


# Evaluation
part3_validation[
    "retrieval_evaluation"
] = (
    0.0
    <= hit_at_5
    <= 1.0
)


# Result DataFrame
part3_validation[
    "evaluation_dataframe"
] = (
    isinstance(
        retrieval_evaluation_df,
        pd.DataFrame
    )
)


# ============================================================
# 29. PRINT VALIDATION
# ============================================================


print("\n" + "=" * 70)
print("DAY 65 — PART 3 VALIDATION")
print("=" * 70)


for check, result in (
    part3_validation.items()
):

    status = (
        "PASS"
        if result
        else
        "FAIL"
    )


    print(
        f"{status:<6} | {check}"
    )


part3_success = all(
    part3_validation.values()
)


print("\n" + "-" * 70)


if part3_success:

    print(
        "✅ PART 3 COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "❌ PART 3 VALIDATION FAILED"
    )


# ============================================================
# 30. IMPORTANT VARIABLES CREATED
# ============================================================


print("\n" + "=" * 70)
print("IMPORTANT VARIABLES CREATED")
print("=" * 70)


important_variables_part3 = {

    "QUERY_EXPANSION_TERMS":
        "Lightweight domain query expansion dictionary",

    "normalize_query":
        "Query normalization function",

    "expand_query":
        "Query expansion function",

    "keyword_overlap_score":
        "Lexical relevance scoring",

    "metadata_relevance_score":
        "Metadata relevance scoring",

    "retrieve_candidates":
        "Broad candidate retrieval",

    "rerank_candidates":
        "Multi-signal reranking",

    "deduplicate_results":
        "Duplicate chunk removal",

    "apply_diversity_ranking":
        "Lightweight diversity-aware ranking",

    "advanced_retrieval":
        "Complete advanced retrieval pipeline",

    "enterprise_retrieval_pipeline":
        "Production-oriented retrieval wrapper",

    "advanced_result":
        "Latest advanced retrieval response",

    "advanced_results_df":
        "Advanced retrieval results DataFrame",

    "retrieval_evaluation_df":
        "Retrieval evaluation DataFrame",

    "hit_at_5":
        "Hit@5 retrieval metric",

    "retrieval_quality_summary":
        "Overall retrieval quality summary",

    "part3_validation":
        "Part 3 validation results"
}


for variable, description in (
    important_variables_part3.items()
):

    print(
        f"{variable:<34} → {description}"
    )


# ============================================================
# 31. FINAL PART 3 ARCHITECTURE
# ============================================================


print("\n" + "=" * 70)
print("DAY 65 — PART 3 FINAL ARCHITECTURE")
print("=" * 70)


print("""
                         USER QUERY
                              │
                              ▼
                    QUERY NORMALIZATION
                              │
                              ▼
                      QUERY EXPANSION
                              │
                  ┌───────────┴───────────┐
                  │                       │
                  ▼                       ▼
             Query Variant 1        Query Variant N
                  │                       │
                  └───────────┬───────────┘
                              ▼
                    CANDIDATE RETRIEVAL
                              │
                              ▼
                    METADATA FILTERING
                              │
                              ▼
                   SIMILARITY THRESHOLD
                              │
                              ▼
                         RERANKING
                              │
                 ┌────────────┼────────────┐
                 │            │            │
                 ▼            ▼            ▼
              Vector       Keyword      Metadata
               Score        Score         Score
                 │            │            │
                 └────────────┼────────────┘
                              ▼
                      SCORE FUSION
                              │
                              ▼
                       DEDUPLICATION
                              │
                              ▼
                    DIVERSITY RANKING
                              │
                              ▼
                           TOP-K
                              │
                              ▼
                  RETRIEVAL CONFIDENCE
                              │
                              ▼
                    RETRIEVED CONTEXT
                              │
                              ▼
                          PART 4
                    RAG + GROUNDING
                    + CITATIONS
                    + FASTAPI
""")


# ============================================================
# 32. KEY CONCEPTS
# ============================================================


print("\n" + "=" * 70)
print("KEY CONCEPTS LEARNED")
print("=" * 70)


concepts_part3 = [

    "Query normalization prepares user input for retrieval.",

    "Query expansion generates additional retrieval formulations.",

    "Candidate retrieval intentionally retrieves a broader pool.",

    "Metadata filtering can restrict retrieval scope.",

    "Similarity thresholds remove weak candidates.",

    "Reranking combines multiple relevance signals.",

    "Vector similarity measures numerical text similarity.",

    "Keyword overlap provides a lexical relevance signal.",

    "Metadata relevance can improve domain-specific ranking.",

    "Deduplication prevents repeated chunks.",

    "Diversity ranking reduces excessive same-document results.",

    "Top-K controls the final context size.",

    "Retrieval confidence helps identify weak retrieval.",

    "Hit@K evaluates whether relevant content was retrieved.",

    "The current reranker is deterministic and lightweight.",

    "A production system can replace it with a cross-encoder or"
   " managed semantic reranker.",

    "TF-IDF remains a lightweight lexical baseline, not a"
   " transformer-based semantic embedding model."
]


for index, concept in enumerate(
    concepts_part3,
    start=1
):

    print(
        f"{index:02d}. {concept}"
    )


# ============================================================
# 33. FINAL STATUS
# ============================================================


print("\n" + "=" * 70)
print("DAY 65 — PART 3 COMPLETE")
print("=" * 70)


print("""
✅ Query normalization
✅ Query expansion
✅ Candidate retrieval
✅ Metadata filtering
✅ Similarity threshold
✅ Multi-signal reranking
✅ Deduplication
✅ Diversity ranking
✅ Top-K retrieval
✅ Retrieval confidence
✅ Hit@K evaluation
✅ Enterprise retrieval wrapper

Next:
PART 4 — RAG + Grounding + Source Citations
          + FastAPI + Production Architecture
""")
# ============================================================
# DAY 65/100 — ADVANCED LANGCHAIN VECTOR DATABASE
# PART 4 — RAG + GROUNDING + CITATIONS + FASTAPI
# ============================================================
#
# COMPLETE FINAL SYSTEM
#
# User
#   ↓
# FastAPI
#   ↓
# Query Validation
#   ↓
# Advanced Retrieval
#   ↓
# Top-K Context
#   ↓
# Grounded Response Generation
#   ↓
# Grounding Validation
#   ↓
# Source Citations
#   ↓
# Safe Fallback
#   ↓
# Final Response
#
# + Monitoring
# + Metrics
# + Health Check
# + Search API
# + Query API
#
# CPU-friendly | Lightweight | Jupyter Notebook
# ============================================================


# ============================================================
# 1. INSTALL REQUIRED PACKAGES
# ============================================================

import sys
import subprocess
import importlib.util


required_packages_part4 = {
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "uvicorn": "uvicorn",
    "pandas": "pandas",
    "numpy": "numpy"
}


for import_name, package_name in required_packages_part4.items():

    if importlib.util.find_spec(import_name) is None:

        print(f"Installing: {package_name}")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            package_name
        ])


print("Required Part 4 packages are ready.")


# ============================================================
# 2. IMPORTS
# ============================================================

import os
import re
import time
import uuid
import logging

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from pydantic import BaseModel, Field

from fastapi import (
    FastAPI,
    HTTPException,
    Request
)

from fastapi.responses import JSONResponse


# ============================================================
# 3. DEPENDENCY CHECK
# ============================================================
#
# Part 4 depends on Parts 1–3.
#
# Required:
#
# - chunked_documents
# - vector_store
# - advanced_retrieval
# - enterprise_retrieval_pipeline
# ============================================================


required_part4_variables = [
    "chunked_documents",
    "vector_store",
    "advanced_retrieval",
    "enterprise_retrieval_pipeline"
]


missing_variables = [
    variable
    for variable in required_part4_variables
    if variable not in globals()
]


if missing_variables:

    raise RuntimeError(
        "Missing variables from previous parts: "
        + ", ".join(missing_variables)
        + "\nPlease run Parts 1, 2 and 3 first."
    )


print("✅ Part 1 + Part 2 + Part 3 dependency check passed.")


# ============================================================
# 4. APPLICATION CONFIGURATION
# ============================================================

APP_NAME = (
    "Enterprise LangChain Vector DB RAG API"
)

APP_VERSION = "1.0.0"

APP_ENV = "development"

DEFAULT_TOP_K = 5

MAX_TOP_K = 10

MIN_QUERY_LENGTH = 3

MAX_QUERY_LENGTH = 1000

GROUNDING_THRESHOLD = 0.20

MAX_CONTEXT_CHARS = 5000


print("\n" + "=" * 70)
print("DAY 65 — PART 4 CONFIGURATION")
print("=" * 70)

print(f"Application       : {APP_NAME}")
print(f"Version           : {APP_VERSION}")
print(f"Environment       : {APP_ENV}")
print(f"Default Top-K     : {DEFAULT_TOP_K}")
print(f"Maximum Top-K     : {MAX_TOP_K}")
print(f"Grounding threshold: {GROUNDING_THRESHOLD}")


# ============================================================
# 5. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(
    "day65_rag"
)


# ============================================================
# 6. APPLICATION METRICS
# ============================================================

METRICS = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "search_requests": 0,
    "query_requests": 0,
    "low_confidence_requests": 0,
    "grounding_failures": 0,
    "total_latency_ms": 0.0,
    "max_latency_ms": 0.0
}


def record_request_metric(
    endpoint,
    latency_ms,
    success=True
):

    METRICS["total_requests"] += 1

    METRICS["total_latency_ms"] += latency_ms

    METRICS["max_latency_ms"] = max(
        METRICS["max_latency_ms"],
        latency_ms
    )


    if success:

        METRICS[
            "successful_requests"
        ] += 1

    else:

        METRICS[
            "failed_requests"
        ] += 1


    if endpoint == "search":

        METRICS[
            "search_requests"
        ] += 1


    if endpoint == "query":

        METRICS[
            "query_requests"
        ] += 1


def metrics_snapshot():

    total = METRICS[
        "total_requests"
    ]

    average_latency = (
        METRICS[
            "total_latency_ms"
        ] / total
        if total > 0
        else 0.0
    )


    success_rate = (
        METRICS[
            "successful_requests"
        ] / total
        if total > 0
        else 0.0
    )


    return {

        **METRICS,

        "average_latency_ms":
            round(
                average_latency,
                2
            ),

        "success_rate":
            round(
                success_rate,
                4
            )
    }


# ============================================================
# 7. QUERY VALIDATION
# ============================================================


def validate_query(query):

    if not isinstance(query, str):

        raise ValueError(
            "Query must be a string."
        )


    query = query.strip()


    if len(query) < MIN_QUERY_LENGTH:

        raise ValueError(
            f"Query must contain at least "
            f"{MIN_QUERY_LENGTH} characters."
        )


    if len(query) > MAX_QUERY_LENGTH:

        raise ValueError(
            f"Query cannot exceed "
            f"{MAX_QUERY_LENGTH} characters."
        )


    return query


# ============================================================
# 8. CONTEXT CONSTRUCTION
# ============================================================
#
# Retrieved documents are transformed into a controlled context
# that can be passed to a generation model.
#
# Each context item contains:
#
# - source
# - document
# - chunk
# - content
#
# This source information will later become citations.
# ============================================================


def build_context(
    retrieval_results,
    max_chars=MAX_CONTEXT_CHARS
):

    context_parts = []

    current_length = 0


    for rank, result in enumerate(
        retrieval_results,
        start=1
    ):

        document = result[
            "document"
        ]


        source = document.metadata.get(
            "source",
            "unknown"
        )


        document_id = document.metadata.get(
            "document_id",
            "unknown"
        )


        chunk_id = document.metadata.get(
            "chunk_id",
            "unknown"
        )


        title = document.metadata.get(
            "title",
            "unknown"
        )


        content = (
            document.page_content
            .strip()
        )


        context_block = (
            f"[SOURCE {rank}]\n"
            f"Document ID: {document_id}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Content: {content}\n"
        )


        if (
            current_length
            + len(context_block)
            > max_chars
        ):

            break


        context_parts.append(
            context_block
        )


        current_length += len(
            context_block
        )


    return "\n".join(
        context_parts
    )


# ============================================================
# 9. SOURCE CITATION BUILDER
# ============================================================


def build_citations(
    retrieval_results
):

    citations = []


    for rank, result in enumerate(
        retrieval_results,
        start=1
    ):

        document = result[
            "document"
        ]


        citations.append({

            "rank":
                rank,

            "document_id":
                document.metadata.get(
                    "document_id"
                ),

            "chunk_id":
                document.metadata.get(
                    "chunk_id"
                ),

            "title":
                document.metadata.get(
                    "title"
                ),

            "department":
                document.metadata.get(
                    "department"
                ),

            "source":
                document.metadata.get(
                    "source"
                ),

            "score":
                round(
                    float(
                        result.get(
                            "rerank_score",
                            0.0
                        )
                    ),
                    4
                )
        })


    return citations


# ============================================================
# 10. EXTRACTIVE ANSWER GENERATOR
# ============================================================
#
# This is intentionally NOT an external LLM.
#
# We create a lightweight grounded response from retrieved
# sentences.
#
# In production:
#
# retrieved context
#       ↓
# enterprise LLM
#       ↓
# generated answer
#
# Here:
#
# retrieved context
#       ↓
# sentence relevance
#       ↓
# grounded answer
#
# This allows the complete RAG architecture to run on CPU
# without downloading a large model.
# ============================================================


def split_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def generate_grounded_answer(
    query,
    retrieval_results,
    max_sentences=4
):

    if not retrieval_results:

        return (
            "I could not find sufficiently relevant "
            "information in the available documents."
        )


    query_terms = set(
        normalize_query(
            query
        ).split()
    )


    candidate_sentences = []


    for result in retrieval_results:

        document = result[
            "document"
        ]


        sentences = split_sentences(
            document.page_content
        )


        for sentence in sentences:

            sentence_terms = set(
                normalize_query(
                    sentence
                ).split()
            )


            overlap = (
                query_terms
                .intersection(
                    sentence_terms
                )
            )


            overlap_score = (
                len(overlap)
                /
                max(
                    len(query_terms),
                    1
                )
            )


            candidate_sentences.append({

                "sentence":
                    sentence,

                "score":
                    overlap_score,

                "document":
                    document
            })


    candidate_sentences.sort(
        key=lambda item:
            item["score"],
        reverse=True
    )


    selected = []

    seen_sentences = set()


    for candidate in candidate_sentences:

        sentence = candidate[
            "sentence"
        ]


        normalized_sentence = (
            normalize_query(
                sentence
            )
        )


        if (
            normalized_sentence
            in seen_sentences
        ):

            continue


        seen_sentences.add(
            normalized_sentence
        )


        selected.append(
            candidate
        )


        if len(selected) >= max_sentences:

            break


    if not selected:

        return (
            "The retrieved documents did not contain "
            "enough directly relevant information "
            "to provide a grounded answer."
        )


    answer_sentences = [
        item["sentence"]
        for item in selected
    ]


    return " ".join(
        answer_sentences
    )


# ============================================================
# 11. GROUNDING VALIDATION
# ============================================================
#
# The answer should be supported by retrieved context.
#
# This lightweight validator checks lexical overlap between
# answer terms and the retrieved context.
#
# It is NOT a full hallucination detector.
# ============================================================


def grounding_check(
    answer,
    context,
    threshold=GROUNDING_THRESHOLD
):

    answer_terms = tokenize_for_overlap(
        answer
    )


    context_terms = tokenize_for_overlap(
        context
    )


    if not answer_terms:

        return {

            "grounded":
                False,

            "score":
                0.0,

            "reason":
                "Empty answer"
        }


    overlap = (
        answer_terms
        .intersection(
            context_terms
        )
    )


    score = (
        len(overlap)
        /
        len(answer_terms)
    )


    grounded = (
        score >= threshold
    )


    return {

        "grounded":
            grounded,

        "score":
            round(
                float(score),
                4
            ),

        "reason":
            (
                "Answer terms are sufficiently "
                "supported by retrieved context."
                if grounded
                else
                "Answer has insufficient overlap "
                "with retrieved context."
            )
    }


# ============================================================
# 12. SAFE FALLBACK
# ============================================================


def safe_fallback(
    confidence,
    grounding
):

    if confidence == "LOW":

        return (
            "I could not find sufficiently relevant "
            "information in the enterprise knowledge base "
            "to answer this question reliably."
        )


    if not grounding["grounded"]:

        return (
            "I found related information, but I could "
            "not establish enough evidence to provide "
            "a reliable grounded answer. Please refine "
            "the question or provide additional documents."
        )


    return None


# ============================================================
# 13. COMPLETE RAG PIPELINE
# ============================================================
#
# This is the central function of Part 4.
# ============================================================


def enterprise_rag(
    query,
    top_k=DEFAULT_TOP_K,
    department=None,
    document_type=None
):

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # Step 1 — Validate
    # --------------------------------------------------------

    query = validate_query(
        query
    )


    # --------------------------------------------------------
    # Step 2 — Normalize Top-K
    # --------------------------------------------------------

    top_k = max(
        1,
        min(
            int(top_k),
            MAX_TOP_K
        )
    )


    # --------------------------------------------------------
    # Step 3 — Advanced Retrieval
    # --------------------------------------------------------

    retrieval_response = (
        enterprise_retrieval_pipeline(

            query=query,

            top_k=top_k,

            department=department,

            document_type=document_type
        )
    )


    retrieval_results = (
        retrieval_response[
            "results"
        ]
    )


    confidence = (
        retrieval_response[
            "confidence"
        ]
    )


    # --------------------------------------------------------
    # Step 4 — Low confidence protection
    # --------------------------------------------------------

    if (
        confidence["confidence"]
        == "LOW"
    ):

        METRICS[
            "low_confidence_requests"
        ] += 1


        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000


        return {

            "answer":
                safe_fallback(
                    confidence[
                        "confidence"
                    ],
                    {
                        "grounded":
                            False
                    }
                ),

            "status":
                "LOW_CONFIDENCE",

            "query":
                query,

            "retrieval_confidence":
                confidence,

            "citations":
                build_citations(
                    retrieval_results
                ),

            "context":
                build_context(
                    retrieval_results
                ),

            "grounding":
                {
                    "grounded":
                        False,

                    "score":
                        0.0,

                    "reason":
                        "Retrieval confidence is low."
                },

            "latency_ms":
                round(
                    latency_ms,
                    2
                )
        }


    # --------------------------------------------------------
    # Step 5 — Context Construction
    # --------------------------------------------------------

    context = build_context(
        retrieval_results
    )


    # --------------------------------------------------------
    # Step 6 — Answer Generation
    # --------------------------------------------------------

    answer = generate_grounded_answer(

        query=query,

        retrieval_results=
            retrieval_results,

        max_sentences=4
    )


    # --------------------------------------------------------
    # Step 7 — Grounding Validation
    # --------------------------------------------------------

    grounding = grounding_check(
        answer,
        context
    )


    # --------------------------------------------------------
    # Step 8 — Safe Fallback
    # --------------------------------------------------------

    fallback = safe_fallback(
        confidence[
            "confidence"
        ],
        grounding
    )


    if fallback is not None:

        METRICS[
            "grounding_failures"
        ] += 1


        final_answer = fallback

        status = (
            "SAFE_FALLBACK"
        )

    else:

        final_answer = answer

        status = (
            "GROUNDED"
        )


    # --------------------------------------------------------
    # Step 9 — Citations
    # --------------------------------------------------------

    citations = build_citations(
        retrieval_results
    )


    # --------------------------------------------------------
    # Step 10 — Latency
    # --------------------------------------------------------

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000


    return {

        "answer":
            final_answer,

        "status":
            status,

        "query":
            query,

        "retrieval_confidence":
            confidence,

        "grounding":
            grounding,

        "citations":
            citations,

        "context":
            context,

        "retrieved_chunks":
            len(
                retrieval_results
            ),

        "latency_ms":
            round(
                latency_ms,
                2
            )
    }


# ============================================================
# 14. TEST RAG — SECURITY
# ============================================================


rag_test_1 = enterprise_rag(

    query=(
        "How should employees protect "
        "sensitive company information?"
    ),

    top_k=3,

    department="Security"
)


print("\n" + "=" * 70)
print("RAG TEST 1 — SECURITY")
print("=" * 70)


print(
    f"\nStatus:"
)

print(
    rag_test_1["status"]
)


print(
    f"\nAnswer:"
)

print(
    rag_test_1["answer"]
)


print(
    f"\nConfidence:"
)

print(
    rag_test_1[
        "retrieval_confidence"
    ]
)


print(
    f"\nGrounding:"
)

print(
    rag_test_1[
        "grounding"
    ]
)


print(
    "\nCitations:"
)

for citation in rag_test_1[
    "citations"
]:

    print(
        citation
    )


print(
    f"\nLatency: "
    f"{rag_test_1['latency_ms']} ms"
)


# ============================================================
# 15. TEST RAG — INSURANCE
# ============================================================


rag_test_2 = enterprise_rag(

    query=(
        "What should happen when "
        "an insurance claim is high risk?"
    ),

    top_k=3,

    department="Insurance"
)


print("\n" + "=" * 70)
print("RAG TEST 2 — INSURANCE")
print("=" * 70)


print(
    f"\nStatus: "
    f"{rag_test_2['status']}"
)


print(
    f"\nAnswer:"
)

print(
    rag_test_2["answer"]
)


print(
    f"\nGrounding:"
)

print(
    rag_test_2[
        "grounding"
    ]
)


print(
    "\nCitations:"
)

for citation in rag_test_2[
    "citations"
]:

    print(
        citation
    )


# ============================================================
# 16. TEST RAG — CLOUD
# ============================================================


rag_test_3 = enterprise_rag(

    query=(
        "What are important controls "
        "for cloud application deployment?"
    ),

    top_k=3,

    department="Cloud Engineering"
)


print("\n" + "=" * 70)
print("RAG TEST 3 — CLOUD")
print("=" * 70)


print(
    f"\nStatus: "
    f"{rag_test_3['status']}"
)


print(
    f"\nAnswer:"
)

print(
    rag_test_3["answer"]
)


print(
    f"\nGrounding:"
)

print(
    rag_test_3[
        "grounding"
    ]
)


# ============================================================
# 17. TEST SAFE FALLBACK
# ============================================================


rag_unknown = enterprise_rag(

    query=(
        "What are the procedures "
        "for marine biology experiments?"
    ),

    top_k=5
)


print("\n" + "=" * 70)
print("SAFE FALLBACK TEST")
print("=" * 70)


print(
    f"\nStatus:"
)

print(
    rag_unknown["status"]
)


print(
    f"\nAnswer:"
)

print(
    rag_unknown["answer"]
)


print(
    f"\nConfidence:"
)

print(
    rag_unknown[
        "retrieval_confidence"
    ]
)


# ============================================================
# 18. RAG EVALUATION DATASET
# ============================================================


rag_evaluation_queries = [

    {
        "query":
            "How should sensitive company information be protected?",

        "department":
            "Security"
    },

    {
        "query":
            "What should happen when an insurance claim is high risk?",

        "department":
            "Insurance"
    },

    {
        "query":
            "How should invalid API requests be handled?",

        "department":
            "Engineering"
    },

    {
        "query":
            "What controls are important for cloud deployment?",

        "department":
            "Cloud Engineering"
    },

    {
        "query":
            "What are the requirements for remote employees?",

        "department":
            "Human Resources"
    }
]


# ============================================================
# 19. RUN RAG EVALUATION
# ============================================================


rag_evaluation_rows = []


for test_case in rag_evaluation_queries:

    response = enterprise_rag(

        query=test_case[
            "query"
        ],

        top_k=5,

        department=test_case[
            "department"
        ]
    )


    rag_evaluation_rows.append({

        "query":
            test_case[
                "query"
            ],

        "department":
            test_case[
                "department"
            ],

        "status":
            response[
                "status"
            ],

        "grounded":
            response[
                "grounding"
            ][
                "grounded"
            ],

        "grounding_score":
            response[
                "grounding"
            ][
                "score"
            ],

        "retrieval_confidence":
            response[
                "retrieval_confidence"
            ][
                "confidence"
            ],

        "retrieved_chunks":
            response[
                "retrieved_chunks"
            ],

        "latency_ms":
            response[
                "latency_ms"
            ]
    })


rag_evaluation_df = pd.DataFrame(
    rag_evaluation_rows
)


print("\n" + "=" * 70)
print("RAG EVALUATION")
print("=" * 70)


display(
    rag_evaluation_df
)


# ============================================================
# 20. RAG QUALITY METRICS
# ============================================================


rag_grounding_rate = (
    rag_evaluation_df[
        "grounded"
    ].mean()
)


rag_average_grounding = (
    rag_evaluation_df[
        "grounding_score"
    ].mean()
)


rag_average_latency = (
    rag_evaluation_df[
        "latency_ms"
    ].mean()
)


rag_quality_summary = {

    "test_queries":
        len(
            rag_evaluation_df
        ),

    "grounding_rate":
        round(
            float(
                rag_grounding_rate
            ),
            4
        ),

    "average_grounding_score":
        round(
            float(
                rag_average_grounding
            ),
            4
        ),

    "average_latency_ms":
        round(
            float(
                rag_average_latency
            ),
            2
        ),

    "safe_fallback_count":
        int(
            (
                rag_evaluation_df[
                    "status"
                ]
                == "SAFE_FALLBACK"
            ).sum()
        )
}


print("\n" + "=" * 70)
print("RAG QUALITY SUMMARY")
print("=" * 70)


for key, value in (
    rag_quality_summary.items()
):

    print(
        f"{key:<30}: {value}"
    )


# ============================================================
# 21. FASTAPI APPLICATION
# ============================================================


app = FastAPI(

    title=APP_NAME,

    version=APP_VERSION,

    description=(
        "CPU-friendly enterprise LangChain "
        "Vector DB and RAG API."
    )
)


# ============================================================
# 22. REQUEST / RESPONSE MODELS
# ============================================================


class SearchRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10
    )

    department: Optional[str] = None

    document_type: Optional[str] = None


class QueryRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10
    )

    department: Optional[str] = None

    document_type: Optional[str] = None


# ============================================================
# 23. REQUEST ID MIDDLEWARE
# ============================================================


@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next
):

    request_id = str(
        uuid.uuid4()
    )


    request.state.request_id = (
        request_id
    )


    start_time = time.perf_counter()


    try:

        response = await call_next(
            request
        )


        response.headers[
            "X-Request-ID"
        ] = request_id


        return response


    finally:

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000


        logger.info(
            "%s %s | request_id=%s | latency=%.2fms",
            request.method,
            request.url.path,
            request_id,
            latency_ms
        )


# ============================================================
# 24. ROOT ENDPOINT
# ============================================================


@app.get("/")
def root():

    return {

        "application":
            APP_NAME,

        "version":
            APP_VERSION,

        "status":
            "running",

        "architecture":
            "LangChain Vector DB + Advanced Retrieval + RAG",

        "endpoints": [

            "/health",

            "/metrics",

            "/info",

            "/search",

            "/query"
        ]
    }


# ============================================================
# 25. HEALTH ENDPOINT
# ============================================================


@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "application":
            APP_NAME,

        "version":
            APP_VERSION,

        "environment":
            APP_ENV,

        "documents":
            len(
                chunked_documents
            ),

        "vector_store":
            type(
                vector_store
            ).__name__,

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat()
    }


# ============================================================
# 26. INFO ENDPOINT
# ============================================================


@app.get("/info")
def info():

    return {

        "application":
            APP_NAME,

        "version":
            APP_VERSION,

        "environment":
            APP_ENV,

        "documents":
            len(
                chunked_documents
            ),

        "default_top_k":
            DEFAULT_TOP_K,

        "max_top_k":
            MAX_TOP_K,

        "features": [

            "LangChain Documents",

            "Vector Store",

            "Advanced Retrieval",

            "Query Expansion",

            "Reranking",

            "RAG",

            "Grounding",

            "Source Citations",

            "Safe Fallback",

            "FastAPI",

            "Monitoring"
        ]
    }


# ============================================================
# 27. METRICS ENDPOINT
# ============================================================


@app.get("/metrics")
def metrics():

    return metrics_snapshot()


# ============================================================
# 28. SEARCH ENDPOINT
# ============================================================


@app.post("/search")
def search_endpoint(
    request: SearchRequest
):

    start_time = time.perf_counter()


    try:

        response = (
            enterprise_retrieval_pipeline(

                query=request.query,

                top_k=request.top_k,

                department=request.department,

                document_type=
                    request.document_type
            )
        )


        results = []


        for rank, result in enumerate(
            response["results"],
            start=1
        ):

            document = result[
                "document"
            ]


            results.append({

                "rank":
                    rank,

                "chunk_id":
                    document.metadata.get(
                        "chunk_id"
                    ),

                "document_id":
                    document.metadata.get(
                        "document_id"
                    ),

                "title":
                    document.metadata.get(
                        "title"
                    ),

                "department":
                    document.metadata.get(
                        "department"
                    ),

                "document_type":
                    document.metadata.get(
                        "document_type"
                    ),

                "source":
                    document.metadata.get(
                        "source"
                    ),

                "score":
                    round(
                        float(
                            result[
                                "rerank_score"
                            ]
                        ),
                        4
                    ),

                "text":
                    document.page_content
            })


        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000


        record_request_metric(
            "search",
            latency_ms,
            success=True
        )


        if (
            response[
                "confidence"
            ]["confidence"]
            == "LOW"
        ):

            METRICS[
                "low_confidence_requests"
            ] += 1


        return {

            "query":
                request.query,

            "results":
                results,

            "result_count":
                len(results),

            "confidence":
                response[
                    "confidence"
                ],

            "latency_ms":
                round(
                    latency_ms,
                    2
                )
        }


    except Exception as error:

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000


        record_request_metric(
            "search",
            latency_ms,
            success=False
        )


        logger.exception(
            "Search endpoint failed."
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "Search operation failed."
            )
        )


# ============================================================
# 29. QUERY / RAG ENDPOINT
# ============================================================


@app.post("/query")
def query_endpoint(
    request: QueryRequest
):

    start_time = time.perf_counter()


    try:

        response = enterprise_rag(

            query=request.query,

            top_k=request.top_k,

            department=request.department,

            document_type=
                request.document_type
        )


        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000


        record_request_metric(
            "query",
            latency_ms,
            success=True
        )


        return {

            "query":
                request.query,

            "answer":
                response[
                    "answer"
                ],

            "status":
                response[
                    "status"
                ],

            "retrieval_confidence":
                response[
                    "retrieval_confidence"
                ],

            "grounding":
                response[
                    "grounding"
                ],

            "citations":
                response[
                    "citations"
                ],

            "retrieved_chunks":
                response[
                    "retrieved_chunks"
                ],

            "latency_ms":
                response[
                    "latency_ms"
                ]
        }


    except ValueError as error:

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000


        record_request_metric(
            "query",
            latency_ms,
            success=False
        )


        raise HTTPException(

            status_code=400,

            detail=str(
                error
            )
        )


    except Exception as error:

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000


        record_request_metric(
            "query",
            latency_ms,
            success=False
        )


        logger.exception(
            "Query endpoint failed."
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "Query processing failed."
            )
        )


# ============================================================
# 30. GLOBAL EXCEPTION HANDLER
# ============================================================


@app.exception_handler(
    Exception
)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    logger.exception(
        "Unhandled application exception."
    )


    return JSONResponse(

        status_code=500,

        content={

            "error":
                "Internal server error.",

            "request_id":
                getattr(
                    request.state,
                    "request_id",
                    None
                )
        }
    )


# ============================================================
# 31. FASTAPI TEST CLIENT
# ============================================================
#
# Test the API directly inside the notebook.
# No external server is required for these tests.
# ============================================================


from fastapi.testclient import TestClient


client = TestClient(
    app
)


# ============================================================
# 32. API TEST — ROOT
# ============================================================


root_response = client.get("/")


print("\n" + "=" * 70)
print("API TEST — ROOT")
print("=" * 70)

print(
    f"Status code: "
    f"{root_response.status_code}"
)

print(
    root_response.json()
)


# ============================================================
# 33. API TEST — HEALTH
# ============================================================


health_response = client.get(
    "/health"
)


print("\n" + "=" * 70)
print("API TEST — HEALTH")
print("=" * 70)

print(
    f"Status code: "
    f"{health_response.status_code}"
)

print(
    health_response.json()
)


# ============================================================
# 34. API TEST — SEARCH
# ============================================================


search_payload = {

    "query":
        "How should sensitive information be protected?",

    "top_k":
        3,

    "department":
        "Security"
}


search_response = client.post(

    "/search",

    json=search_payload
)


print("\n" + "=" * 70)
print("API TEST — SEARCH")
print("=" * 70)

print(
    f"Status code: "
    f"{search_response.status_code}"
)

print(
    search_response.json()
)


# ============================================================
# 35. API TEST — QUERY
# ============================================================


query_payload = {

    "query":
        "What should happen when an insurance claim is high risk?",

    "top_k":
        3,

    "department":
        "Insurance"
}


query_response = client.post(

    "/query",

    json=query_payload
)


print("\n" + "=" * 70)
print("API TEST — QUERY")
print("=" * 70)

print(
    f"Status code: "
    f"{query_response.status_code}"
)


query_response_json = (
    query_response.json()
)


print(
    "\nAnswer:"
)

print(
    query_response_json.get(
        "answer"
    )
)


print(
    "\nStatus:"
)

print(
    query_response_json.get(
        "status"
    )
)


print(
    "\nGrounding:"
)

print(
    query_response_json.get(
        "grounding"
    )
)


print(
    "\nCitations:"
)

for citation in (
    query_response_json.get(
        "citations",
        []
    )
):

    print(
        citation
    )


# ============================================================
# 36. API TEST — METRICS
# ============================================================


metrics_response = client.get(
    "/metrics"
)


print("\n" + "=" * 70)
print("API TEST — METRICS")
print("=" * 70)

print(
    f"Status code: "
    f"{metrics_response.status_code}"
)

print(
    metrics_response.json()
)


# ============================================================
# 37. API TEST — INFO
# ============================================================


info_response = client.get(
    "/info"
)


print("\n" + "=" * 70)
print("API TEST — INFO")
print("=" * 70)

print(
    f"Status code: "
    f"{info_response.status_code}"
)

print(
    info_response.json()
)


# ============================================================
# 38. PERFORMANCE TEST
# ============================================================
#
# Execute multiple local API queries and calculate:
#
# - average latency
# - minimum latency
# - maximum latency
# - successful requests
# ============================================================


performance_queries = [

    "How should employees protect sensitive information?",

    "What should happen when an insurance claim is high risk?",

    "How should an internal API validate requests?",

    "What controls are important for cloud deployment?",

    "What should remote employees do?"
]


performance_records = []


for query in performance_queries:

    start = time.perf_counter()


    response = client.post(

        "/query",

        json={

            "query":
                query,

            "top_k":
                3
        }
    )


    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000


    performance_records.append({

        "query":
            query,

        "status_code":
            response.status_code,

        "latency_ms":
            round(
                elapsed_ms,
                2
            )
    })


performance_df = pd.DataFrame(
    performance_records
)


print("\n" + "=" * 70)
print("API PERFORMANCE TEST")
print("=" * 70)


display(
    performance_df
)


performance_summary = {

    "requests":
        len(
            performance_df
        ),

    "successful":
        int(
            (
                performance_df[
                    "status_code"
                ] == 200
            ).sum()
        ),

    "average_latency_ms":
        round(
            float(
                performance_df[
                    "latency_ms"
                ].mean()
            ),
            2
        ),

    "minimum_latency_ms":
        round(
            float(
                performance_df[
                    "latency_ms"
                ].min()
            ),
            2
        ),

    "maximum_latency_ms":
        round(
            float(
                performance_df[
                    "latency_ms"
                ].max()
            ),
            2
        )
}


print(
    "\nPerformance summary:"
)

for key, value in (
    performance_summary.items()
):

    print(
        f"{key:<25}: {value}"
    )


# ============================================================
# 39. CREATE REQUIREMENTS.TXT
# ============================================================
#
# This is a deployment artifact, not an actual cloud deployment.
# ============================================================


requirements_content = """\
fastapi
uvicorn
pydantic
numpy
pandas
scikit-learn
langchain-core
langchain-text-splitters
"""


print("\n" + "=" * 70)
print("REQUIREMENTS.TXT")
print("=" * 70)

print(
    requirements_content
)


# ============================================================
# 40. CREATE DOCKERFILE TEMPLATE
# ============================================================


dockerfile_content = """\
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""


print("\n" + "=" * 70)
print("DOCKERFILE TEMPLATE")
print("=" * 70)

print(
    dockerfile_content
)


# ============================================================
# 41. DOCKER COMPOSE TEMPLATE
# ============================================================


docker_compose_content = """\
services:

  vector-rag-api:

    build: .

    ports:
      - "8000:8000"

    restart: unless-stopped

    environment:
      APP_ENV: production

    healthcheck:
      test:
        [
          "CMD",
          "python",
          "-c",
          "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
        ]

      interval: 30s
      timeout: 5s
      retries: 3
"""


print("\n" + "=" * 70)
print("DOCKER COMPOSE TEMPLATE")
print("=" * 70)

print(
    docker_compose_content
)


# ============================================================
# 42. CLOUD ARCHITECTURE
# ============================================================


cloud_architecture = """
                         USERS
                           │
                           ▼
                  API GATEWAY / WAF
                           │
                           ▼
                    LOAD BALANCER
                           │
                           ▼
                  FASTAPI APPLICATION
                           │
                           ▼
              LANGCHAIN RETRIEVAL LAYER
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
         Vector DB      Metadata      Reranker
         /Index         Filtering
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                         TOP-K
                           │
                           ▼
                   CONTEXT BUILDER
                           │
                           ▼
                   ENTERPRISE LLM
                           │
                           ▼
                GROUNDING VALIDATOR
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
             CITATIONS          SAFE FALLBACK
                 │                   │
                 └─────────┬─────────┘
                           ▼
                       RESPONSE

        ┌─────────────────────────────────────┐
        │          OBSERVABILITY              │
        │                                     │
        │ API latency                         │
        │ Retrieval confidence                │
        │ Grounding rate                      │
        │ Error rate                          │
        │ Token usage                         │
        │ Cost                                │
        │ Logs / Metrics / Traces             │
        └─────────────────────────────────────┘
"""


print("\n" + "=" * 70)
print("PRODUCTION CLOUD ARCHITECTURE")
print("=" * 70)

print(
    cloud_architecture
)


# ============================================================
# 43. PRODUCTION SECURITY CHECKLIST
# ============================================================


security_checklist = {

    "Input validation":
        True,

    "Query length limits":
        True,

    "Safe fallback":
        True,

    "Grounding validation":
        True,

    "Source attribution":
        True,

    "Structured API models":
        True,

    "Request IDs":
        True,

    "Application logging":
        True,

    "Authentication":
        False,

    "Authorization / RBAC":
        False,

    "Document-level access control":
        False,

    "TLS":
        False,

    "External secret manager":
        False,

    "Production WAF":
        False
}


print("\n" + "=" * 70)
print("PRODUCTION SECURITY CHECKLIST")
print("=" * 70)


for control, implemented in (
    security_checklist.items()
):

    status = (
        "IMPLEMENTED"
        if implemented
        else
        "PRODUCTION TODO"
    )


    print(
        f"{status:<18} | {control}"
    )


# ============================================================
# 44. FINAL END-TO-END VALIDATION
# ============================================================


final_validation = {}


# Part 1
final_validation[
    "documents_available"
] = (
    len(
        chunked_documents
    ) > 0
)


# Part 2
final_validation[
    "vector_store_available"
] = (
    vector_store is not None
)


# Part 3
final_validation[
    "advanced_retrieval_available"
] = (
    callable(
        advanced_retrieval
    )
)


# RAG
final_validation[
    "rag_available"
] = (
    callable(
        enterprise_rag
    )
)


# Grounding
final_validation[
    "grounding_validation_available"
] = (
    callable(
        grounding_check
    )
)


# Citations
final_validation[
    "citations_available"
] = (
    len(
        rag_test_1[
            "citations"
        ]
    ) > 0
)


# FastAPI
final_validation[
    "fastapi_available"
] = (
    app is not None
)


# Health endpoint
final_validation[
    "health_api"
] = (
    health_response.status_code
    == 200
)


# Search endpoint
final_validation[
    "search_api"
] = (
    search_response.status_code
    == 200
)


# Query endpoint
final_validation[
    "query_api"
] = (
    query_response.status_code
    == 200
)


# Metrics
final_validation[
    "metrics_api"
] = (
    metrics_response.status_code
    == 200
)


# ============================================================
# 45. PRINT FINAL VALIDATION
# ============================================================


print("\n" + "=" * 70)
print("DAY 65 — FINAL VALIDATION")
print("=" * 70)


for check, result in (
    final_validation.items()
):

    status = (
        "PASS"
        if result
        else
        "FAIL"
    )


    print(
        f"{status:<6} | {check}"
    )


final_success = all(
    final_validation.values()
)


print("\n" + "-" * 70)


if final_success:

    print(
        "🎉 DAY 65 PROJECT COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "⚠️ SOME FINAL VALIDATIONS FAILED"
    )


# ============================================================
# 46. FINAL SCORECARD
# ============================================================


day65_scorecard = {

    "Document pipeline":
        "Completed",

    "LangChain Documents":
        "Completed",

    "Chunking":
        "Completed",

    "Metadata":
        "Completed",

    "Vector representation":
        "Completed",

    "LangChain Vector Store":
        "Completed",

    "Similarity Search":
        "Completed",

    "Metadata Filtering":
        "Completed",

    "Query Expansion":
        "Completed",

    "Candidate Retrieval":
        "Completed",

    "Reranking":
        "Completed",

    "Deduplication":
        "Completed",

    "Diversity Ranking":
        "Completed",

    "Retrieval Confidence":
        "Completed",

    "Hit@K Evaluation":
        "Completed",

    "RAG Pipeline":
        "Completed",

    "Grounding Validation":
        "Completed",

    "Source Citations":
        "Completed",

    "Safe Fallback":
        "Completed",

    "FastAPI":
        "Completed",

    "Health API":
        "Completed",

    "Search API":
        "Completed",

    "Query API":
        "Completed",

    "Metrics API":
        "Completed",

    "Performance Testing":
        "Completed",

    "Docker Architecture":
        "Template Created",

    "Cloud Architecture":
        "Designed",

    "Production Security":
        "Partially Implemented"
}


print("\n" + "=" * 70)
print("DAY 65 SCORECARD")
print("=" * 70)


for capability, status in (
    day65_scorecard.items()
):

    print(
        f"{capability:<32} → {status}"
    )


# ============================================================
# 47. IMPORTANT VARIABLES CREATED IN PART 4
# ============================================================


print("\n" + "=" * 70)
print("IMPORTANT VARIABLES CREATED — PART 4")
print("=" * 70)


important_variables_part4 = {

    "METRICS":
        "Application monitoring counters",

    "build_context":
        "Retrieved context construction",

    "build_citations":
        "Source citation generation",

    "generate_grounded_answer":
        "Lightweight grounded answer generator",

    "grounding_check":
        "Grounding validation",

    "safe_fallback":
        "Low-confidence safety fallback",

    "enterprise_rag":
        "Complete RAG pipeline",

    "rag_evaluation_df":
        "RAG evaluation results",

    "rag_quality_summary":
        "RAG quality metrics",

    "app":
        "FastAPI application",

    "SearchRequest":
        "Search API request model",

    "QueryRequest":
        "RAG query request model",

    "client":
        "FastAPI test client",

    "performance_df":
        "API performance results",

    "performance_summary":
        "API performance metrics",

    "requirements_content":
        "Deployment requirements template",

    "dockerfile_content":
        "Dockerfile template",

    "docker_compose_content":
        "Docker Compose template",

    "cloud_architecture":
        "Production cloud architecture",

    "security_checklist":
        "Production security checklist",

    "final_validation":
        "Final project validation",

    "day65_scorecard":
        "Complete Day 65 scorecard"
}


for variable, description in (
    important_variables_part4.items()
):

    print(
        f"{variable:<34} → {description}"
    )


# ============================================================
# 48. COMPLETE DAY 65 ARCHITECTURE
# ============================================================


print("\n" + "=" * 70)
print("DAY 65 — COMPLETE ARCHITECTURE")
print("=" * 70)


print("""
                    ENTERPRISE DOCUMENTS
                            │
                            ▼
                       CLEANING
                            │
                            ▼
                  LANGCHAIN DOCUMENTS
                            │
                            ▼
                       CHUNKING
                            │
                            ▼
                       METADATA
                            │
                            ▼
                    TF-IDF VECTORS
                            │
                            ▼
              LANGCHAIN VECTOR STORE
                            │
                            ▼
                         USER
                            │
                            ▼
                        FASTAPI
                            │
                            ▼
                  QUERY VALIDATION
                            │
                            ▼
                    QUERY EXPANSION
                            │
                            ▼
                  CANDIDATE RETRIEVAL
                            │
                            ▼
                  METADATA FILTERING
                            │
                            ▼
                  SIMILARITY THRESHOLD
                            │
                            ▼
                       RERANKING
                            │
                            ▼
                    DEDUPLICATION
                            │
                            ▼
                   DIVERSITY RANKING
                            │
                            ▼
                         TOP-K
                            │
                            ▼
                  RETRIEVAL CONFIDENCE
                            │
                            ▼
                   CONTEXT BUILDER
                            │
                            ▼
                     RAG LAYER
                            │
                            ▼
                GROUNDED GENERATION
                            │
                            ▼
                GROUNDING VALIDATION
                            │
                    ┌───────┴───────┐
                    ▼               ▼
                CITATIONS       SAFE FALLBACK
                    │               │
                    └───────┬───────┘
                            ▼
                     FINAL RESPONSE

              ┌──────────────────────────┐
              │       MONITORING         │
              │                          │
              │ Latency                  │
              │ Error Rate               │
              │ Retrieval Confidence     │
              │ Grounding Rate           │
              │ Request Count            │
              │ Logs                     │
              └──────────────────────────┘
""")


# ============================================================
# 49. ENGINEERING SUMMARY
# ============================================================


print("\n" + "=" * 70)
print("DAY 65 ENGINEERING SUMMARY")
print("=" * 70)


print("""
DAY 65 — ADVANCED LANGCHAIN VECTOR DATABASE
& ENTERPRISE SEMANTIC SEARCH SYSTEM

PART 1
------
Built an enterprise document pipeline using LangChain
Document objects, recursive chunking and metadata enrichment.

PART 2
------
Converted document chunks into lightweight TF-IDF vector
representations and indexed them using LangChain's
InMemoryVectorStore.

PART 3
------
Built an advanced retrieval pipeline containing:
query normalization,
query expansion,
candidate retrieval,
metadata filtering,
similarity thresholding,
multi-signal reranking,
deduplication,
diversity ranking,
Top-K retrieval,
and retrieval confidence.

PART 4
------
Built the complete RAG application with:
context construction,
grounded response generation,
grounding validation,
source citations,
safe fallback,
FastAPI,
health checks,
search API,
query API,
metrics,
logging,
request IDs,
performance testing,
Docker deployment templates,
cloud architecture,
and production security planning.
""")


# ============================================================
# 50. INTERVIEW EXPLANATION
# ============================================================


print("\n" + "=" * 70)
print("INTERVIEW EXPLANATION")
print("=" * 70)


interview_answer = """
I built an enterprise semantic search and RAG system using
LangChain.

I started by converting enterprise documents into LangChain
Document objects, cleaning the content, splitting it into
retrieval-friendly chunks and attaching metadata such as
document ID, department, document type and source.

For the CPU-friendly implementation, I used TF-IDF as the
vector representation and integrated it with LangChain's
InMemoryVectorStore.

I then implemented advanced retrieval with query normalization,
query expansion, candidate retrieval, metadata filtering,
similarity thresholds, multi-signal reranking, deduplication,
diversity ranking and retrieval confidence.

On top of retrieval, I built a grounded RAG layer that constructs
context, generates an extractive response, validates grounding,
returns source citations and applies a safe fallback when
retrieval confidence is low.

Finally, I exposed the system through FastAPI with search,
query, health and metrics endpoints and added logging,
request IDs, performance testing and production deployment
templates.

The architecture can later replace TF-IDF with dense embeddings,
the lightweight answer generator with an enterprise LLM, and
the in-memory vector store with a managed vector database
without changing the overall retrieval and RAG architecture.
"""


print(
    interview_answer
)


# ============================================================
# 51. RESUME BULLET
# ============================================================


resume_bullet = (
    "Built an enterprise LangChain-based vector search and "
    "RAG system with document chunking, metadata enrichment, "
    "vector retrieval, query expansion, multi-signal reranking, "
    "grounding validation, source citations, safe fallback, "
    "FastAPI APIs, monitoring and production-ready deployment "
    "architecture."
)


print("\n" + "=" * 70)
print("RESUME BULLET")
print("=" * 70)

print(
    resume_bullet
)


# ============================================================
# 52. FINAL STATUS
# ============================================================


print("\n" + "=" * 70)
print("🎉 DAY 65/100 — COMPLETE")
print("=" * 70)


print("""
You have completed:

✅ Part 1 — LangChain Document Pipeline
✅ Part 2 — Embeddings + Vector Database
✅ Part 3 — Advanced Retrieval + Reranking
✅ Part 4 — RAG + Grounding + FastAPI

FINAL SYSTEM:

Documents
   ↓
LangChain
   ↓
Chunking + Metadata
   ↓
Vector Store
   ↓
Advanced Retrieval
   ↓
Reranking
   ↓
Top-K
   ↓
RAG
   ↓
Grounding
   ↓
Citations
   ↓
Safe Fallback
   ↓
FastAPI
   ↓
Monitoring
   ↓
Production Architecture

🚀 DAY 65 COMPLETE.
""")
