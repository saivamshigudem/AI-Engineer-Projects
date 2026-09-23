# ============================================================
# DAY 58 — ENTERPRISE DOCUMENT INTELLIGENCE ASSISTANT
# PART 1 — DOCUMENT INGESTION + CHUNKING + VECTOR SEARCH
# ============================================================
#
# Architecture:
#
# Enterprise Documents
#        ↓
# Document Ingestion
#        ↓
# Text Cleaning
#        ↓
# Chunking
#        ↓
# Metadata Creation
#        ↓
# TF-IDF Vector Representation
#        ↓
# Lightweight Vector Index
#        ↓
# Cosine Similarity Search
#        ↓
# Top-K Relevant Chunks
#
# NOTE:
# TF-IDF is used here because the environment is CPU/storage constrained.
# In a production system, TF-IDF can be replaced with dense embeddings
# from models/APIs such as Azure/OpenAI/Sentence Transformers.
#
# ============================================================

import re
import time
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CREATE A SMALL ENTERPRISE DOCUMENT CORPUS
# ============================================================
#
# In a real enterprise application these documents could come from:
#
# - PDF files
# - Word documents
# - SharePoint
# - Confluence
# - Azure Blob Storage
# - Internal knowledge bases
# - Policy documents
# - Technical documentation
#
# For our CPU-friendly project we create a small local corpus.
# ============================================================

documents = [

    {
        "doc_id": "HR-001",
        "title": "Employee Leave Policy",
        "department": "HR",
        "document_type": "Policy",
        "text": """
        Employees are eligible for annual leave based on their employment
        category and tenure. Leave requests should normally be submitted
        through the employee portal at least three working days in advance.
        Emergency leave may be requested through the manager when advance
        notice is not possible. Managers are responsible for approving or
        rejecting leave requests. Employees should check their remaining
        leave balance before submitting a request.
        """
    },

    {
        "doc_id": "HR-002",
        "title": "Remote Work Policy",
        "department": "HR",
        "document_type": "Policy",
        "text": """
        Employees may work remotely when their role and business requirements
        permit remote work. Remote work must be approved by the employee's
        manager. Employees working remotely are expected to maintain normal
        working hours, remain reachable through approved communication tools,
        and protect confidential company information.
        """
    },

    {
        "doc_id": "IT-001",
        "title": "Password Security Policy",
        "department": "IT Security",
        "document_type": "Security Policy",
        "text": """
        Employees must use strong passwords for corporate applications.
        Passwords must not be shared with other employees. Multi-factor
        authentication must be enabled for applications that support it.
        Employees should immediately report suspected credential compromise
        to the security operations team. Corporate passwords should not be
        reused on personal websites.
        """
    },

    {
        "doc_id": "IT-002",
        "title": "Incident Management Procedure",
        "department": "IT",
        "document_type": "Procedure",
        "text": """
        Production incidents must be reported through the approved incident
        management system. Critical incidents should be escalated immediately
        to the incident response team. The incident record should contain the
        affected service, business impact, timestamp, error details, and
        actions already performed. After resolution, the responsible team
        should document the root cause and corrective actions.
        """
    },

    {
        "doc_id": "FIN-001",
        "title": "Employee Expense Policy",
        "department": "Finance",
        "document_type": "Policy",
        "text": """
        Employees may claim eligible business expenses incurred during
        approved business activities. Expense claims should include receipts
        and relevant supporting documentation. Claims must be submitted through
        the expense management system. Managers review submitted expenses
        before finance performs the final validation and reimbursement process.
        """
    },

    {
        "doc_id": "FIN-002",
        "title": "Travel Reimbursement Policy",
        "department": "Finance",
        "document_type": "Policy",
        "text": """
        Business travel expenses may be reimbursed when the travel has been
        approved according to company procedures. Employees should submit
        hotel, transportation, and meal receipts through the expense system.
        International travel may require additional approval. Finance reviews
        the submitted documentation before reimbursement is processed.
        """
    },

    {
        "doc_id": "SEC-001",
        "title": "Data Classification Standard",
        "department": "Information Security",
        "document_type": "Security Standard",
        "text": """
        Company information should be classified according to its sensitivity.
        Public information can be shared externally. Internal information is
        intended for authorized employees. Confidential information requires
        additional access controls. Highly confidential information should
        only be accessed by explicitly authorized personnel and must be stored
        using approved security controls.
        """
    },

    {
        "doc_id": "SEC-002",
        "title": "Security Incident Response",
        "department": "Information Security",
        "document_type": "Procedure",
        "text": """
        Security incidents should be reported immediately to the security
        operations team. Examples include suspected credential theft,
        unauthorized access, malware infections, and data leakage.
        The security team evaluates severity, contains the incident,
        investigates affected systems, and documents remediation actions.
        """
    },

    {
        "doc_id": "DEV-001",
        "title": "API Development Standards",
        "department": "Engineering",
        "document_type": "Engineering Standard",
        "text": """
        Internal APIs should use clear resource-oriented endpoints and
        consistent HTTP status codes. APIs should validate incoming requests
        before processing them. Authentication and authorization must be
        implemented for protected resources. API errors should return
        structured responses that allow clients to understand the failure.
        """
    },

    {
        "doc_id": "DEV-002",
        "title": "Production Deployment Procedure",
        "department": "Engineering",
        "document_type": "Procedure",
        "text": """
        Production deployments must pass automated tests before release.
        Changes should be reviewed and approved before deployment.
        Applications should be deployed through the approved CI/CD pipeline.
        Production deployments should be monitored after release and rollback
        procedures must be available when a deployment introduces critical
        problems.
        """
    },

    {
        "doc_id": "AI-001",
        "title": "AI Application Governance",
        "department": "AI Engineering",
        "document_type": "AI Policy",
        "text": """
        AI applications used in enterprise environments should provide
        appropriate safeguards for sensitive information. AI outputs should
        be evaluated for accuracy and reliability. Applications should
        maintain appropriate logging and monitoring. High-impact decisions
        should include human review where required by organizational policy.
        """
    },

    {
        "doc_id": "AI-002",
        "title": "RAG Application Standard",
        "department": "AI Engineering",
        "document_type": "Engineering Standard",
        "text": """
        Retrieval augmented generation applications should retrieve relevant
        enterprise knowledge before generating an answer. Retrieved context
        should be traceable to source documents. Applications should monitor
        retrieval quality, answer quality, latency, and failure rates.
        Responses should avoid unsupported claims when the required knowledge
        is not available in the retrieved context.
        """
    },

    {
        "doc_id": "BFSI-001",
        "title": "Customer Data Protection",
        "department": "BFSI Compliance",
        "document_type": "Compliance Policy",
        "text": """
        Customer information must be protected from unauthorized access.
        Sensitive customer data should only be accessed for legitimate
        business purposes. Systems processing customer information should
        implement appropriate authentication, authorization, auditing,
        encryption, and monitoring controls.
        """
    },

    {
        "doc_id": "BFSI-002",
        "title": "Financial Application Access Control",
        "department": "BFSI Compliance",
        "document_type": "Security Policy",
        "text": """
        Access to financial applications must follow the principle of least
        privilege. Users should only receive permissions required for their
        responsibilities. Access should be reviewed periodically and removed
        when employees change roles or leave the organization.
        """
    }
]


documents_df = pd.DataFrame(documents)

print("Documents loaded:", len(documents_df))
print("\nDocument distribution:")
print(documents_df["department"].value_counts())

display(documents_df[[
    "doc_id",
    "title",
    "department",
    "document_type"
]])


# ============================================================
# 2. TEXT CLEANING
# ============================================================
#
# Enterprise documents normally contain:
# - extra spaces
# - line breaks
# - special characters
# - formatting artifacts
#
# We normalize the text before chunking.
# ============================================================

def clean_text(text):
    """
    Basic enterprise-document text normalization.
    """

    text = str(text)

    # Replace multiple whitespace characters
    text = re.sub(r"\s+", " ", text)

    # Remove unnecessary leading/trailing whitespace
    text = text.strip()

    return text


documents_df["clean_text"] = documents_df["text"].apply(clean_text)

print("\nExample cleaned document:")
print(documents_df.iloc[0]["clean_text"])


# ============================================================
# 3. CHUNKING
# ============================================================
#
# Why chunk documents?
#
# Sending an entire large document to an AI system is inefficient.
#
# Instead:
#
# Document
#    ↓
# Smaller chunks
#    ↓
# Vectorize each chunk
#    ↓
# Retrieve only relevant chunks
#
# This improves:
# - retrieval precision
# - context size
# - latency
# - traceability
#
# We use a simple word-based chunker for this CPU-friendly project.
# ============================================================

def chunk_text(text, chunk_size=45, overlap=10):
    """
    Split text into overlapping word chunks.

    chunk_size = maximum number of words per chunk
    overlap    = number of words shared between neighboring chunks
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        # Move forward while keeping overlap
        start += chunk_size - overlap

    return chunks


# ============================================================
# 4. BUILD CHUNK DATASET WITH METADATA
# ============================================================

chunk_records = []

for _, row in documents_df.iterrows():

    chunks = chunk_text(
        row["clean_text"],
        chunk_size=45,
        overlap=10
    )

    for chunk_index, chunk in enumerate(chunks):

        chunk_records.append({

            "chunk_id": f'{row["doc_id"]}-CH{chunk_index+1}',

            "doc_id": row["doc_id"],

            "title": row["title"],

            "department": row["department"],

            "document_type": row["document_type"],

            "chunk_index": chunk_index,

            "text": chunk
        })


chunks_df = pd.DataFrame(chunk_records)


print("\nTotal chunks created:", len(chunks_df))

display(
    chunks_df[
        [
            "chunk_id",
            "doc_id",
            "title",
            "department",
            "chunk_index",
            "text"
        ]
    ].head(10)
)


# ============================================================
# 5. CREATE LIGHTWEIGHT VECTOR REPRESENTATIONS
# ============================================================
#
# Production RAG systems commonly use dense embeddings.
#
# Example:
#
# Text
#   ↓
# Embedding Model
#   ↓
# [0.12, -0.44, 0.73, ...]
#
# Here we use TF-IDF because:
#
# - very lightweight
# - CPU friendly
# - no model download
# - no GPU required
# - good for demonstrating vector retrieval
#
# IMPORTANT:
# TF-IDF is NOT a semantic embedding model.
#
# It is a sparse lexical vector representation.
#
# Later we can replace this component with dense embeddings
# without changing the overall RAG architecture.
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)


chunk_vectors = vectorizer.fit_transform(
    chunks_df["text"]
)


print("\nVector index created.")
print("Number of chunks:", chunk_vectors.shape[0])
print("Vector dimensions:", chunk_vectors.shape[1])


# ============================================================
# 6. LIGHTWEIGHT VECTOR STORE
# ============================================================
#
# This class acts as our tiny local vector database.
#
# It stores:
#
# - chunk vectors
# - chunk metadata
#
# and performs:
#
# query → vectorize → similarity → ranking → top-K
# ============================================================

class LightweightVectorStore:

    def __init__(self, vectorizer, vectors, metadata_df):

        self.vectorizer = vectorizer

        self.vectors = vectors

        self.metadata = metadata_df.reset_index(drop=True)


    def search(
        self,
        query,
        top_k=5,
        department=None,
        document_type=None
    ):
        """
        Search the vector index.

        Optional metadata filters can be used before ranking.
        """

        # --------------------------------------------
        # Step 1: Convert query into vector
        # --------------------------------------------

        query_vector = self.vectorizer.transform([query])


        # --------------------------------------------
        # Step 2: Apply metadata filtering
        # --------------------------------------------

        candidate_indices = np.arange(
            len(self.metadata)
        )


        if department is not None:

            mask = (
                self.metadata["department"]
                .str.lower()
                .eq(department.lower())
            )

            candidate_indices = candidate_indices[mask.values]


        if document_type is not None:

            mask = (
                self.metadata["document_type"]
                .str.lower()
                .eq(document_type.lower())
            )

            candidate_indices = candidate_indices[mask.values]


        if len(candidate_indices) == 0:

            return pd.DataFrame()


        # --------------------------------------------
        # Step 3: Calculate cosine similarity
        # --------------------------------------------

        candidate_vectors = self.vectors[
            candidate_indices
        ]

        similarities = cosine_similarity(
            query_vector,
            candidate_vectors
        ).flatten()


        # --------------------------------------------
        # Step 4: Rank results
        # --------------------------------------------

        ranked_positions = np.argsort(
            similarities
        )[::-1]


        ranked_positions = ranked_positions[:top_k]


        # --------------------------------------------
        # Step 5: Build result dataframe
        # --------------------------------------------

        selected_indices = candidate_indices[
            ranked_positions
        ]

        results = self.metadata.iloc[
            selected_indices
        ].copy()


        results["score"] = similarities[
            ranked_positions
        ]


        # Highest score first
        results = results.sort_values(
            "score",
            ascending=False
        ).reset_index(drop=True)


        return results


# ============================================================
# 7. CREATE OUR VECTOR STORE
# ============================================================

vector_store = LightweightVectorStore(
    vectorizer=vectorizer,
    vectors=chunk_vectors,
    metadata_df=chunks_df
)


print("\nVector store ready.")


# ============================================================
# 8. CREATE A SEARCH FUNCTION
# ============================================================

def search_documents(
    query,
    top_k=5,
    department=None,
    document_type=None,
    show_results=True
):

    start_time = time.perf_counter()


    results = vector_store.search(
        query=query,
        top_k=top_k,
        department=department,
        document_type=document_type
    )


    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000


    if show_results:

        print("\n" + "=" * 80)
        print("QUERY")
        print("=" * 80)

        print(query)

        print(
            f"\nSearch latency: {latency_ms:.2f} ms"
        )

        print("\nTOP RESULTS")
        print("-" * 80)


        if results.empty:

            print("No matching documents found.")

        else:

            for i, row in results.iterrows():

                print(
                    f"\nRank: {i + 1}"
                )

                print(
                    f"Score: {row['score']:.4f}"
                )

                print(
                    f"Document: {row['doc_id']}"
                )

                print(
                    f"Title: {row['title']}"
                )

                print(
                    f"Department: {row['department']}"
                )

                print(
                    f"Chunk: {row['chunk_id']}"
                )

                print(
                    f"Text: {row['text']}"
                )

    return results, latency_ms


# ============================================================
# 9. TEST QUERY 1 — HR
# ============================================================

results_1, latency_1 = search_documents(
    "How do employees request annual leave?",
    top_k=3
)


# ============================================================
# 10. TEST QUERY 2 — SECURITY
# ============================================================

results_2, latency_2 = search_documents(
    "What should an employee do if their password is compromised?",
    top_k=3
)


# ============================================================
# 11. TEST QUERY 3 — FINANCE
# ============================================================

results_3, latency_3 = search_documents(
    "How can I claim business travel expenses?",
    top_k=3
)


# ============================================================
# 12. TEST QUERY 4 — ENGINEERING
# ============================================================

results_4, latency_4 = search_documents(
    "What are the requirements for production deployment?",
    top_k=3
)


# ============================================================
# 13. TEST QUERY 5 — AI / RAG
# ============================================================

results_5, latency_5 = search_documents(
    "How should a RAG application monitor retrieval quality?",
    top_k=3
)


# ============================================================
# 14. TEST METADATA FILTERING
# ============================================================
#
# Example:
#
# Search only within Finance documents.
#
# This becomes extremely useful in enterprise RAG systems.
#
# Example production request:
#
# "Search only Finance policies for this question."
# ============================================================

finance_results, finance_latency = search_documents(
    query="How are employee expenses reimbursed?",
    top_k=5,
    department="Finance"
)


# ============================================================
# 15. RETRIEVAL EVALUATION DATASET
# ============================================================
#
# We create a small ground-truth dataset.
#
# In production, this could come from:
#
# - domain experts
# - historical questions
# - evaluation datasets
# - manually labelled queries
#
# Ground truth tells us which document should be retrieved.
# ============================================================

evaluation_queries = [

    {
        "query": "How do employees request annual leave?",
        "expected_doc": "HR-001"
    },

    {
        "query": "Can employees work remotely?",
        "expected_doc": "HR-002"
    },

    {
        "query": "What happens if my corporate password is compromised?",
        "expected_doc": "IT-001"
    },

    {
        "query": "How are production incidents reported?",
        "expected_doc": "IT-002"
    },

    {
        "query": "How do employees submit business expenses?",
        "expected_doc": "FIN-001"
    },

    {
        "query": "What receipts are required for business travel?",
        "expected_doc": "FIN-002"
    },

    {
        "query": "How should confidential information be protected?",
        "expected_doc": "SEC-001"
    },

    {
        "query": "What should happen after a security incident?",
        "expected_doc": "SEC-002"
    },

    {
        "query": "What are the API development standards?",
        "expected_doc": "DEV-001"
    },

    {
        "query": "How should production deployments be handled?",
        "expected_doc": "DEV-002"
    },

    {
        "query": "What safeguards are needed for enterprise AI applications?",
        "expected_doc": "AI-001"
    },

    {
        "query": "How should RAG applications be monitored?",
        "expected_doc": "AI-002"
    },

    {
        "query": "How should customer data be protected?",
        "expected_doc": "BFSI-001"
    },

    {
        "query": "How should access to financial applications be controlled?",
        "expected_doc": "BFSI-002"
    }
]


# ============================================================
# 16. RUN RETRIEVAL EVALUATION
# ============================================================

def evaluate_retrieval(
    evaluation_data,
    top_k=3
):

    evaluation_results = []


    for item in evaluation_data:

        start_time = time.perf_counter()


        results = vector_store.search(
            query=item["query"],
            top_k=top_k
        )


        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000


        retrieved_docs = (
            results["doc_id"].tolist()
            if not results.empty
            else []
        )


        expected_doc = item["expected_doc"]


        hit = (
            expected_doc in retrieved_docs
        )


        rank = None

        if hit:

            rank = (
                retrieved_docs.index(
                    expected_doc
                ) + 1
            )


        evaluation_results.append({

            "query": item["query"],

            "expected_doc": expected_doc,

            "retrieved_docs": retrieved_docs,

            "hit": hit,

            "rank": rank,

            "latency_ms": latency_ms
        })


    return pd.DataFrame(
        evaluation_results
    )


evaluation_df = evaluate_retrieval(
    evaluation_queries,
    top_k=3
)


# ============================================================
# 17. CALCULATE RETRIEVAL METRICS
# ============================================================

hit_rate = (
    evaluation_df["hit"].mean()
)


mean_latency = (
    evaluation_df["latency_ms"].mean()
)


median_latency = (
    evaluation_df["latency_ms"].median()
)


mrr_values = []

for rank in evaluation_df["rank"]:

    if rank is not None:

        mrr_values.append(
            1 / rank
        )

    else:

        mrr_values.append(0)


mrr = np.mean(mrr_values)


print("\n" + "=" * 80)
print("RETRIEVAL EVALUATION")
print("=" * 80)

print(
    f"Queries evaluated : {len(evaluation_df)}"
)

print(
    f"Top-K             : 3"
)

print(
    f"Hit Rate @ 3      : {hit_rate:.2%}"
)

print(
    f"MRR               : {mrr:.4f}"
)

print(
    f"Mean Latency      : {mean_latency:.2f} ms"
)

print(
    f"Median Latency    : {median_latency:.2f} ms"
)


# ============================================================
# 18. DISPLAY EVALUATION DETAILS
# ============================================================

display(
    evaluation_df[
        [
            "query",
            "expected_doc",
            "retrieved_docs",
            "hit",
            "rank",
            "latency_ms"
        ]
    ]
)


# ============================================================
# 19. SIMPLE RETRIEVAL QUALITY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("RETRIEVAL QUALITY SUMMARY")
print("=" * 80)


if hit_rate >= 0.90:

    print("Excellent retrieval coverage on this small evaluation set.")

elif hit_rate >= 0.75:

    print("Good retrieval coverage, but some queries need improvement.")

else:

    print("Retrieval needs improvement.")


print(
    f"\nTop-3 retrieval hit rate: {hit_rate:.2%}"
)

print(
    f"Mean retrieval latency: {mean_latency:.2f} ms"
)


# ============================================================
# 20. INSPECT THE VECTOR SPACE
# ============================================================
#
# This gives us a basic understanding of our representation.
# ============================================================

feature_names = vectorizer.get_feature_names_out()


print("\n" + "=" * 80)
print("VECTOR REPRESENTATION")
print("=" * 80)

print(
    "Vocabulary size:",
    len(feature_names)
)

print(
    "Vector matrix shape:",
    chunk_vectors.shape
)

print(
    "Sparse matrix type:",
    type(chunk_vectors)
)


# ============================================================
# 21. SHOW IMPORTANT VARIABLES CREATED
# ============================================================

print("\n" + "=" * 80)
print("IMPORTANT VARIABLES CREATED")
print("=" * 80)

print("""
documents
    → Original enterprise document corpus

documents_df
    → DataFrame containing enterprise documents

chunks_df
    → Chunked documents with metadata

vectorizer
    → TF-IDF vectorizer

chunk_vectors
    → Vector representation of document chunks

LightweightVectorStore
    → Local vector-search implementation

vector_store
    → Active vector index

search_documents()
    → Main document retrieval function

evaluation_queries
    → Retrieval evaluation dataset

evaluation_df
    → Retrieval evaluation results

hit_rate
    → Top-K retrieval coverage

mrr
    → Mean Reciprocal Rank

mean_latency
    → Average retrieval latency
""")


# ============================================================
# 22. FINAL ARCHITECTURE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PART 1 COMPLETE")
print("=" * 80)

print("""
ENTERPRISE DOCUMENT INTELLIGENCE ASSISTANT

                USER QUERY
                    │
                    ▼
            Query Processing
                    │
                    ▼
             TF-IDF Vector
                    │
                    ▼
        ┌───────────────────────┐
        │   Vector Index        │
        │                       │
        │ Enterprise Chunks     │
        │ + Metadata            │
        └───────────────────────┘
                    │
                    ▼
          Cosine Similarity
                    │
                    ▼
                Top-K
                    │
                    ▼
          Relevant Chunks
                    │
                    ▼
        PART 2 → ADVANCED
             RETRIEVAL
""")

print("""
NEXT PART:

PART 2 — ADVANCED ENTERPRISE RETRIEVAL

We will add:

1. Metadata filtering
2. Department-aware retrieval
3. Top-K tuning
4. Similarity threshold
5. Query expansion
6. Hybrid-style retrieval
7. Lightweight reranking
8. Retrieval evaluation
9. Precision@K
10. Recall@K
11. MRR
12. Retrieval failure analysis

Then Part 3 will convert the retrieved context
into a grounded RAG assistant.
""")

# ============================================================
# DAY 58 — ENTERPRISE DOCUMENT INTELLIGENCE ASSISTANT
# PART 2 — ADVANCED RETRIEVAL + RERANKING + EVALUATION
# ============================================================
#
# CONTINUES FROM PART 1
#
# Variables already available:
#
# documents_df
# chunks_df
# vectorizer
# chunk_vectors
# vector_store
# search_documents()
# evaluation_queries
# evaluation_df
#
# We will NOT recreate those variables.
#
# ============================================================


import re
import time
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. QUERY NORMALIZATION
# ============================================================
#
# Enterprise users may ask:
#
# "How do I submit expenses?"
# "How can I claim an expense?"
# "What is the process for expense reimbursement?"
#
# The wording differs, but the intent can be similar.
#
# We normalize the query before retrieval.
# ============================================================

def normalize_query(query):
    """
    Normalize a user query for retrieval.
    """

    query = str(query).lower()

    # Remove punctuation
    query = re.sub(r"[^a-z0-9\s]", " ", query)

    # Normalize whitespace
    query = re.sub(r"\s+", " ", query)

    return query.strip()


# ============================================================
# TEST QUERY NORMALIZATION
# ============================================================

test_query = "How do I submit business expenses?"

print("Original query:")
print(test_query)

print("\nNormalized query:")
print(normalize_query(test_query))


# ============================================================
# 2. ENTERPRISE QUERY EXPANSION
# ============================================================
#
# Query expansion adds related terminology.
#
# Example:
#
# Query:
# "password compromised"
#
# Expansion:
# password compromised credentials security authentication
#
# This helps lexical retrieval find documents containing related
# enterprise terminology.
#
# In a production system, query expansion could be performed
# using an LLM, synonym service, or domain ontology.
# ============================================================


QUERY_EXPANSION = {

    "password": [
        "password",
        "credential",
        "authentication",
        "login",
        "security"
    ],

    "compromised": [
        "compromised",
        "breach",
        "stolen",
        "unauthorized"
    ],

    "leave": [
        "leave",
        "vacation",
        "absence",
        "time off"
    ],

    "expense": [
        "expense",
        "expenses",
        "reimbursement",
        "claim",
        "receipt"
    ],

    "travel": [
        "travel",
        "business trip",
        "hotel",
        "transportation",
        "meal"
    ],

    "incident": [
        "incident",
        "issue",
        "outage",
        "failure",
        "escalation"
    ],

    "security": [
        "security",
        "protection",
        "access",
        "unauthorized",
        "incident"
    ],

    "deployment": [
        "deployment",
        "release",
        "production",
        "CI/CD",
        "rollback"
    ],

    "api": [
        "API",
        "endpoint",
        "HTTP",
        "authentication",
        "authorization"
    ],

    "rag": [
        "RAG",
        "retrieval",
        "context",
        "grounding",
        "source"
    ],

    "customer": [
        "customer",
        "client",
        "data",
        "sensitive",
        "privacy"
    ],

    "access": [
        "access",
        "permission",
        "authorization",
        "privilege",
        "role"
    ]
}


def expand_query(query):
    """
    Expand query using a lightweight enterprise terminology map.
    """

    normalized = normalize_query(query)

    terms = [normalized]

    for keyword, expansions in QUERY_EXPANSION.items():

        if keyword in normalized:

            terms.extend(expansions)

    # Remove duplicates while preserving order
    unique_terms = list(dict.fromkeys(terms))

    return " ".join(unique_terms)


# ============================================================
# TEST QUERY EXPANSION
# ============================================================

query = "What should I do if my password is compromised?"

print("\nOriginal:")
print(query)

print("\nExpanded:")
print(expand_query(query))


# ============================================================
# 3. TOKENIZATION HELPER
# ============================================================
#
# Used later by the lightweight reranker.
# ============================================================

def tokenize(text):

    text = str(text).lower()

    tokens = re.findall(
        r"\b[a-z0-9]+\b",
        text
    )

    return set(tokens)


# ============================================================
# 4. LIGHTWEIGHT RERANKING
# ============================================================
#
# Initial retrieval uses TF-IDF cosine similarity.
#
# But initial retrieval may return:
#
# 1. Relevant document
# 2. Related document
# 3. Partially relevant document
#
# We now rerank candidates using additional signals.
#
# Final score:
#
#   0.70 * vector similarity
# + 0.20 * keyword overlap
# + 0.10 * phrase match
#
# This is NOT a neural reranker.
#
# It is deliberately lightweight and CPU friendly.
#
# Production alternatives:
#
# - Cross-encoder reranker
# - Cohere rerank
# - Azure AI Search semantic ranker
# - LLM-based reranking
# ============================================================


def keyword_overlap_score(
    query,
    document
):

    query_tokens = tokenize(query)

    document_tokens = tokenize(document)

    if not query_tokens:
        return 0.0

    overlap = (
        query_tokens.intersection(
            document_tokens
        )
    )

    return len(overlap) / len(query_tokens)


def phrase_match_score(
    query,
    document
):

    query = normalize_query(query)

    document = normalize_query(document)

    query_words = query.split()

    if len(query_words) < 2:
        return 0.0

    matched_phrases = 0
    total_phrases = 0

    # Create adjacent two-word phrases
    for i in range(len(query_words) - 1):

        phrase = (
            query_words[i]
            + " "
            + query_words[i + 1]
        )

        total_phrases += 1

        if phrase in document:

            matched_phrases += 1

    if total_phrases == 0:

        return 0.0

    return (
        matched_phrases /
        total_phrases
    )


# ============================================================
# 5. ADVANCED RETRIEVAL FUNCTION
# ============================================================
#
# Pipeline:
#
# Query
#   ↓
# Normalize
#   ↓
# Expand
#   ↓
# Vectorize
#   ↓
# Candidate retrieval
#   ↓
# Similarity threshold
#   ↓
# Keyword scoring
#   ↓
# Phrase scoring
#   ↓
# Combined score
#   ↓
# Rerank
#   ↓
# Top-K
# ============================================================


def advanced_retrieve(
    query,
    top_k=5,
    candidate_k=10,
    similarity_threshold=0.05,
    department=None,
    document_type=None
):

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # STEP 1 — Normalize query
    # --------------------------------------------------------

    normalized_query = normalize_query(
        query
    )


    # --------------------------------------------------------
    # STEP 2 — Expand query
    # --------------------------------------------------------

    expanded_query = expand_query(
        normalized_query
    )


    # --------------------------------------------------------
    # STEP 3 — Convert query into vector
    # --------------------------------------------------------

    query_vector = vectorizer.transform(
        [expanded_query]
    )


    # --------------------------------------------------------
    # STEP 4 — Candidate filtering
    # --------------------------------------------------------

    candidate_indices = np.arange(
        len(chunks_df)
    )


    if department is not None:

        department_mask = (
            chunks_df["department"]
            .str.lower()
            .eq(department.lower())
        )

        candidate_indices = candidate_indices[
            department_mask.values
        ]


    if document_type is not None:

        type_mask = (
            chunks_df["document_type"]
            .str.lower()
            .eq(document_type.lower())
        )

        candidate_indices = candidate_indices[
            type_mask.values
        ]


    if len(candidate_indices) == 0:

        return pd.DataFrame(), 0.0


    # --------------------------------------------------------
    # STEP 5 — Calculate vector similarity
    # --------------------------------------------------------

    candidate_vectors = chunk_vectors[
        candidate_indices
    ]


    similarities = cosine_similarity(
        query_vector,
        candidate_vectors
    ).flatten()


    # --------------------------------------------------------
    # STEP 6 — Apply similarity threshold
    # --------------------------------------------------------

    valid_mask = (
        similarities >=
        similarity_threshold
    )


    filtered_indices = candidate_indices[
        valid_mask
    ]

    filtered_similarities = similarities[
        valid_mask
    ]


    if len(filtered_indices) == 0:

        return pd.DataFrame(), (
            time.perf_counter() -
            start_time
        ) * 1000


    # --------------------------------------------------------
    # STEP 7 — Initial candidate ranking
    # --------------------------------------------------------

    initial_order = np.argsort(
        filtered_similarities
    )[::-1]


    initial_order = initial_order[
        :candidate_k
    ]


    selected_indices = filtered_indices[
        initial_order
    ]


    selected_similarities = (
        filtered_similarities[
            initial_order
        ]
    )


    # --------------------------------------------------------
    # STEP 8 — Build candidate dataframe
    # --------------------------------------------------------

    candidates = chunks_df.iloc[
        selected_indices
    ].copy()


    candidates["vector_score"] = (
        selected_similarities
    )


    # --------------------------------------------------------
    # STEP 9 — Calculate keyword overlap
    # --------------------------------------------------------

    candidates["keyword_score"] = (
        candidates["text"]
        .apply(
            lambda text:
            keyword_overlap_score(
                expanded_query,
                text
            )
        )
    )


    # --------------------------------------------------------
    # STEP 10 — Calculate phrase matching
    # --------------------------------------------------------

    candidates["phrase_score"] = (
        candidates["text"]
        .apply(
            lambda text:
            phrase_match_score(
                normalized_query,
                text
            )
        )
    )


    # --------------------------------------------------------
    # STEP 11 — Combined reranking score
    # --------------------------------------------------------
    #
    # Vector similarity = strongest signal
    # Keyword overlap   = supporting signal
    # Phrase match      = precision signal
    # --------------------------------------------------------

    candidates["final_score"] = (

        0.70 *
        candidates["vector_score"]

        +

        0.20 *
        candidates["keyword_score"]

        +

        0.10 *
        candidates["phrase_score"]
    )


    # --------------------------------------------------------
    # STEP 12 — Final ranking
    # --------------------------------------------------------

    candidates = candidates.sort_values(
        "final_score",
        ascending=False
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # STEP 13 — Top-K
    # --------------------------------------------------------

    results = candidates.head(
        top_k
    ).copy()


    # --------------------------------------------------------
    # STEP 14 — Retrieval latency
    # --------------------------------------------------------

    latency_ms = (
        time.perf_counter() -
        start_time
    ) * 1000


    return results, latency_ms


# ============================================================
# 6. TEST ADVANCED RETRIEVAL — EXPENSE QUERY
# ============================================================

results, latency = advanced_retrieve(
    query="How can I claim business expenses?",
    top_k=5,
    candidate_k=10
)


print("=" * 80)
print("ADVANCED RETRIEVAL")
print("=" * 80)

print("\nQuery:")
print(
    "How can I claim business expenses?"
)

print(
    f"\nLatency: {latency:.2f} ms"
)

print("\nResults:")

display(
    results[
        [
            "chunk_id",
            "doc_id",
            "title",
            "department",
            "vector_score",
            "keyword_score",
            "phrase_score",
            "final_score",
            "text"
        ]
    ]
)


# ============================================================
# 7. TEST ADVANCED RETRIEVAL — SECURITY
# ============================================================

security_results, security_latency = (
    advanced_retrieve(
        query=(
            "What should I do if "
            "my password is compromised?"
        ),
        top_k=5,
        candidate_k=10
    )
)


print("=" * 80)
print("SECURITY RETRIEVAL")
print("=" * 80)

display(
    security_results[
        [
            "chunk_id",
            "doc_id",
            "title",
            "department",
            "vector_score",
            "keyword_score",
            "phrase_score",
            "final_score"
        ]
    ]
)


# ============================================================
# 8. METADATA-AWARE RETRIEVAL
# ============================================================
#
# Enterprise systems often know additional information about
# the user's request.
#
# Example:
#
# Search only Finance documents.
#
# This prevents unrelated HR/IT documents from entering the
# final context.
# ============================================================

finance_results, finance_latency = (
    advanced_retrieve(
        query=(
            "How are expenses "
            "reimbursed?"
        ),
        top_k=5,
        candidate_k=10,
        department="Finance"
    )
)


print("=" * 80)
print("FINANCE FILTERED RETRIEVAL")
print("=" * 80)

display(
    finance_results[
        [
            "chunk_id",
            "doc_id",
            "title",
            "department",
            "final_score",
            "text"
        ]
    ]
)


# ============================================================
# 9. SIMILARITY THRESHOLD TEST
# ============================================================
#
# Important production problem:
#
# What happens when the user asks something unrelated?
#
# Example:
#
# "What is the weather today?"
#
# We don't want the system to retrieve random enterprise
# documents and later hallucinate an answer.
#
# Similarity threshold helps detect weak retrieval.
# ============================================================

unknown_results, unknown_latency = (
    advanced_retrieve(
        query="What is the weather today?",
        top_k=5,
        candidate_k=10,
        similarity_threshold=0.20
    )
)


print("=" * 80)
print("UNKNOWN QUERY TEST")
print("=" * 80)

print(
    "\nQuery: What is the weather today?"
)

print(
    f"\nLatency: {unknown_latency:.2f} ms"
)


if unknown_results.empty:

    print(
        "\nNo sufficiently relevant enterprise "
        "documents were retrieved."
    )

else:

    display(
        unknown_results[
            [
                "doc_id",
                "title",
                "final_score",
                "text"
            ]
        ]
    )


# ============================================================
# 10. BUILD A REUSABLE ENTERPRISE RETRIEVER
# ============================================================
#
# This wrapper provides a cleaner interface for Part 3.
# ============================================================


class EnterpriseRetriever:

    def __init__(
        self,
        top_k=5,
        candidate_k=10,
        similarity_threshold=0.05
    ):

        self.top_k = top_k

        self.candidate_k = candidate_k

        self.similarity_threshold = (
            similarity_threshold
        )


    def retrieve(
        self,
        query,
        department=None,
        document_type=None
    ):

        results, latency = advanced_retrieve(

            query=query,

            top_k=self.top_k,

            candidate_k=self.candidate_k,

            similarity_threshold=(
                self.similarity_threshold
            ),

            department=department,

            document_type=document_type
        )


        return {
            "query": query,

            "results": results,

            "latency_ms": latency,

            "result_count": len(results)
        }


# ============================================================
# 11. CREATE RETRIEVER
# ============================================================

enterprise_retriever = EnterpriseRetriever(
    top_k=3,
    candidate_k=10,
    similarity_threshold=0.05
)


retrieval_response = (
    enterprise_retriever.retrieve(
        "How do I submit an expense claim?"
    )
)


print("=" * 80)
print("ENTERPRISE RETRIEVER")
print("=" * 80)

print(
    "\nQuery:",
    retrieval_response["query"]
)

print(
    "Results:",
    retrieval_response["result_count"]
)

print(
    "Latency:",
    f'{retrieval_response["latency_ms"]:.2f} ms'
)


display(
    retrieval_response["results"][
        [
            "chunk_id",
            "doc_id",
            "title",
            "department",
            "final_score",
            "text"
        ]
    ]
)


# ============================================================
# 12. RETRIEVAL METRICS
# ============================================================
#
# We now evaluate:
#
# Hit Rate
# Precision@K
# Recall@K
# MRR
#
# Definitions:
#
# Hit Rate@K:
# Did the correct document appear in the top K?
#
# Precision@K:
# How many retrieved results are relevant?
#
# Recall@K:
# How many relevant documents did we retrieve?
#
# MRR:
# How high was the first correct result?
# ============================================================


def evaluate_advanced_retrieval(
    evaluation_data,
    top_k=3
):

    rows = []


    for item in evaluation_data:

        start_time = time.perf_counter()


        results, _ = advanced_retrieve(

            query=item["query"],

            top_k=top_k,

            candidate_k=max(
                10,
                top_k * 3
            ),

            similarity_threshold=0.05
        )


        latency_ms = (
            time.perf_counter() -
            start_time
        ) * 1000


        retrieved_docs = (
            results["doc_id"].tolist()
            if not results.empty
            else []
        )


        expected_doc = (
            item["expected_doc"]
        )


        # ----------------------------------------
        # Hit Rate
        # ----------------------------------------

        hit = (
            expected_doc in retrieved_docs
        )


        # ----------------------------------------
        # Rank
        # ----------------------------------------

        rank = None

        if hit:

            rank = (
                retrieved_docs.index(
                    expected_doc
                ) + 1
            )


        # ----------------------------------------
        # Precision@K
        # ----------------------------------------
        #
        # Since our evaluation set has one known
        # relevant document:
        #
        # relevant retrieved result = 1
        # otherwise = 0
        # ----------------------------------------

        relevant_count = (
            1 if hit else 0
        )


        precision_at_k = (
            relevant_count /
            top_k
        )


        # ----------------------------------------
        # Recall@K
        # ----------------------------------------
        #
        # One expected document exists.
        # ----------------------------------------

        recall_at_k = (
            1.0 if hit else 0.0
        )


        # ----------------------------------------
        # Reciprocal Rank
        # ----------------------------------------

        reciprocal_rank = (
            1 / rank
            if rank is not None
            else 0
        )


        rows.append({

            "query": item["query"],

            "expected_doc": expected_doc,

            "retrieved_docs": retrieved_docs,

            "hit": hit,

            "rank": rank,

            "precision_at_k": (
                precision_at_k
            ),

            "recall_at_k": (
                recall_at_k
            ),

            "reciprocal_rank": (
                reciprocal_rank
            ),

            "latency_ms": latency_ms
        })


    return pd.DataFrame(rows)


# ============================================================
# 13. RUN ADVANCED EVALUATION
# ============================================================

advanced_evaluation_df = (
    evaluate_advanced_retrieval(
        evaluation_queries,
        top_k=3
    )
)


# ============================================================
# 14. CALCULATE FINAL METRICS
# ============================================================

advanced_hit_rate = (
    advanced_evaluation_df["hit"].mean()
)


advanced_precision = (
    advanced_evaluation_df[
        "precision_at_k"
    ].mean()
)


advanced_recall = (
    advanced_evaluation_df[
        "recall_at_k"
    ].mean()
)


advanced_mrr = (
    advanced_evaluation_df[
        "reciprocal_rank"
    ].mean()
)


advanced_mean_latency = (
    advanced_evaluation_df[
        "latency_ms"
    ].mean()
)


advanced_median_latency = (
    advanced_evaluation_df[
        "latency_ms"
    ].median()
)


# ============================================================
# 15. DISPLAY ADVANCED RETRIEVAL METRICS
# ============================================================

print("\n" + "=" * 80)
print("ADVANCED RETRIEVAL EVALUATION")
print("=" * 80)

print(
    f"\nQueries evaluated : "
    f"{len(advanced_evaluation_df)}"
)

print(
    f"Top-K             : 3"
)

print(
    f"Hit Rate @ 3      : "
    f"{advanced_hit_rate:.2%}"
)

print(
    f"Precision @ 3     : "
    f"{advanced_precision:.2%}"
)

print(
    f"Recall @ 3        : "
    f"{advanced_recall:.2%}"
)

print(
    f"MRR               : "
    f"{advanced_mrr:.4f}"
)

print(
    f"Mean Latency      : "
    f"{advanced_mean_latency:.2f} ms"
)

print(
    f"Median Latency    : "
    f"{advanced_median_latency:.2f} ms"
)


# ============================================================
# 16. DISPLAY DETAILED EVALUATION
# ============================================================

display(
    advanced_evaluation_df[
        [
            "query",
            "expected_doc",
            "retrieved_docs",
            "hit",
            "rank",
            "precision_at_k",
            "recall_at_k",
            "reciprocal_rank",
            "latency_ms"
        ]
    ]
)


# ============================================================
# 17. BASELINE VS ADVANCED RETRIEVAL
# ============================================================
#
# Part 1:
#
# Query
#   ↓
# TF-IDF
#   ↓
# Cosine similarity
#   ↓
# Top-K
#
#
# Part 2:
#
# Query
#   ↓
# Normalization
#   ↓
# Query expansion
#   ↓
# TF-IDF
#   ↓
# Candidate retrieval
#   ↓
# Metadata filtering
#   ↓
# Similarity threshold
#   ↓
# Keyword scoring
#   ↓
# Phrase scoring
#   ↓
# Reranking
#   ↓
# Top-K
#
# Let's compare the retrieval metrics.
# ============================================================


baseline_hit_rate = (
    evaluation_df["hit"].mean()
)

baseline_mrr = np.mean([
    1 / rank
    if rank is not None
    else 0
    for rank in evaluation_df["rank"]
])


comparison_df = pd.DataFrame({

    "Metric": [

        "Hit Rate @ 3",

        "MRR",

        "Mean Latency (ms)"
    ],

    "Part 1 - Baseline": [

        baseline_hit_rate,

        baseline_mrr,

        evaluation_df[
            "latency_ms"
        ].mean()
    ],

    "Part 2 - Advanced": [

        advanced_hit_rate,

        advanced_mrr,

        advanced_mean_latency
    ]
})


print("\n" + "=" * 80)
print("BASELINE VS ADVANCED RETRIEVAL")
print("=" * 80)

display(comparison_df)


# ============================================================
# 18. RETRIEVAL FAILURE ANALYSIS
# ============================================================
#
# This is extremely important in real RAG systems.
#
# If retrieval fails, we need to know which queries failed.
# ============================================================


failed_queries = (
    advanced_evaluation_df[
        advanced_evaluation_df["hit"] == False
    ]
)


print("\n" + "=" * 80)
print("RETRIEVAL FAILURE ANALYSIS")
print("=" * 80)


if failed_queries.empty:

    print(
        "\nNo retrieval failures detected "
        "in this evaluation dataset."
    )

else:

    print(
        f"\nFailed queries: "
        f"{len(failed_queries)}"
    )

    display(
        failed_queries[
            [
                "query",
                "expected_doc",
                "retrieved_docs",
                "latency_ms"
            ]
        ]
    )


# ============================================================
# 19. TOP-K ANALYSIS
# ============================================================
#
# We test different K values.
#
# This helps answer:
#
# "How many chunks should I send to the LLM?"
#
# Too small:
#     Relevant context may be missed.
#
# Too large:
#     More noise + more tokens + higher latency.
# ============================================================


k_results = []


for k in [1, 2, 3, 5]:

    temp_df = (
        evaluate_advanced_retrieval(
            evaluation_queries,
            top_k=k
        )
    )


    k_results.append({

        "K": k,

        "Hit Rate": (
            temp_df["hit"].mean()
        ),

        "MRR": (
            temp_df[
                "reciprocal_rank"
            ].mean()
        ),

        "Mean Latency (ms)": (
            temp_df[
                "latency_ms"
            ].mean()
        )
    })


top_k_analysis_df = pd.DataFrame(
    k_results
)


print("\n" + "=" * 80)
print("TOP-K ANALYSIS")
print("=" * 80)

display(
    top_k_analysis_df
)


# ============================================================
# 20. RETRIEVAL CONTEXT BUILDER
# ============================================================
#
# Part 3 will need a clean context object.
#
# We create it now.
#
# IMPORTANT:
# The generator should know where information came from.
# This enables source citations and grounded responses.
# ============================================================


def build_retrieval_context(
    retrieval_results
):

    if retrieval_results.empty:

        return ""


    context_parts = []


    for _, row in retrieval_results.iterrows():

        context_parts.append(

            f"""
SOURCE ID: {row['chunk_id']}
DOCUMENT: {row['title']}
DOCUMENT ID: {row['doc_id']}
DEPARTMENT: {row['department']}
DOCUMENT TYPE: {row['document_type']}

CONTENT:
{row['text']}
""".strip()
        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# 21. TEST CONTEXT BUILDER
# ============================================================

context_results, context_latency = (
    advanced_retrieve(
        query=(
            "How should a RAG "
            "application be monitored?"
        ),
        top_k=3,
        candidate_k=10
    )
)


retrieval_context = (
    build_retrieval_context(
        context_results
    )
)


print("\n" + "=" * 80)
print("RETRIEVED CONTEXT FOR FUTURE RAG GENERATION")
print("=" * 80)

print(retrieval_context)


# ============================================================
# 22. FINAL RETRIEVER HEALTH CHECK
# ============================================================

print("\n" + "=" * 80)
print("PART 2 RETRIEVER HEALTH CHECK")
print("=" * 80)


health_checks = {

    "Vector index available":
        chunk_vectors.shape[0] > 0,

    "Metadata available":
        len(chunks_df) > 0,

    "Query expansion available":
        len(QUERY_EXPANSION) > 0,

    "Reranking available":
        callable(
            keyword_overlap_score
        ),

    "Advanced retrieval available":
        callable(
            advanced_retrieve
        ),

    "Enterprise retriever available":
        callable(
            enterprise_retriever.retrieve
        ),

    "Evaluation completed":
        len(
            advanced_evaluation_df
        ) > 0
}


for check, status in health_checks.items():

    print(
        f"{'PASS' if status else 'FAIL'} : "
        f"{check}"
    )


# ============================================================
# 23. IMPORTANT VARIABLES CREATED
# ============================================================

print("\n" + "=" * 80)
print("IMPORTANT VARIABLES CREATED IN PART 2")
print("=" * 80)

print("""
QUERY_EXPANSION
    → Enterprise terminology expansion dictionary

normalize_query()
    → Query normalization

expand_query()
    → Query expansion

keyword_overlap_score()
    → Keyword relevance score

phrase_match_score()
    → Phrase matching score

advanced_retrieve()
    → Complete advanced retrieval pipeline

EnterpriseRetriever
    → Reusable production-style retrieval interface

enterprise_retriever
    → Configured enterprise retriever

advanced_evaluation_df
    → Advanced retrieval evaluation results

advanced_hit_rate
    → Hit Rate @ K

advanced_precision
    → Precision @ K

advanced_recall
    → Recall @ K

advanced_mrr
    → Mean Reciprocal Rank

advanced_mean_latency
    → Average advanced retrieval latency

top_k_analysis_df
    → K-value experiment results

retrieval_context
    → Retrieved context ready for RAG generation
""")


# ============================================================
# 24. FINAL ARCHITECTURE
# ============================================================

print("\n" + "=" * 80)
print("PART 2 COMPLETE")
print("=" * 80)

print("""
ENTERPRISE DOCUMENT INTELLIGENCE ASSISTANT
ADVANCED RETRIEVAL ARCHITECTURE

                    USER QUERY
                        │
                        ▼
               Query Normalization
                        │
                        ▼
                 Query Expansion
                        │
                        ▼
                 Query Vector
                        │
                        ▼
              Candidate Retrieval
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
      Metadata Filter       Similarity Score
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
                Candidate Chunks
                        │
                        ▼
                 Lightweight
                   Reranking
                        │
             ┌──────────┼──────────┐
             │          │          │
             ▼          ▼          ▼
          Vector     Keyword     Phrase
          Score       Score       Score
             │          │          │
             └──────────┼──────────┘
                        │
                        ▼
                  Final Score
                        │
                        ▼
                      Top-K
                        │
                        ▼
               Retrieved Context
                        │
                        ▼
              PART 3 — RAG GENERATION
""")


# ============================================================
# 25. INTERVIEW-READY SUMMARY
# ============================================================

print("""
INTERVIEW EXPLANATION:

"I designed the retrieval layer as a two-stage pipeline.

First, I perform lightweight candidate retrieval using vector
similarity. I then apply metadata filtering and a similarity
threshold to remove weak candidates.

For the remaining candidates, I perform a lightweight reranking
step using vector similarity, keyword overlap, and phrase matching.

I also introduced query normalization and domain-specific query
expansion to improve retrieval for enterprise terminology.

Finally, I evaluate the retriever using Hit Rate@K, Precision@K,
Recall@K, MRR, and retrieval latency.

The retrieval layer is intentionally separated from the generation
layer, so the TF-IDF representation can later be replaced with
dense embeddings and the lightweight reranker can be replaced by
a neural cross-encoder or enterprise search reranker without
changing the overall architecture."
""")


# ============================================================
# END OF PART 2
# ============================================================
# ============================================================
# DAY 58 — ENTERPRISE DOCUMENT INTELLIGENCE ASSISTANT
# PART 3 — GROUNDED RAG + CITATIONS + EVALUATION
# ============================================================
#
# CONTINUES DIRECTLY FROM PART 1 + PART 2
#
# Existing important variables:
#
# documents_df
# chunks_df
# vectorizer
# chunk_vectors
# vector_store
# enterprise_retriever
# advanced_retrieve()
# build_retrieval_context()
#
# We will NOT recreate those.
#
# ============================================================


import re
import time
import numpy as np
import pandas as pd


# ============================================================
# 1. RAG CONFIGURATION
# ============================================================
#
# Central configuration for the generation layer.
#
# These values can later be moved into environment variables
# or a configuration file in the production version.
# ============================================================

RAG_CONFIG = {

    "top_k": 3,

    "candidate_k": 10,

    "similarity_threshold": 0.05,

    "min_grounding_score": 0.35,

    "max_context_chunks": 3,

    "max_answer_sentences": 4
}


print("=" * 80)
print("RAG CONFIGURATION")
print("=" * 80)

for key, value in RAG_CONFIG.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# 2. ENTERPRISE RESPONSE SCHEMA
# ============================================================
#
# A production RAG API should return structured information.
#
# Example:
#
# {
#     answer,
#     sources,
#     confidence,
#     grounded,
#     retrieval_latency,
#     generation_latency
# }
#
# This is much more useful than returning plain text.
# ============================================================

def create_response(
    query,
    answer,
    sources,
    grounded,
    confidence,
    retrieval_latency_ms,
    generation_latency_ms,
    total_latency_ms,
    status="success"
):

    return {

        "query": query,

        "answer": answer,

        "sources": sources,

        "grounded": grounded,

        "confidence": confidence,

        "retrieval_latency_ms":
            round(
                retrieval_latency_ms,
                2
            ),

        "generation_latency_ms":
            round(
                generation_latency_ms,
                2
            ),

        "total_latency_ms":
            round(
                total_latency_ms,
                2
            ),

        "status": status
    }


# ============================================================
# 3. BUILD GROUNDED CONTEXT
# ============================================================
#
# The LLM should never receive anonymous text.
#
# Every chunk carries:
#
# - source ID
# - document ID
# - title
# - department
# - document type
# - content
#
# This allows the final response to cite the source.
# ============================================================

def build_grounded_context(
    retrieval_results,
    max_chunks=3
):

    if retrieval_results is None:

        return ""


    if retrieval_results.empty:

        return ""


    results = retrieval_results.head(
        max_chunks
    )


    context_blocks = []


    for _, row in results.iterrows():

        block = f"""
SOURCE_ID: {row['chunk_id']}
DOCUMENT_ID: {row['doc_id']}
TITLE: {row['title']}
DEPARTMENT: {row['department']}
DOCUMENT_TYPE: {row['document_type']}
RELEVANCE_SCORE: {row['final_score']:.4f}

CONTENT:
{row['text']}
""".strip()


        context_blocks.append(
            block
        )


    return "\n\n---\n\n".join(
        context_blocks
    )


# ============================================================
# 4. CREATE A GROUNDED PROMPT
# ============================================================
#
# This is the prompt structure that we would send to an LLM
# in a production implementation.
#
# The current notebook does not call an external LLM because
# we are keeping it CPU/storage friendly.
#
# Part 3 therefore separates:
#
# RETRIEVAL
#     ↓
# PROMPT CONSTRUCTION
#     ↓
# GENERATION
#
# This separation is important for production architecture.
# ============================================================

SYSTEM_PROMPT = """
You are an enterprise document assistant.

Answer the user's question using ONLY the provided enterprise
context.

Rules:

1. Do not invent information.
2. Do not use knowledge outside the provided context.
3. If the context does not contain enough information, say that
   the available enterprise documents do not provide enough
   information.
4. Cite the source ID for factual claims.
5. Keep the answer concise and professional.
6. Never fabricate a policy, procedure, number, or requirement.
"""


def build_rag_prompt(
    query,
    context
):

    prompt = f"""
SYSTEM INSTRUCTIONS
{SYSTEM_PROMPT.strip()}

ENTERPRISE CONTEXT
{context if context else "NO RELEVANT CONTEXT FOUND"}

USER QUESTION
{query}

ANSWER
""".strip()


    return prompt


# ============================================================
# 5. TEST PROMPT CONSTRUCTION
# ============================================================

test_results, test_latency = (
    advanced_retrieve(
        query=(
            "How should a RAG application "
            "be monitored?"
        ),
        top_k=3,
        candidate_k=10
    )
)


test_context = (
    build_grounded_context(
        test_results
    )
)


test_prompt = build_rag_prompt(
    query=(
        "How should a RAG application "
        "be monitored?"
    ),
    context=test_context
)


print("=" * 80)
print("GROUNDED RAG PROMPT")
print("=" * 80)

print(test_prompt)


# ============================================================
# 6. EXTRACT SENTENCES FROM RETRIEVED CONTEXT
# ============================================================
#
# Since we are not downloading an LLM, we build a deterministic
# answer generator.
#
# It extracts the most relevant sentences from retrieved chunks.
#
# This gives us a completely local RAG pipeline while preserving
# the architecture of a real LLM-based system.
# ============================================================

def split_sentences(text):

    text = str(text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# 7. GENERATION KEYWORD EXTRACTION
# ============================================================

def query_keywords(query):

    stop_words = {
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "is",
        "are",
        "the",
        "a",
        "an",
        "do",
        "does",
        "can",
        "should",
        "i",
        "we",
        "my",
        "to",
        "for",
        "of",
        "and",
        "in",
        "on",
        "be",
        "it"
    }


    tokens = re.findall(
        r"\b[a-z0-9]+\b",
        normalize_query(query)
    )


    return [
        token
        for token in tokens
        if token not in stop_words
    ]


# ============================================================
# 8. SENTENCE RELEVANCE SCORING
# ============================================================
#
# Each sentence receives a simple relevance score based on
# overlap with the user's query.
# ============================================================

def sentence_relevance_score(
    query,
    sentence
):

    q_tokens = set(
        query_keywords(query)
    )


    s_tokens = tokenize(
        sentence
    )


    if not q_tokens:

        return 0.0


    overlap = (
        q_tokens.intersection(
            s_tokens
        )
    )


    return (
        len(overlap) /
        len(q_tokens)
    )


# ============================================================
# 9. DETERMINE IF CONTEXT IS SUFFICIENT
# ============================================================
#
# This is a critical RAG behavior.
#
# Bad RAG:
#
# Query
#   ↓
# Weak retrieval
#   ↓
# LLM guesses
#   ↓
# Hallucination
#
#
# Good RAG:
#
# Query
#   ↓
# Retrieval
#   ↓
# Check relevance
#   ↓
# If insufficient → "I don't have enough information"
#
# ============================================================

def assess_context_quality(
    retrieval_results
):

    if (
        retrieval_results is None
        or retrieval_results.empty
    ):

        return {
            "sufficient": False,
            "score": 0.0,
            "reason": "No relevant documents retrieved."
        }


    top_score = float(
        retrieval_results.iloc[0][
            "final_score"
        ]
    )


    average_score = float(
        retrieval_results[
            "final_score"
        ].head(3).mean()
    )


    # Weighted quality score
    quality_score = (
        0.7 * top_score
        +
        0.3 * average_score
    )


    sufficient = (
        quality_score >=
        RAG_CONFIG[
            "min_grounding_score"
        ]
    )


    reason = (
        "Relevant enterprise context found."
        if sufficient
        else
        "Retrieved context is too weak."
    )


    return {

        "sufficient": sufficient,

        "score": quality_score,

        "reason": reason
    }


# ============================================================
# 10. SOURCE EXTRACTION
# ============================================================
#
# Sources are returned explicitly.
#
# This is important for enterprise applications because users
# often need to verify the answer.
# ============================================================

def extract_sources(
    retrieval_results
):

    if (
        retrieval_results is None
        or retrieval_results.empty
    ):

        return []


    sources = []


    for _, row in retrieval_results.iterrows():

        sources.append({

            "source_id":
                row["chunk_id"],

            "document_id":
                row["doc_id"],

            "title":
                row["title"],

            "department":
                row["department"],

            "document_type":
                row["document_type"],

            "relevance_score":
                round(
                    float(
                        row["final_score"]
                    ),
                    4
                )
        })


    return sources


# ============================================================
# 11. DETERMINISTIC GROUNDED GENERATOR
# ============================================================
#
# This is our local CPU-friendly generation engine.
#
# Production replacement:
#
#     generate_answer()
#
# can later call:
#
#     Azure OpenAI
#     OpenAI
#     Claude
#     Gemini
#     local LLM
#
# without changing the retrieval interface.
# ============================================================

def generate_grounded_answer(
    query,
    retrieval_results
):

    generation_start = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # STEP 1 — Check context
    # --------------------------------------------------------

    context_quality = (
        assess_context_quality(
            retrieval_results
        )
    )


    if not context_quality[
        "sufficient"
    ]:

        answer = (
            "I could not find sufficiently "
            "relevant information in the "
            "available enterprise documents "
            "to answer this question reliably."
        )


        generation_latency = (
            time.perf_counter()
            -
            generation_start
        ) * 1000


        return {

            "answer": answer,

            "grounded": False,

            "confidence":
                round(
                    context_quality["score"],
                    4
                ),

            "sources":
                extract_sources(
                    retrieval_results
                ),

            "generation_latency_ms":
                generation_latency
        }


    # --------------------------------------------------------
    # STEP 2 — Collect candidate sentences
    # --------------------------------------------------------

    candidate_sentences = []


    for _, row in retrieval_results.head(
        RAG_CONFIG[
            "max_context_chunks"
        ]
    ).iterrows():

        sentences = split_sentences(
            row["text"]
        )


        for sentence in sentences:

            score = (
                sentence_relevance_score(
                    query,
                    sentence
                )
            )


            candidate_sentences.append({

                "sentence":
                    sentence,

                "score":
                    score,

                "source_id":
                    row["chunk_id"],

                "doc_id":
                    row["doc_id"],

                "title":
                    row["title"]
            })


    # --------------------------------------------------------
    # STEP 3 — Sort sentences
    # --------------------------------------------------------

    candidate_sentences = sorted(
        candidate_sentences,
        key=lambda x: x["score"],
        reverse=True
    )


    # --------------------------------------------------------
    # STEP 4 — Select strongest sentences
    # --------------------------------------------------------

    selected = []

    seen = set()


    for item in candidate_sentences:

        sentence = item[
            "sentence"
        ]


        normalized_sentence = (
            normalize_query(
                sentence
            )
        )


        if (
            normalized_sentence
            in seen
        ):

            continue


        seen.add(
            normalized_sentence
        )


        selected.append(
            item
        )


        if len(selected) >= (
            RAG_CONFIG[
                "max_answer_sentences"
            ]
        ):

            break


    # --------------------------------------------------------
    # STEP 5 — Build cited answer
    # --------------------------------------------------------

    answer_parts = []


    for item in selected:

        answer_parts.append(

            f"{item['sentence']} "
            f"[Source: {item['source_id']}]"
        )


    if not answer_parts:

        answer = (
            "The available enterprise "
            "documents do not contain enough "
            "information to answer this question."
        )

        grounded = False

    else:

        answer = " ".join(
            answer_parts
        )

        grounded = True


    # --------------------------------------------------------
    # STEP 6 — Generation latency
    # --------------------------------------------------------

    generation_latency = (
        time.perf_counter()
        -
        generation_start
    ) * 1000


    return {

        "answer": answer,

        "grounded": grounded,

        "confidence":
            round(
                context_quality["score"],
                4
            ),

        "sources":
            extract_sources(
                retrieval_results
            ),

        "generation_latency_ms":
            generation_latency
    }


# ============================================================
# 12. COMPLETE RAG PIPELINE
# ============================================================
#
# This is the main function of Part 3.
#
# Query
#   ↓
# Retrieval
#   ↓
# Context quality
#   ↓
# Generation
#   ↓
# Citations
#   ↓
# Grounding
#   ↓
# Structured response
# ============================================================

def enterprise_rag(
    query,
    department=None,
    document_type=None
):

    total_start = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # STEP 1 — RETRIEVAL
    # --------------------------------------------------------

    retrieval_results, retrieval_latency = (
        advanced_retrieve(

            query=query,

            top_k=RAG_CONFIG[
                "top_k"
            ],

            candidate_k=RAG_CONFIG[
                "candidate_k"
            ],

            similarity_threshold=(
                RAG_CONFIG[
                    "similarity_threshold"
                ]
            ),

            department=department,

            document_type=document_type
        )
    )


    # --------------------------------------------------------
    # STEP 2 — GENERATION
    # --------------------------------------------------------

    generation_result = (
        generate_grounded_answer(

            query=query,

            retrieval_results=
                retrieval_results
        )
    )


    # --------------------------------------------------------
    # STEP 3 — TOTAL LATENCY
    # --------------------------------------------------------

    total_latency = (
        time.perf_counter()
        -
        total_start
    ) * 1000


    # --------------------------------------------------------
    # STEP 4 — STRUCTURED RESPONSE
    # --------------------------------------------------------

    response = create_response(

        query=query,

        answer=
            generation_result[
                "answer"
            ],

        sources=
            generation_result[
                "sources"
            ],

        grounded=
            generation_result[
                "grounded"
            ],

        confidence=
            generation_result[
                "confidence"
            ],

        retrieval_latency_ms=
            retrieval_latency,

        generation_latency_ms=
            generation_result[
                "generation_latency_ms"
            ],

        total_latency_ms=
            total_latency
    )


    # Keep raw retrieval results internally
    response[
        "_retrieval_results"
    ] = retrieval_results


    return response


# ============================================================
# 13. TEST RAG — HR QUERY
# ============================================================

response_hr = enterprise_rag(
    "How do employees request annual leave?"
)


print("\n" + "=" * 80)
print("ENTERPRISE RAG — HR QUERY")
print("=" * 80)

print("\nQUESTION:")
print(response_hr["query"])

print("\nANSWER:")
print(response_hr["answer"])

print("\nGROUNDED:")
print(response_hr["grounded"])

print("\nCONFIDENCE:")
print(response_hr["confidence"])

print("\nLATENCY:")
print(
    response_hr["total_latency_ms"],
    "ms"
)

print("\nSOURCES:")

for source in response_hr["sources"]:

    print(
        f"- {source['source_id']} | "
        f"{source['title']} | "
        f"score={source['relevance_score']}"
    )


# ============================================================
# 14. TEST RAG — SECURITY QUERY
# ============================================================

response_security = enterprise_rag(
    "What should an employee do if "
    "their password is compromised?"
)


print("\n" + "=" * 80)
print("ENTERPRISE RAG — SECURITY QUERY")
print("=" * 80)

print("\nQUESTION:")
print(response_security["query"])

print("\nANSWER:")
print(response_security["answer"])

print("\nSOURCES:")

for source in response_security["sources"]:

    print(
        f"- {source['source_id']} | "
        f"{source['title']}"
    )


# ============================================================
# 15. TEST RAG — FINANCE QUERY
# ============================================================

response_finance = enterprise_rag(
    "How can I claim business travel expenses?"
)


print("\n" + "=" * 80)
print("ENTERPRISE RAG — FINANCE QUERY")
print("=" * 80)

print("\nQUESTION:")
print(response_finance["query"])

print("\nANSWER:")
print(response_finance["answer"])

print("\nSOURCES:")

for source in response_finance["sources"]:

    print(
        f"- {source['source_id']} | "
        f"{source['title']}"
    )


# ============================================================
# 16. TEST METADATA-AWARE RAG
# ============================================================
#
# We can constrain retrieval to Finance.
# ============================================================

response_finance_filtered = enterprise_rag(

    query=(
        "How are employee expenses "
        "reimbursed?"
    ),

    department="Finance"
)


print("\n" + "=" * 80)
print("FINANCE-FILTERED RAG")
print("=" * 80)

print("\nANSWER:")
print(
    response_finance_filtered[
        "answer"
    ]
)

print("\nSOURCES:")

for source in (
    response_finance_filtered[
        "sources"
    ]
):

    print(
        f"- {source['source_id']} | "
        f"{source['department']}"
    )


# ============================================================
# 17. TEST OUT-OF-DOMAIN QUESTION
# ============================================================
#
# This test is extremely important.
#
# We intentionally ask something that does not exist in our
# enterprise knowledge base.
#
# Expected behavior:
#
# DO NOT HALLUCINATE.
#
# Instead:
#
# "I could not find sufficiently relevant information..."
# ============================================================

unknown_query = (
    "What is the weather in Pune today?"
)


unknown_response = enterprise_rag(
    unknown_query
)


print("\n" + "=" * 80)
print("OUT-OF-DOMAIN / UNKNOWN QUERY")
print("=" * 80)

print("\nQUESTION:")
print(
    unknown_response["query"]
)

print("\nANSWER:")
print(
    unknown_response["answer"]
)

print("\nGROUNDED:")
print(
    unknown_response["grounded"]
)

print("\nCONFIDENCE:")
print(
    unknown_response["confidence"]
)


# ============================================================
# 18. BUILD A HUMAN-READABLE RESPONSE
# ============================================================
#
# This represents what could eventually be displayed in a
# React/Angular/Streamlit enterprise UI.
# ============================================================

def format_enterprise_response(
    response
):

    output = []

    output.append(
        "ANSWER\n"
    )

    output.append(
        response["answer"]
    )


    output.append(
        "\n\nSOURCES\n"
    )


    if response["sources"]:

        for source in response[
            "sources"
        ]:

            output.append(

                f"- "
                f"{source['source_id']} | "
                f"{source['title']} | "
                f"{source['department']}"
            )

    else:

        output.append(
            "No sources available."
        )


    output.append(
        "\n\nMETADATA\n"
    )


    output.append(
        f"Grounded: "
        f"{response['grounded']}"
    )

    output.append(
        f"Confidence: "
        f"{response['confidence']}"
    )

    output.append(
        f"Latency: "
        f"{response['total_latency_ms']} ms"
    )


    return "\n".join(
        output
    )


print("\n" + "=" * 80)
print("USER-FACING ENTERPRISE RESPONSE")
print("=" * 80)

print(
    format_enterprise_response(
        response_security
    )
)


# ============================================================
# 19. GROUNDING CHECK
# ============================================================
#
# A key RAG evaluation question:
#
# "Does the answer actually use retrieved context?"
#
# We perform a lightweight lexical grounding check.
#
# This is not a replacement for an LLM-as-a-judge or
# semantic entailment model, but it gives us a CPU-friendly
# baseline.
# ============================================================

def grounding_check(
    answer,
    retrieval_results
):

    if (
        retrieval_results is None
        or retrieval_results.empty
    ):

        return {

            "grounded": False,

            "score": 0.0
        }


    answer_tokens = tokenize(
        answer
    )


    context_text = " ".join(
        retrieval_results["text"]
        .tolist()
    )


    context_tokens = tokenize(
        context_text
    )


    if not answer_tokens:

        return {

            "grounded": False,

            "score": 0.0
        }


    overlap = (
        answer_tokens.intersection(
            context_tokens
        )
    )


    score = (
        len(overlap) /
        len(answer_tokens)
    )


    return {

        "grounded":
            score >= 0.50,

        "score":
            score
    }


# ============================================================
# 20. RUN GROUNDING CHECK
# ============================================================

grounding_result = grounding_check(

    response_security[
        "answer"
    ],

    response_security[
        "_retrieval_results"
    ]
)


print("\n" + "=" * 80)
print("GROUNDING CHECK")
print("=" * 80)

print(
    "Grounded:",
    grounding_result["grounded"]
)

print(
    "Grounding score:",
    round(
        grounding_result["score"],
        4
    )
)


# ============================================================
# 21. BUILD RAG EVALUATION DATASET
# ============================================================
#
# We evaluate the complete RAG pipeline.
#
# Each query has:
#
# - expected source document
# - expected answer concepts
#
# ============================================================

rag_evaluation_dataset = [

    {
        "query":
            "How do employees request annual leave?",

        "expected_doc":
            "HR-001",

        "expected_keywords":
            [
                "leave",
                "requests",
                "portal",
                "manager"
            ]
    },

    {
        "query":
            "Can employees work remotely?",

        "expected_doc":
            "HR-002",

        "expected_keywords":
            [
                "remote",
                "manager",
                "working"
            ]
    },

    {
        "query":
            "What should I do if my password is compromised?",

        "expected_doc":
            "IT-001",

        "expected_keywords":
            [
                "password",
                "security",
                "report"
            ]
    },

    {
        "query":
            "How are production incidents reported?",

        "expected_doc":
            "IT-002",

        "expected_keywords":
            [
                "incident",
                "reported",
                "system"
            ]
    },

    {
        "query":
            "How do I submit business expenses?",

        "expected_doc":
            "FIN-001",

        "expected_keywords":
            [
                "expense",
                "receipts",
                "system"
            ]
    },

    {
        "query":
            "How should confidential information be protected?",

        "expected_doc":
            "SEC-001",

        "expected_keywords":
            [
                "confidential",
                "access",
                "controls"
            ]
    },

    {
        "query":
            "What are the API development standards?",

        "expected_doc":
            "DEV-001",

        "expected_keywords":
            [
                "API",
                "authentication",
                "authorization"
            ]
    },

    {
        "query":
            "How should production deployments be handled?",

        "expected_doc":
            "DEV-002",

        "expected_keywords":
            [
                "deployment",
                "tests",
                "pipeline"
            ]
    },

    {
        "query":
            "How should a RAG application be monitored?",

        "expected_doc":
            "AI-002",

        "expected_keywords":
            [
                "retrieval",
                "quality",
                "latency"
            ]
    },

    {
        "query":
            "How should customer data be protected?",

        "expected_doc":
            "BFSI-001",

        "expected_keywords":
            [
                "customer",
                "data",
                "access"
            ]
    }
]


# ============================================================
# 22. RAG EVALUATION FUNCTION
# ============================================================

def evaluate_rag(
    evaluation_data
):

    rows = []


    for item in evaluation_data:

        start_time = (
            time.perf_counter()
        )


        response = enterprise_rag(
            item["query"]
        )


        total_latency = (
            time.perf_counter()
            -
            start_time
        ) * 1000


        # ----------------------------------------
        # Source correctness
        # ----------------------------------------

        retrieved_doc_ids = [

            source["document_id"]

            for source in
            response["sources"]
        ]


        source_hit = (
            item["expected_doc"]
            in retrieved_doc_ids
        )


        # ----------------------------------------
        # Answer keyword coverage
        # ----------------------------------------

        answer_tokens = tokenize(
            response["answer"]
        )


        expected_keywords = set(
            [
                keyword.lower()
                for keyword in
                item["expected_keywords"]
            ]
        )


        matched_keywords = (
            expected_keywords
            .intersection(
                answer_tokens
            )
        )


        keyword_coverage = (

            len(matched_keywords)

            /

            len(expected_keywords)
        )


        # ----------------------------------------
        # Grounding
        # ----------------------------------------

        grounding = grounding_check(

            response["answer"],

            response[
                "_retrieval_results"
            ]
        )


        rows.append({

            "query":
                item["query"],

            "expected_doc":
                item["expected_doc"],

            "source_hit":
                source_hit,

            "grounded":
                response["grounded"],

            "confidence":
                response["confidence"],

            "keyword_coverage":
                keyword_coverage,

            "grounding_score":
                grounding["score"],

            "latency_ms":
                total_latency
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# 23. RUN RAG EVALUATION
# ============================================================

rag_evaluation_df = evaluate_rag(
    rag_evaluation_dataset
)


# ============================================================
# 24. CALCULATE RAG METRICS
# ============================================================

rag_source_accuracy = (
    rag_evaluation_df[
        "source_hit"
    ].mean()
)


rag_grounded_rate = (
    rag_evaluation_df[
        "grounded"
    ].mean()
)


rag_keyword_coverage = (
    rag_evaluation_df[
        "keyword_coverage"
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


rag_median_latency = (
    rag_evaluation_df[
        "latency_ms"
    ].median()
)


# ============================================================
# 25. DISPLAY RAG EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("COMPLETE RAG EVALUATION")
print("=" * 80)

print(
    f"\nQueries evaluated       : "
    f"{len(rag_evaluation_df)}"
)

print(
    f"Source Accuracy         : "
    f"{rag_source_accuracy:.2%}"
)

print(
    f"Grounded Response Rate  : "
    f"{rag_grounded_rate:.2%}"
)

print(
    f"Keyword Coverage        : "
    f"{rag_keyword_coverage:.2%}"
)

print(
    f"Average Grounding Score : "
    f"{rag_average_grounding:.4f}"
)

print(
    f"Average Latency         : "
    f"{rag_average_latency:.2f} ms"
)

print(
    f"Median Latency          : "
    f"{rag_median_latency:.2f} ms"
)


# ============================================================
# 26. DISPLAY EVALUATION DETAILS
# ============================================================

display(
    rag_evaluation_df
)


# ============================================================
# 27. FIND WEAK RAG RESPONSES
# ============================================================
#
# Production systems should automatically identify weak cases.
# ============================================================

weak_rag_cases = (
    rag_evaluation_df[
        (
            rag_evaluation_df[
                "source_hit"
            ] == False
        )
        |
        (
            rag_evaluation_df[
                "grounding_score"
            ] < 0.50
        )
    ]
)


print("\n" + "=" * 80)
print("WEAK RAG CASES")
print("=" * 80)


if weak_rag_cases.empty:

    print(
        "No weak cases detected "
        "in this evaluation set."
    )

else:

    display(
        weak_rag_cases
    )


# ============================================================
# 28. COMPLETE PIPELINE TEST
# ============================================================
#
# This demonstrates the entire Day 58 system so far.
# ============================================================

demo_queries = [

    "How do I request leave?",

    "What should I do about a compromised password?",

    "How do I claim a business expense?",

    "What is required for a production deployment?",

    "How should a RAG application be monitored?"
]


demo_results = []


for query in demo_queries:

    response = enterprise_rag(
        query
    )


    demo_results.append({

        "query":
            query,

        "answer":
            response["answer"],

        "grounded":
            response["grounded"],

        "confidence":
            response["confidence"],

        "latency_ms":
            response["total_latency_ms"],

        "sources":
            ", ".join(
                [
                    source[
                        "document_id"
                    ]

                    for source in
                    response["sources"]
                ]
            )
    })


demo_df = pd.DataFrame(
    demo_results
)


print("\n" + "=" * 80)
print("ENTERPRISE RAG DEMO")
print("=" * 80)

display(
    demo_df
)


# ============================================================
# 29. FINAL SYSTEM HEALTH CHECK
# ============================================================

print("\n" + "=" * 80)
print("PART 3 SYSTEM HEALTH CHECK")
print("=" * 80)


system_checks = {

    "Document corpus available":
        len(documents_df) > 0,

    "Chunk index available":
        len(chunks_df) > 0,

    "Vector index available":
        chunk_vectors.shape[0] > 0,

    "Advanced retriever available":
        callable(
            advanced_retrieve
        ),

    "Enterprise retriever available":
        callable(
            enterprise_retriever.retrieve
        ),

    "RAG pipeline available":
        callable(
            enterprise_rag
        ),

    "Source citation available":
        callable(
            extract_sources
        ),

    "Grounding check available":
        callable(
            grounding_check
        ),

    "RAG evaluation completed":
        len(
            rag_evaluation_df
        ) > 0
}


for check, status in system_checks.items():

    print(
        f"{'PASS' if status else 'FAIL'} : "
        f"{check}"
    )


# ============================================================
# 30. IMPORTANT VARIABLES CREATED
# ============================================================

print("\n" + "=" * 80)
print("IMPORTANT VARIABLES CREATED IN PART 3")
print("=" * 80)

print("""
RAG_CONFIG
    → Central RAG configuration

SYSTEM_PROMPT
    → Grounding instructions for future LLM generation

build_grounded_context()
    → Converts retrieved chunks into structured context

build_rag_prompt()
    → Creates an LLM-ready grounded prompt

assess_context_quality()
    → Determines whether retrieved context is sufficient

extract_sources()
    → Creates source/citation metadata

generate_grounded_answer()
    → CPU-friendly deterministic generation layer

enterprise_rag()
    → Complete end-to-end RAG pipeline

grounding_check()
    → Lightweight answer/context grounding check

rag_evaluation_dataset
    → Ground-truth RAG evaluation data

rag_evaluation_df
    → RAG evaluation results

rag_source_accuracy
    → Correct-source retrieval rate

rag_grounded_rate
    → Percentage of grounded responses

rag_keyword_coverage
    → Answer keyword coverage

rag_average_grounding
    → Average grounding score

rag_average_latency
    → End-to-end RAG latency

demo_df
    → End-to-end demonstration results
""")


# ============================================================
# 31. FINAL ARCHITECTURE
# ============================================================

print("\n" + "=" * 80)
print("PART 3 COMPLETE")
print("=" * 80)

print("""
ENTERPRISE DOCUMENT INTELLIGENCE ASSISTANT

                         USER
                          │
                          ▼
                       QUERY
                          │
                          ▼
                 Query Normalization
                          │
                          ▼
                   Query Expansion
                          │
                          ▼
                Advanced Retrieval
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
       Vector Search            Metadata Filter
             │                         │
             └────────────┬────────────┘
                          │
                          ▼
                       Rerank
                          │
                          ▼
                        Top-K
                          │
                          ▼
                 Context Quality Check
                    /            \\
                   /              \\
              Sufficient        Weak
                 │                │
                 ▼                ▼
          Grounded Answer      Safe
                 │             Fallback
                 ▼
            Source Citations
                 │
                 ▼
         Grounding Validation
                 │
                 ▼
        Structured RAG Response
                 │
                 ▼
                USER
""")


# ============================================================
# 32. PRODUCTION LLM REPLACEMENT POINT
# ============================================================
#
# IMPORTANT ARCHITECTURAL INSIGHT
#
# The current system intentionally does NOT depend on an LLM.
#
# In production:
#
# generate_grounded_answer()
#
# can be replaced by:
#
#     Azure OpenAI
#     OpenAI
#     Claude
#     Gemini
#     Llama
#     Qwen
#     enterprise-hosted model
#
# The interface remains:
#
#     query + retrieved_context
#                 ↓
#             LLM
#                 ↓
#             answer
#
# Everything before the generation layer remains reusable.
# ============================================================


print("""
PRODUCTION GENERATION LAYER

Current:

Advanced Retrieval
       ↓
Context
       ↓
Deterministic Generator


Production:

Advanced Retrieval
       ↓
Context
       ↓
Grounded Prompt
       ↓
Enterprise LLM
       ↓
Answer
       ↓
Citation / Grounding Validation
       ↓
Final Response
""")


# ============================================================
# 33. INTERVIEW-READY EXPLANATION
# ============================================================

print("""
INTERVIEW ANSWER:

"I designed the RAG system as a modular pipeline where retrieval
and generation are decoupled.

The user query first goes through normalization and expansion.
The retrieval layer performs candidate retrieval, metadata
filtering and lightweight reranking.

The top relevant chunks are then converted into structured
context containing source IDs and document metadata.

For generation, I use a grounded prompt that explicitly instructs
the model to answer only from retrieved enterprise context and
to avoid unsupported claims.

I also added a context sufficiency check. If retrieval quality
is below a configured threshold, the system does not blindly
generate an answer. Instead, it returns a safe fallback indicating
that the available enterprise knowledge is insufficient.

Every response contains source information so the user can trace
the answer back to the original document.

For evaluation, I measure source accuracy, grounding rate,
keyword coverage, grounding score and end-to-end latency.

In production, the deterministic generator can be replaced with
an enterprise LLM such as Azure OpenAI without changing the
retrieval architecture."
""")


# ============================================================
# END OF PART 3
# ============================================================
# ============================================================
# DAY 58 — ENTERPRISE DOCUMENT INTELLIGENCE ASSISTANT
# PART 4 — PRODUCTION FASTAPI + MONITORING + CLOUD ARCHITECTURE
# ============================================================
#
# CONTINUES DIRECTLY FROM PART 1 + PART 2 + PART 3
#
# Existing important variables/functions:
#
# documents_df
# chunks_df
# vectorizer
# chunk_vectors
# vector_store
# advanced_retrieve()
# enterprise_retriever
# enterprise_rag()
# grounding_check()
# RAG_CONFIG
#
# We will NOT recreate the RAG pipeline.
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import pandas as pd


# ============================================================
# 2. FASTAPI IMPORTS
# ============================================================

from fastapi import (
    FastAPI,
    HTTPException,
    Request
)

from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field


# ============================================================
# 3. CONFIGURATION
# ============================================================
#
# In production these values should come from environment
# variables / secret management.
#
# We keep the current notebook CPU-friendly and local.
# ============================================================

APP_CONFIG = {

    "APP_NAME":
        "Enterprise Document Intelligence Assistant",

    "VERSION":
        "1.0.0",

    "ENVIRONMENT":
        os.getenv(
            "APP_ENV",
            "development"
        ),

    "HOST":
        os.getenv(
            "APP_HOST",
            "0.0.0.0"
        ),

    "PORT":
        int(
            os.getenv(
                "APP_PORT",
                "8000"
            )
        ),

    "TOP_K":
        int(
            os.getenv(
                "RAG_TOP_K",
                "3"
            )
        ),

    "CANDIDATE_K":
        int(
            os.getenv(
                "RAG_CANDIDATE_K",
                "10"
            )
        ),

    "SIMILARITY_THRESHOLD":
        float(
            os.getenv(
                "RAG_SIMILARITY_THRESHOLD",
                "0.05"
            )
        ),

    "MIN_GROUNDING_SCORE":
        float(
            os.getenv(
                "RAG_MIN_GROUNDING_SCORE",
                "0.35"
            )
        )
}


print("=" * 80)
print("APPLICATION CONFIGURATION")
print("=" * 80)

for key, value in APP_CONFIG.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# 4. LOGGING
# ============================================================
#
# Production AI systems need observability.
#
# We log:
#
# - timestamp
# - log level
# - service
# - message
#
# We deliberately do NOT log sensitive user content.
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "enterprise-rag | "
        "%(message)s"
    )
)

logger = logging.getLogger(
    "enterprise-rag"
)


logger.info(
    "Enterprise RAG application initialized."
)


# ============================================================
# 5. APPLICATION METRICS
# ============================================================
#
# Lightweight in-memory metrics.
#
# In production these metrics can be exported to:
#
# Prometheus
# Grafana
# Azure Monitor
# Application Insights
# CloudWatch
# Datadog
#
# ============================================================

class ApplicationMetrics:

    def __init__(self):

        self.total_requests = 0

        self.successful_requests = 0

        self.failed_requests = 0

        self.rag_queries = 0

        self.search_queries = 0

        self.grounded_responses = 0

        self.ungrounded_responses = 0

        self.total_latency_ms = 0.0

        self.total_retrieval_latency_ms = 0.0

        self.total_generation_latency_ms = 0.0


    def record_request(
        self,
        latency_ms
    ):

        self.total_requests += 1

        self.total_latency_ms += (
            latency_ms
        )


    def record_success(self):

        self.successful_requests += 1


    def record_failure(self):

        self.failed_requests += 1


    def record_rag(
        self,
        grounded,
        retrieval_latency_ms,
        generation_latency_ms
    ):

        self.rag_queries += 1

        self.total_retrieval_latency_ms += (
            retrieval_latency_ms
        )

        self.total_generation_latency_ms += (
            generation_latency_ms
        )


        if grounded:

            self.grounded_responses += 1

        else:

            self.ungrounded_responses += 1


    def record_search(self):

        self.search_queries += 1


    def snapshot(self):

        average_latency = (

            self.total_latency_ms
            /
            self.total_requests

            if self.total_requests > 0

            else 0
        )


        average_retrieval_latency = (

            self.total_retrieval_latency_ms
            /
            self.rag_queries

            if self.rag_queries > 0

            else 0
        )


        average_generation_latency = (

            self.total_generation_latency_ms
            /
            self.rag_queries

            if self.rag_queries > 0

            else 0
        )


        grounding_rate = (

            self.grounded_responses
            /
            self.rag_queries

            if self.rag_queries > 0

            else 0
        )


        success_rate = (

            self.successful_requests
            /
            self.total_requests

            if self.total_requests > 0

            else 0
        )


        return {

            "total_requests":
                self.total_requests,

            "successful_requests":
                self.successful_requests,

            "failed_requests":
                self.failed_requests,

            "success_rate":
                round(
                    success_rate,
                    4
                ),

            "rag_queries":
                self.rag_queries,

            "search_queries":
                self.search_queries,

            "grounded_responses":
                self.grounded_responses,

            "ungrounded_responses":
                self.ungrounded_responses,

            "grounding_rate":
                round(
                    grounding_rate,
                    4
                ),

            "average_latency_ms":
                round(
                    average_latency,
                    2
                ),

            "average_retrieval_latency_ms":
                round(
                    average_retrieval_latency,
                    2
                ),

            "average_generation_latency_ms":
                round(
                    average_generation_latency,
                    2
                )
        }


metrics = ApplicationMetrics()


# ============================================================
# 6. PYDANTIC REQUEST MODELS
# ============================================================
#
# API contract:
#
# POST /query
#
# {
#     "query": "...",
#     "department": "Finance",
#     "document_type": "Policy"
# }
#
# ============================================================

class QueryRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Enterprise question"
    )

    department: Optional[str] = Field(
        default=None,
        description="Optional department filter"
    )

    document_type: Optional[str] = Field(
        default=None,
        description="Optional document type filter"
    )


class SearchRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=2,
        max_length=1000
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10
    )

    department: Optional[str] = None

    document_type: Optional[str] = None


# ============================================================
# 7. RESPONSE MODELS
# ============================================================

class SourceResponse(BaseModel):

    source_id: str

    document_id: str

    title: str

    department: str

    document_type: str

    relevance_score: float


class QueryResponse(BaseModel):

    request_id: str

    query: str

    answer: str

    grounded: bool

    confidence: float

    sources: List[SourceResponse]

    retrieval_latency_ms: float

    generation_latency_ms: float

    total_latency_ms: float

    status: str


class SearchResultResponse(BaseModel):

    chunk_id: str

    document_id: str

    title: str

    department: str

    document_type: str

    score: float

    text: str


class SearchResponse(BaseModel):

    request_id: str

    query: str

    results: List[SearchResultResponse]

    latency_ms: float

    status: str


# ============================================================
# 8. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=APP_CONFIG[
        "APP_NAME"
    ],

    description=(
        "Production-style enterprise "
        "document intelligence and "
        "grounded RAG API."
    ),

    version=APP_CONFIG[
        "VERSION"
    ]
)


# ============================================================
# 9. REQUEST MIDDLEWARE
# ============================================================
#
# Generates request IDs and measures total API latency.
# ============================================================

@app.middleware("http")
async def request_middleware(
    request: Request,
    call_next
):

    request_id = str(
        uuid.uuid4()
    )


    request.state.request_id = (
        request_id
    )


    start_time = (
        time.perf_counter()
    )


    metrics.total_requests += 1


    try:

        response = await call_next(
            request
        )


        metrics.successful_requests += 1

        response.headers[
            "X-Request-ID"
        ] = request_id


        return response


    except Exception as exc:

        metrics.failed_requests += 1


        logger.exception(
            "Request failed | "
            f"request_id={request_id} | "
            f"error={type(exc).__name__}"
        )


        raise


    finally:

        latency_ms = (
            time.perf_counter()
            -
            start_time
        ) * 1000


        metrics.total_latency_ms += (
            latency_ms
        )


        logger.info(
            "Request completed | "
            f"request_id={request_id} | "
            f"method={request.method} | "
            f"path={request.url.path} | "
            f"latency_ms={latency_ms:.2f}"
        )


# ============================================================
# 10. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "service":
            APP_CONFIG[
                "APP_NAME"
            ],

        "version":
            APP_CONFIG[
                "VERSION"
            ],

        "environment":
            APP_CONFIG[
                "ENVIRONMENT"
            ],

        "status":
            "running"
    }


# ============================================================
# 11. HEALTH ENDPOINT
# ============================================================
#
# Health checks are used by:
#
# - Docker
# - Kubernetes
# - Azure Container Apps
# - load balancers
# - monitoring systems
#
# ============================================================

@app.get("/health")
def health():

    vector_index_ready = (
        chunk_vectors is not None
        and
        chunk_vectors.shape[0] > 0
    )


    documents_ready = (
        len(documents_df) > 0
    )


    chunks_ready = (
        len(chunks_df) > 0
    )


    retriever_ready = callable(
        advanced_retrieve
    )


    rag_ready = callable(
        enterprise_rag
    )


    overall_ready = all([

        vector_index_ready,

        documents_ready,

        chunks_ready,

        retriever_ready,

        rag_ready
    ])


    return {

        "status":
            "healthy"
            if overall_ready
            else
            "degraded",

        "service":
            APP_CONFIG[
                "APP_NAME"
            ],

        "checks": {

            "documents":
                documents_ready,

            "chunks":
                chunks_ready,

            "vector_index":
                vector_index_ready,

            "retriever":
                retriever_ready,

            "rag_pipeline":
                rag_ready
        },

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat()
    }


# ============================================================
# 12. METRICS ENDPOINT
# ============================================================

@app.get("/metrics")
def get_metrics():

    return {

        "service":
            APP_CONFIG[
                "APP_NAME"
            ],

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "metrics":
            metrics.snapshot(),

        "knowledge_base": {

            "documents":
                len(documents_df),

            "chunks":
                len(chunks_df),

            "vector_dimensions":
                chunk_vectors.shape[1]
        }
    }


# ============================================================
# 13. SEARCH ENDPOINT
# ============================================================
#
# This exposes the retrieval layer separately.
#
# Useful for:
#
# - debugging
# - retrieval evaluation
# - admin dashboards
# - search interfaces
# - RAG troubleshooting
#
# ============================================================

@app.post(
    "/search",
    response_model=SearchResponse
)
def search_endpoint(
    request: SearchRequest,
    http_request: Request
):

    request_id = (
        http_request.state.request_id
    )


    start_time = (
        time.perf_counter()
    )


    try:

        results, _ = advanced_retrieve(

            query=request.query,

            top_k=request.top_k,

            candidate_k=max(
                request.top_k * 3,
                10
            ),

            similarity_threshold=(
                APP_CONFIG[
                    "SIMILARITY_THRESHOLD"
                ]
            ),

            department=request.department,

            document_type=request.document_type
        )


        latency_ms = (
            time.perf_counter()
            -
            start_time
        ) * 1000


        metrics.record_search()


        response_results = []


        for _, row in results.iterrows():

            response_results.append(

                SearchResultResponse(

                    chunk_id=str(
                        row["chunk_id"]
                    ),

                    document_id=str(
                        row["doc_id"]
                    ),

                    title=str(
                        row["title"]
                    ),

                    department=str(
                        row["department"]
                    ),

                    document_type=str(
                        row["document_type"]
                    ),

                    score=round(
                        float(
                            row[
                                "final_score"
                            ]
                        ),
                        4
                    ),

                    text=str(
                        row["text"]
                    )
                )
            )


        return SearchResponse(

            request_id=request_id,

            query=request.query,

            results=response_results,

            latency_ms=round(
                latency_ms,
                2
            ),

            status="success"
        )


    except Exception as exc:

        logger.exception(
            "Search error | "
            f"request_id={request_id} | "
            f"error={type(exc).__name__}"
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "Search operation failed."
            )
        )


# ============================================================
# 14. MAIN RAG QUERY ENDPOINT
# ============================================================
#
# POST /query
#
# This is the primary endpoint for the AI assistant.
# ============================================================

@app.post(
    "/query",
    response_model=QueryResponse
)
def query_endpoint(
    request: QueryRequest,
    http_request: Request
):

    request_id = (
        http_request.state.request_id
    )


    logger.info(
        "RAG query received | "
        f"request_id={request_id}"
    )


    try:

        # --------------------------------------------
        # RAG PIPELINE
        # --------------------------------------------

        response = enterprise_rag(

            query=request.query,

            department=request.department,

            document_type=request.document_type
        )


        # --------------------------------------------
        # METRICS
        # --------------------------------------------

        metrics.record_rag(

            grounded=
                response[
                    "grounded"
                ],

            retrieval_latency_ms=
                response[
                    "retrieval_latency_ms"
                ],

            generation_latency_ms=
                response[
                    "generation_latency_ms"
                ]
        )


        # --------------------------------------------
        # SOURCES
        # --------------------------------------------

        sources = [

            SourceResponse(

                source_id=
                    source[
                        "source_id"
                    ],

                document_id=
                    source[
                        "document_id"
                    ],

                title=
                    source[
                        "title"
                    ],

                department=
                    source[
                        "department"
                    ],

                document_type=
                    source[
                        "document_type"
                    ],

                relevance_score=
                    source[
                        "relevance_score"
                    ]
            )

            for source in
            response["sources"]
        ]


        # --------------------------------------------
        # RESPONSE
        # --------------------------------------------

        return QueryResponse(

            request_id=request_id,

            query=request.query,

            answer=
                response[
                    "answer"
                ],

            grounded=
                response[
                    "grounded"
                ],

            confidence=
                response[
                    "confidence"
                ],

            sources=sources,

            retrieval_latency_ms=
                response[
                    "retrieval_latency_ms"
                ],

            generation_latency_ms=
                response[
                    "generation_latency_ms"
                ],

            total_latency_ms=
                response[
                    "total_latency_ms"
                ],

            status="success"
        )


    except Exception as exc:

        logger.exception(
            "RAG query failed | "
            f"request_id={request_id} | "
            f"error={type(exc).__name__}"
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to process the "
                "enterprise query."
            )
        )


# ============================================================
# 15. GLOBAL EXCEPTION HANDLER
# ============================================================

@app.exception_handler(
    Exception
)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    request_id = getattr(
        request.state,
        "request_id",
        "unknown"
    )


    logger.exception(
        "Unhandled exception | "
        f"request_id={request_id} | "
        f"error={type(exc).__name__}"
    )


    return JSONResponse(

        status_code=500,

        content={

            "error":
                "Internal server error.",

            "request_id":
                request_id,

            "status":
                "failed"
        }
    )


# ============================================================
# 16. API INFORMATION ENDPOINT
# ============================================================

@app.get("/info")
def application_info():

    return {

        "application":
            APP_CONFIG[
                "APP_NAME"
            ],

        "version":
            APP_CONFIG[
                "VERSION"
            ],

        "environment":
            APP_CONFIG[
                "ENVIRONMENT"
            ],

        "architecture": {

            "retrieval":
                "TF-IDF + cosine similarity",

            "query_expansion":
                "Enterprise terminology mapping",

            "reranking":
                "Vector + keyword + phrase scoring",

            "generation":
                "Grounded deterministic generator",

            "api":
                "FastAPI",

            "monitoring":
                "Application metrics + logging"
        },

        "knowledge_base": {

            "documents":
                len(documents_df),

            "chunks":
                len(chunks_df),

            "vector_dimensions":
                chunk_vectors.shape[1]
        }
    }


# ============================================================
# 17. RUN API TESTS USING FASTAPI TESTCLIENT
# ============================================================

from fastapi.testclient import TestClient


client = TestClient(
    app
)


# ============================================================
# 18. TEST ROOT ENDPOINT
# ============================================================

root_response = client.get("/")


print("\n" + "=" * 80)
print("ROOT API TEST")
print("=" * 80)

print(
    "Status:",
    root_response.status_code
)

print(
    root_response.json()
)


# ============================================================
# 19. TEST HEALTH ENDPOINT
# ============================================================

health_response = client.get(
    "/health"
)


print("\n" + "=" * 80)
print("HEALTH API TEST")
print("=" * 80)

print(
    "Status:",
    health_response.status_code
)

print(
    health_response.json()
)


# ============================================================
# 20. TEST SEARCH ENDPOINT
# ============================================================

search_payload = {

    "query":
        "How do employees submit expenses?",

    "top_k":
        3
}


search_response = client.post(

    "/search",

    json=search_payload
)


print("\n" + "=" * 80)
print("SEARCH API TEST")
print("=" * 80)

print(
    "Status:",
    search_response.status_code
)

print(
    search_response.json()
)


# ============================================================
# 21. TEST RAG QUERY ENDPOINT
# ============================================================

query_payload = {

    "query":
        "What should I do if my password is compromised?"
}


query_response = client.post(

    "/query",

    json=query_payload
)


print("\n" + "=" * 80)
print("RAG QUERY API TEST")
print("=" * 80)

print(
    "Status:",
    query_response.status_code
)

query_json = (
    query_response.json()
)


print(
    "Request ID:",
    query_json["request_id"]
)

print(
    "\nAnswer:"
)

print(
    query_json["answer"]
)

print(
    "\nGrounded:",
    query_json["grounded"]
)

print(
    "Confidence:",
    query_json["confidence"]
)

print(
    "Latency:",
    query_json["total_latency_ms"],
    "ms"
)


# ============================================================
# 22. TEST FINANCE FILTER
# ============================================================

finance_payload = {

    "query":
        "How are business expenses reimbursed?",

    "department":
        "Finance"
}


finance_response = client.post(

    "/query",

    json=finance_payload
)


print("\n" + "=" * 80)
print("FILTERED RAG API TEST")
print("=" * 80)

print(
    "Status:",
    finance_response.status_code
)

print(
    finance_response.json()
)


# ============================================================
# 23. TEST UNKNOWN QUERY
# ============================================================

unknown_payload = {

    "query":
        "What is the weather today?"
}


unknown_api_response = client.post(

    "/query",

    json=unknown_payload
)


print("\n" + "=" * 80)
print("UNKNOWN QUERY API TEST")
print("=" * 80)

print(
    "Status:",
    unknown_api_response.status_code
)

print(
    unknown_api_response.json()
)


# ============================================================
# 24. TEST METRICS ENDPOINT
# ============================================================

metrics_response = client.get(
    "/metrics"
)


print("\n" + "=" * 80)
print("METRICS API TEST")
print("=" * 80)

print(
    metrics_response.json()
)


# ============================================================
# 25. TEST INFO ENDPOINT
# ============================================================

info_response = client.get(
    "/info"
)


print("\n" + "=" * 80)
print("INFO API TEST")
print("=" * 80)

print(
    info_response.json()
)


# ============================================================
# 26. AUTOMATED API TEST SUITE
# ============================================================
#
# Simple automated validation.
# ============================================================

api_test_results = []


# --------------------------------------------
# Root
# --------------------------------------------

api_test_results.append({

    "test":
        "GET /",

    "expected":
        200,

    "actual":
        root_response.status_code,

    "passed":
        root_response.status_code == 200
})


# --------------------------------------------
# Health
# --------------------------------------------

api_test_results.append({

    "test":
        "GET /health",

    "expected":
        200,

    "actual":
        health_response.status_code,

    "passed":
        health_response.status_code == 200
})


# --------------------------------------------
# Search
# --------------------------------------------

api_test_results.append({

    "test":
        "POST /search",

    "expected":
        200,

    "actual":
        search_response.status_code,

    "passed":
        search_response.status_code == 200
})


# --------------------------------------------
# Query
# --------------------------------------------

api_test_results.append({

    "test":
        "POST /query",

    "expected":
        200,

    "actual":
        query_response.status_code,

    "passed":
        query_response.status_code == 200
})


# --------------------------------------------
# Metrics
# --------------------------------------------

api_test_results.append({

    "test":
        "GET /metrics",

    "expected":
        200,

    "actual":
        metrics_response.status_code,

    "passed":
        metrics_response.status_code == 200
})


# --------------------------------------------
# Info
# --------------------------------------------

api_test_results.append({

    "test":
        "GET /info",

    "expected":
        200,

    "actual":
        info_response.status_code,

    "passed":
        info_response.status_code == 200
})


api_test_df = pd.DataFrame(
    api_test_results
)


print("\n" + "=" * 80)
print("AUTOMATED API TEST SUITE")
print("=" * 80)

display(
    api_test_df
)


print(
    "\nTests passed:",
    api_test_df["passed"].sum(),
    "/",
    len(api_test_df)
)


# ============================================================
# 27. PERFORMANCE TEST
# ============================================================
#
# Send multiple queries and measure API latency.
# ============================================================

performance_queries = [

    "How do employees request annual leave?",

    "What should I do if my password is compromised?",

    "How do I submit business expenses?",

    "How are production deployments handled?",

    "How should a RAG application be monitored?"
]


performance_results = []


for query in performance_queries:

    start = time.perf_counter()


    response = client.post(

        "/query",

        json={
            "query": query
        }
    )


    latency_ms = (
        time.perf_counter()
        -
        start
    ) * 1000


    performance_results.append({

        "query":
            query,

        "status_code":
            response.status_code,

        "latency_ms":
            latency_ms,

        "success":
            response.status_code == 200
    })


performance_df = pd.DataFrame(
    performance_results
)


print("\n" + "=" * 80)
print("API PERFORMANCE TEST")
print("=" * 80)

display(
    performance_df
)


print(
    "\nAverage API latency:",
    round(
        performance_df[
            "latency_ms"
        ].mean(),
        2
    ),
    "ms"
)


print(
    "Median API latency:",
    round(
        performance_df[
            "latency_ms"
        ].median(),
        2
    ),
    "ms"
)


print(
    "Maximum API latency:",
    round(
        performance_df[
            "latency_ms"
        ].max(),
        2
    ),
    "ms"
)


# ============================================================
# 28. PRODUCTION CONFIGURATION TEMPLATE
# ============================================================
#
# This can become a .env file in a real project.
#
# Never commit real API keys/secrets to Git.
# ============================================================

env_template = """
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=8000

RAG_TOP_K=3
RAG_CANDIDATE_K=10
RAG_SIMILARITY_THRESHOLD=0.05
RAG_MIN_GROUNDING_SCORE=0.35

# Example production model configuration
# LLM_PROVIDER=azure_openai
# AZURE_OPENAI_ENDPOINT=
# AZURE_OPENAI_API_KEY=
# AZURE_OPENAI_DEPLOYMENT=
"""


print("\n" + "=" * 80)
print(".ENV CONFIGURATION TEMPLATE")
print("=" * 80)

print(env_template)


# ============================================================
# 29. REQUIREMENTS.TXT TEMPLATE
# ============================================================
#
# In a real repository, this becomes requirements.txt.
# ============================================================

requirements_txt = """
fastapi
uvicorn[standard]
pydantic
numpy
pandas
scikit-learn
"""


print("\n" + "=" * 80)
print("REQUIREMENTS.TXT")
print("=" * 80)

print(
    requirements_txt
)


# ============================================================
# 30. DOCKERFILE TEMPLATE
# ============================================================
#
# This is a deployment template.
#
# We do not build/download Docker images inside this notebook.
# ============================================================

dockerfile = """
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""


print("\n" + "=" * 80)
print("DOCKERFILE")
print("=" * 80)

print(
    dockerfile
)


# ============================================================
# 31. .DOCKERIGNORE TEMPLATE
# ============================================================

dockerignore = """
__pycache__
*.pyc
.ipynb_checkpoints
.git
.env
venv
.venv
"""


print("\n" + "=" * 80)
print(".DOCKERIGNORE")
print("=" * 80)

print(
    dockerignore
)


# ============================================================
# 32. PRODUCTION PROJECT STRUCTURE
# ============================================================

project_structure = """
enterprise-document-intelligence/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env.example
├── README.md
│
├── src/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── retrieval.py
│   ├── reranking.py
│   ├── rag.py
│   ├── models.py
│   ├── monitoring.py
│   └── config.py
│
├── tests/
│   ├── test_retrieval.py
│   ├── test_rag.py
│   └── test_api.py
│
└── data/
    └── enterprise_documents/
"""


print("\n" + "=" * 80)
print("PRODUCTION PROJECT STRUCTURE")
print("=" * 80)

print(
    project_structure
)


# ============================================================
# 33. CLOUD-READY ARCHITECTURE
# ============================================================
#
# The following architecture can be implemented using:
#
# Azure:
#   Azure Blob Storage
#   Azure AI Search
#   Azure OpenAI
#   Azure Container Apps / AKS
#   Application Insights
#
# AWS:
#   S3
#   OpenSearch
#   Bedrock
#   ECS/EKS
#   CloudWatch
#
# GCP:
#   Cloud Storage
#   Vertex AI Search
#   Gemini
#   Cloud Run/GKE
#   Cloud Monitoring
#
# Our notebook represents the core architecture independently
# of the cloud vendor.
# ============================================================

cloud_architecture = """
                         ┌───────────────────┐
                         │   User / Client   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ API Gateway /     │
                         │ Load Balancer     │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ FastAPI Service   │
                         │ Container         │
                         └─────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
             Query Service   Search Service   Health
                    │
                    ▼
             ┌───────────────┐
             │ RAG Pipeline  │
             └───────┬───────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   Vector Search           Metadata Store
          │
          ▼
      Top-K Context
          │
          ▼
    Reranking Layer
          │
          ▼
     Grounded Prompt
          │
          ▼
       Enterprise
          LLM
          │
          ▼
   Citation / Grounding
       Validation
          │
          ▼
      Final Answer


Document Ingestion Pipeline:

Documents
    ↓
Blob / Object Storage
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Database / Search
    ↓
Metadata


Observability:

FastAPI
    ↓
Metrics + Logs + Traces
    ↓
Prometheus / Grafana
or
Application Insights / Cloud Monitoring
"""


print("\n" + "=" * 80)
print("CLOUD-READY ARCHITECTURE")
print("=" * 80)

print(
    cloud_architecture
)


# ============================================================
# 34. PRODUCTION MONITORING METRICS
# ============================================================
#
# These are the metrics we would monitor after deployment.
# ============================================================

production_metrics = {

    "API metrics": [

        "Request count",

        "Requests per second",

        "HTTP error rate",

        "P50 latency",

        "P95 latency",

        "P99 latency"
    ],

    "Retrieval metrics": [

        "Recall@K",

        "Precision@K",

        "MRR",

        "Hit Rate@K",

        "Similarity score",

        "Empty retrieval rate"
    ],

    "RAG metrics": [

        "Grounded response rate",

        "Citation coverage",

        "Answer relevance",

        "Hallucination rate",

        "Context relevance"
    ],

    "Infrastructure metrics": [

        "CPU utilization",

        "Memory utilization",

        "Container restarts",

        "Network latency",

        "Storage usage"
    ],

    "Business metrics": [

        "Queries per user",

        "Most requested topics",

        "Unanswered questions",

        "Document coverage"
    ]
}


print("\n" + "=" * 80)
print("PRODUCTION MONITORING METRICS")
print("=" * 80)


for category, metric_list in (
    production_metrics.items()
):

    print(
        f"\n{category}"
    )

    for metric in metric_list:

        print(
            f"  - {metric}"
        )


# ============================================================
# 35. PRODUCTION FAILURE SCENARIOS
# ============================================================
#
# A production AI engineer must design for failure.
# ============================================================

failure_strategy = {

    "No documents retrieved":
        "Return safe fallback; do not generate unsupported answer.",

    "Low similarity":
        "Apply similarity threshold and ask user to rephrase.",

    "LLM unavailable":
        "Return controlled service error and log failure.",

    "Vector database unavailable":
        "Fail gracefully and expose degraded health status.",

    "High latency":
        "Monitor P95/P99 and optimize retrieval/model calls.",

    "Hallucinated response":
        "Grounding validation + source citations + evaluation.",

    "Invalid API request":
        "Pydantic validation with HTTP 422.",

    "Internal exception":
        "Global exception handler + request ID + structured logging.",

    "Sensitive information":
        "Authentication, authorization, masking and audit logging.",

    "Prompt injection":
        "Input validation, instruction hierarchy, tool restrictions and output validation."
}


print("\n" + "=" * 80)
print("PRODUCTION FAILURE STRATEGY")
print("=" * 80)


for failure, strategy in (
    failure_strategy.items()
):

    print(
        f"\n{failure}"
    )

    print(
        f"  → {strategy}"
    )


# ============================================================
# 36. FINAL END-TO-END TEST
# ============================================================
#
# This is the final Day 58 validation.
# ============================================================

final_test_queries = [

    (
        "How do employees request annual leave?",
        "HR-001"
    ),

    (
        "Can employees work remotely?",
        "HR-002"
    ),

    (
        "What should I do if my password is compromised?",
        "IT-001"
    ),

    (
        "How are production incidents reported?",
        "IT-002"
    ),

    (
        "How do I submit business expenses?",
        "FIN-001"
    ),

    (
        "What are the API development standards?",
        "DEV-001"
    ),

    (
        "How should production deployments be handled?",
        "DEV-002"
    ),

    (
        "How should a RAG application be monitored?",
        "AI-002"
    ),

    (
        "How should customer data be protected?",
        "BFSI-001"
    )
]


final_validation_rows = []


for query, expected_doc in (
    final_test_queries
):

    start = time.perf_counter()


    response = client.post(

        "/query",

        json={
            "query": query
        }
    )


    latency_ms = (
        time.perf_counter()
        -
        start
    ) * 1000


    response_json = (
        response.json()
    )


    returned_docs = [

        source["document_id"]

        for source in
        response_json.get(
            "sources",
            []
        )
    ]


    source_found = (
        expected_doc
        in returned_docs
    )


    final_validation_rows.append({

        "query":
            query,

        "expected_document":
            expected_doc,

        "retrieved_documents":
            returned_docs,

        "source_found":
            source_found,

        "grounded":
            response_json.get(
                "grounded",
                False
            ),

        "confidence":
            response_json.get(
                "confidence",
                0
            ),

        "latency_ms":
            latency_ms,

        "api_success":
            response.status_code == 200
    })


final_validation_df = pd.DataFrame(
    final_validation_rows
)


print("\n" + "=" * 80)
print("FINAL END-TO-END VALIDATION")
print("=" * 80)

display(
    final_validation_df
)


# ============================================================
# 37. FINAL SCORECARD
# ============================================================

final_source_accuracy = (
    final_validation_df[
        "source_found"
    ].mean()
)


final_grounding_rate = (
    final_validation_df[
        "grounded"
    ].mean()
)


final_api_success_rate = (
    final_validation_df[
        "api_success"
    ].mean()
)


final_average_latency = (
    final_validation_df[
        "latency_ms"
    ].mean()
)


print("\n" + "=" * 80)
print("DAY 58 FINAL SCORECARD")
print("=" * 80)

print(
    f"\nSource Accuracy      : "
    f"{final_source_accuracy:.2%}"
)

print(
    f"Grounding Rate       : "
    f"{final_grounding_rate:.2%}"
)

print(
    f"API Success Rate     : "
    f"{final_api_success_rate:.2%}"
)

print(
    f"Average API Latency  : "
    f"{final_average_latency:.2f} ms"
)


# ============================================================
# 38. FINAL HEALTH CHECK
# ============================================================

final_health = client.get(
    "/health"
)


final_metrics = client.get(
    "/metrics"
)


print("\n" + "=" * 80)
print("FINAL HEALTH CHECK")
print("=" * 80)

print(
    final_health.json()
)


print("\n" + "=" * 80)
print("FINAL APPLICATION METRICS")
print("=" * 80)

print(
    final_metrics.json()
)


# ============================================================
# 39. IMPORTANT VARIABLES CREATED
# ============================================================

print("\n" + "=" * 80)
print("IMPORTANT VARIABLES CREATED IN PART 4")
print("=" * 80)

print("""
APP_CONFIG
    → Application configuration

ApplicationMetrics
    → Lightweight monitoring implementation

metrics
    → Application metrics collector

QueryRequest
    → /query request schema

SearchRequest
    → /search request schema

QueryResponse
    → /query response schema

SearchResponse
    → /search response schema

app
    → FastAPI application

client
    → FastAPI TestClient

api_test_df
    → Automated API test results

performance_df
    → API performance results

env_template
    → Production environment configuration

requirements_txt
    → Python production dependencies

dockerfile
    → Docker deployment configuration

dockerignore
    → Docker ignore configuration

cloud_architecture
    → Production cloud architecture

production_metrics
    → Monitoring metric categories

failure_strategy
    → Production failure-handling strategy

final_validation_df
    → End-to-end validation results

final_source_accuracy
    → Final source retrieval accuracy

final_grounding_rate
    → Final grounded response rate

final_api_success_rate
    → Final API success rate

final_average_latency
    → Final API latency
""")


# ============================================================
# 40. COMPLETE DAY 58 ARCHITECTURE
# ============================================================

print("\n" + "=" * 80)
print("DAY 58 — COMPLETE ENTERPRISE DOCUMENT INTELLIGENCE ASSISTANT")
print("=" * 80)

print("""
                         USER
                          │
                          ▼
                    FastAPI API
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
           /query                  /search
              │                       │
              └───────────┬───────────┘
                          │
                          ▼
                 QUERY NORMALIZATION
                          │
                          ▼
                   QUERY EXPANSION
                          │
                          ▼
                  VECTOR RETRIEVAL
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
                       TOP-K
                          │
                          ▼
                  CONTEXT BUILDER
                          │
                          ▼
                 CONTEXT QUALITY CHECK
                    /            \\
                   /              \\
              Sufficient        Insufficient
                 │                  │
                 ▼                  ▼
          GROUNDED GENERATOR     SAFE FALLBACK
                 │
                 ▼
             CITATIONS
                 │
                 ▼
          GROUNDING CHECK
                 │
                 ▼
          STRUCTURED RESPONSE
                 │
                 ▼
                USER


DOCUMENT INGESTION

Enterprise Documents
        │
        ▼
      Cleaning
        │
        ▼
      Chunking
        │
        ▼
     Metadata
        │
        ▼
  Vector Representation
        │
        ▼
    Vector Index
        │
        ▼
     Retrieval


OBSERVABILITY

API
 │
 ├── Request ID
 ├── Logs
 ├── Latency
 ├── Error Rate
 ├── Retrieval Metrics
 ├── Grounding Metrics
 └── Business Metrics
""")


# ============================================================
# 41. PRODUCTION VERSION WITH AN LLM
# ============================================================

print("""
PRODUCTION LLM VERSION

The current notebook uses a deterministic generator so that
the entire project remains CPU/storage friendly.

For production:

                    Retrieved Context
                           │
                           ▼
                    Grounded Prompt
                           │
                           ▼
                    Enterprise LLM
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
                 Answer       Citations
                    │
                    ▼
              Grounding Check
                    │
                    ▼
               Final Response


The LLM provider can be:

- Azure OpenAI
- OpenAI
- Claude
- Gemini
- Llama
- Qwen
- Enterprise-hosted model

The retrieval API does not need to be redesigned.
Only the generation implementation changes.
""")


# ============================================================
# 42. FINAL PROJECT SUMMARY
# ============================================================

print("""
============================================================
DAY 58 COMPLETE
============================================================

PROJECT:
Enterprise Document Intelligence Assistant

CORE CAPABILITIES:

1. Enterprise document ingestion
2. Text cleaning
3. Intelligent chunking
4. Document metadata
5. Lightweight vector representation
6. Vector similarity search
7. Query normalization
8. Query expansion
9. Metadata filtering
10. Similarity thresholds
11. Candidate retrieval
12. Lightweight reranking
13. Top-K retrieval
14. Retrieval evaluation
15. Grounded context construction
16. Safe answer generation
17. Source citations
18. Grounding validation
19. Hallucination-aware fallback
20. RAG evaluation
21. FastAPI REST API
22. Health monitoring
23. Application metrics
24. Structured logging
25. Request IDs
26. API testing
27. Performance testing
28. Docker-ready architecture
29. Cloud-ready architecture
30. Production monitoring strategy

============================================================
""")


# ============================================================
# 43. RESUME BULLET
# ============================================================

resume_bullet = """
Built an enterprise document intelligence assistant using a
production-style RAG architecture with document chunking,
vector retrieval, metadata filtering, query expansion,
lightweight reranking, grounded response generation, source
citations, hallucination-aware fallbacks, FastAPI APIs,
evaluation metrics, monitoring, and cloud-ready deployment
architecture.
"""


print("\n" + "=" * 80)
print("RESUME BULLET")
print("=" * 80)

print(
    resume_bullet
)


# ============================================================
# 44. GITHUB PROJECT DESCRIPTION
# ============================================================

github_description = """
Enterprise Document Intelligence Assistant — a production-style
RAG application for querying enterprise knowledge bases with
advanced retrieval, metadata filtering, reranking, grounded
responses, source citations, evaluation, FastAPI APIs,
monitoring, and cloud-ready architecture.
"""


print("\n" + "=" * 80)
print("GITHUB DESCRIPTION")
print("=" * 80)

print(
    github_description
)


# ============================================================
# 45. INTERVIEW EXPLANATION
# ============================================================

interview_answer = """
INTERVIEW ANSWER:

"I built an Enterprise Document Intelligence Assistant using a
modular RAG architecture.

The ingestion layer cleans enterprise documents, splits them into
overlapping chunks and attaches metadata such as document ID,
department and document type.

For retrieval, I initially use vector similarity and then improve
retrieval using query normalization, query expansion, metadata
filtering, similarity thresholds and lightweight reranking.

The top relevant chunks are passed into a grounded generation
layer. The generation prompt instructs the model to use only the
retrieved enterprise context and avoid unsupported claims.

I also implemented a context sufficiency check and safe fallback
so that the system does not blindly answer when relevant knowledge
is unavailable.

Every answer contains source identifiers for traceability.

I exposed the complete system through FastAPI with /query,
/search, /health and /metrics endpoints. I added request IDs,
logging, latency monitoring, error handling and API validation.

For evaluation I measured retrieval metrics such as Hit Rate,
Precision@K, Recall@K and MRR, as well as grounding rate,
source accuracy and end-to-end latency.

The architecture is cloud-ready. In production I would replace
the lightweight TF-IDF representation with dense embeddings and
a managed vector search service, and replace the deterministic
generator with an enterprise LLM such as Azure OpenAI.

The important design principle is that ingestion, retrieval,
generation, evaluation and API layers are decoupled, so each
component can evolve independently."
"""  # <--- ADDED MISSING TRIPLE QUOTES HERE TO CLOSE THE STRING


print("\n" + "=" * 80)
print("FAANG-STYLE INTERVIEW EXPLANATION")
print("=" * 80)

print(
    interview_answer
)


# ============================================================
# 46. FINAL PROJECT SCORECARD
# ============================================================

project_scorecard = pd.DataFrame({

    "Area": [

        "Document ingestion",

        "Chunking",

        "Vector retrieval",

        "Advanced retrieval",

        "Reranking",

        "Metadata filtering",

        "RAG generation",

        "Grounding",

        "Citations",

        "Evaluation",

        "FastAPI",

        "Monitoring",

        "Testing",

        "Deployment architecture"
    ],

    "Implemented": [

        True,

        True,

        True,

        True,

        True,

        True,

        True,

        True,

        True,

        True,

        True,

        True,

        True,

        True
    ]
})  # Properly closed DataFrame call


print("\n" + "=" * 80)
print("PROJECT COMPLETION SCORECARD")
print("=" * 80)

display(
    project_scorecard
)


print("\n" + "=" * 80)
print("DAY 58 — ENTERPRISE DOCUMENT INTELLIGENCE ASSISTANT")
print("PROJECT STATUS: COMPLETE")
print("=" * 80)
