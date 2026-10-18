
# ============================================================
# DAY 69/100 — PART 1
# Enterprise Knowledge Base + Keyword Retrieval Engine
# CPU-friendly | Lightweight | Synthetic enterprise data
# ============================================================

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ------------------------------------------------------------

CONFIG = {
    "project_name": "Day 69 - Hybrid Retrieval Engine",
    "chunk_size_words": 55,
    "chunk_overlap_words": 10,
    "top_k": 3,
    "min_similarity": 0.0,
    "output_dir": "day69_hybrid_retrieval/part1",
    "max_features": 5000,
}

OUTPUT_DIR = Path(CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("DAY 69 — PART 1: KNOWLEDGE BASE & KEYWORD RETRIEVAL")
print("=" * 65)


# ------------------------------------------------------------
# 2. CREATE A SYNTHETIC ENTERPRISE DOCUMENT CORPUS
# ------------------------------------------------------------

DOCUMENTS = [
    {
        "doc_id": "DOC001",
        "title": "API Security Policy",
        "category": "Security",
        "text": """
        All internal APIs must use authentication and authorization.
        Production endpoints must validate access tokens and enforce
        role-based access control. Sensitive credentials must never be
        hardcoded in source code. API traffic must use HTTPS. Security
        reviews should be completed before production deployment.
        """
    },
    {
        "doc_id": "DOC002",
        "title": "Incident Response Procedure",
        "category": "Operations",
        "text": """
        When a production incident occurs, the on-call engineer must
        acknowledge the alert, assess the severity, and notify the
        incident response team. Critical incidents require immediate
        escalation. Engineers should preserve logs, document actions,
        identify root causes, and prepare a post-incident report.
        """
    },
    {
        "doc_id": "DOC003",
        "title": "Database Backup and Recovery",
        "category": "Database",
        "text": """
        Production databases must have scheduled backups and a tested
        recovery procedure. Backup frequency depends on recovery point
        objectives. Recovery time objectives define how quickly services
        must be restored. Backup integrity must be verified periodically,
        and restoration tests must be documented.
        """
    },
    {
        "doc_id": "DOC004",
        "title": "Employee Leave Policy",
        "category": "Human Resources",
        "text": """
        Employees should submit planned leave requests through the
        internal HR portal. Managers review requests based on team
        capacity and company policy. Emergency leave should be reported
        to the manager as soon as practical. Leave balances are maintained
        in the employee management system.
        """
    },
    {
        "doc_id": "DOC005",
        "title": "Cloud Deployment Standards",
        "category": "Cloud",
        "text": """
        Applications must pass automated tests before deployment.
        Deployment pipelines should include code quality checks,
        dependency scanning, and environment validation. Production
        secrets must be stored in an approved secrets manager.
        Rollback procedures must be available for critical releases.
        """
    },
    {
        "doc_id": "DOC006",
        "title": "Customer Data Privacy",
        "category": "Compliance",
        "text": """
        Customer personal information must be processed only for
        approved business purposes. Access to sensitive records must
        follow least-privilege principles. Logs should avoid exposing
        personal data. Data retention and deletion must comply with
        applicable policies and regulatory requirements.
        """
    },
    {
        "doc_id": "DOC007",
        "title": "RAG Application Guidelines",
        "category": "AI Engineering",
        "text": """
        Retrieval augmented generation systems retrieve relevant
        document passages before generating answers. Document chunks
        should preserve useful context. Retrieval quality should be
        evaluated using precision at k, recall at k, and reciprocal rank.
        Answers should be grounded in retrieved evidence, and unsupported
        claims should be handled carefully.
        """
    },
    {
        "doc_id": "DOC008",
        "title": "Vector Database Operations",
        "category": "AI Engineering",
        "text": """
        Vector databases store numerical representations of text called
        embeddings. Semantic search compares query and document vectors
        using similarity measures such as cosine similarity. Index
        configuration affects retrieval latency, memory usage, and
        recall. Vector indexes should be monitored and periodically tested.
        """
    },
    {
        "doc_id": "DOC009",
        "title": "API Performance Guidelines",
        "category": "Engineering",
        "text": """
        API response time should be monitored using latency percentiles.
        Caching can reduce repeated computation, while connection pooling
        helps manage database connections. Services should use timeouts,
        controlled retries, and request limits. Performance tests should
        measure throughput, error rate, and response latency.
        """
    },
    {
        "doc_id": "DOC010",
        "title": "Access Review Process",
        "category": "Security",
        "text": """
        Access permissions must be reviewed periodically. Administrators
        should remove obsolete accounts and revoke unnecessary privileges.
        Access reviews should identify excessive permissions, inactive
        users, and policy violations. Review findings must be recorded
        and remediated according to organizational requirements.
        """
    },
    {
        "doc_id": "DOC011",
        "title": "Machine Learning Model Monitoring",
        "category": "Machine Learning",
        "text": """
        Machine learning services should monitor prediction latency,
        input quality, and model performance. Data drift occurs when
        production input distributions change. Model quality should be
        evaluated against a documented baseline. Alerts should trigger
        investigation when important metrics cross defined thresholds.
        """
    },
    {
        "doc_id": "DOC012",
        "title": "Software Testing Standards",
        "category": "Engineering",
        "text": """
        Software teams should maintain unit, integration, and API tests.
        Tests must cover expected behavior, validation failures, and
        important edge cases. Continuous integration should execute
        automated tests before merging changes. Test failures must be
        investigated rather than ignored.
        """
    },
]

DOCUMENTS_DF = pd.DataFrame(DOCUMENTS)

print(f"\nDocuments loaded: {len(DOCUMENTS)}")
print("\nDocument inventory:")
print(
    DOCUMENTS_DF[["doc_id", "title", "category"]]
    .to_string(index=False)
)


# ------------------------------------------------------------
# 3. NORMALIZE DOCUMENT TEXT
# ------------------------------------------------------------

def normalize_text(text):
    """Normalize whitespace while preserving useful word content."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s./-]", " ", text)
    return text.strip()


for document in DOCUMENTS:
    document["normalized_text"] = normalize_text(document["text"])


# ------------------------------------------------------------
# 4. SPLIT DOCUMENTS INTO OVERLAPPING CHUNKS
# ------------------------------------------------------------

def chunk_text(text, chunk_size=55, overlap=10):
    """
    Split text into word-based chunks.

    Overlap preserves context between neighboring chunks.
    Example: size=55, overlap=10 means the next chunk starts
    45 words after the previous chunk starts.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    words = text.split()

    if not words:
        return []

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


CHUNKS = []

for document in DOCUMENTS:
    document_chunks = chunk_text(
        document["normalized_text"],
        chunk_size=CONFIG["chunk_size_words"],
        overlap=CONFIG["chunk_overlap_words"],
    )

    for chunk_index, chunk in enumerate(document_chunks):
        CHUNKS.append({
            "chunk_id": f'{document["doc_id"]}_CH{chunk_index + 1:03d}',
            "doc_id": document["doc_id"],
            "title": document["title"],
            "category": document["category"],
            "chunk_index": chunk_index,
            "text": chunk,
            "word_count": len(chunk.split()),
        })

CHUNKS_DF = pd.DataFrame(CHUNKS)

if CHUNKS_DF.empty:
    raise ValueError("No chunks were created from the document corpus")

print(f"\nTotal chunks created: {len(CHUNKS_DF)}")
print(
    "Average words per chunk:",
    round(CHUNKS_DF["word_count"].mean(), 2)
)
print("\nSample chunks:")
print(
    CHUNKS_DF[
        ["chunk_id", "title", "word_count", "text"]
    ].head(5).to_string(index=False)
)


# ------------------------------------------------------------
# 5. BUILD THE TF-IDF INDEX
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=CONFIG["max_features"],
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
    norm="l2",
)

DOCUMENT_VECTORS = vectorizer.fit_transform(CHUNKS_DF["text"])

print("\nTF-IDF index built.")
print("Vector matrix shape:", DOCUMENT_VECTORS.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))


# ------------------------------------------------------------
# 6. IMPLEMENT KEYWORD RETRIEVAL
# ------------------------------------------------------------

def retrieve_documents(query, top_k=None, min_similarity=None):
    """
    Retrieve ranked chunks using TF-IDF cosine similarity.

    Returns:
        query: original query
        result_count: number of returned chunks
        latency_ms: measured retrieval latency
        results: ranked list containing doc_id, title, text,
                 chunk_id, category, and similarity score
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if top_k is None:
        top_k = CONFIG["top_k"]

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    if min_similarity is None:
        min_similarity = CONFIG["min_similarity"]

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1")

    start_time = time.perf_counter()

    query_vector = vectorizer.transform([normalize_text(query)])

    scores = cosine_similarity(
        query_vector,
        DOCUMENT_VECTORS
    ).ravel()

    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indices:
        score = float(scores[index])

        if score < min_similarity:
            break

        if score <= 0:
            continue

        chunk = CHUNKS_DF.iloc[int(index)]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "category": chunk["category"],
            "text": chunk["text"],
            "similarity_score": round(score, 6),
        })

        if len(results) >= top_k:
            break

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "query": query,
        "result_count": len(results),
        "latency_ms": round(latency_ms, 3),
        "results": results,
    }


# ------------------------------------------------------------
# 7. RUN SAMPLE SEARCH QUERIES
# ------------------------------------------------------------

SAMPLE_QUERIES = [
    "How do we protect internal API endpoints?",
    "What steps should engineers follow during an outage?",
    "How can we restore a production database?",
    "How does semantic search use text embeddings?",
    "How can we measure RAG retrieval quality?",
    "How do we reduce API response latency?",
    "What checks are required before deploying software?",
]

SEARCH_RESULTS = []

print("\n" + "=" * 65)
print("SAMPLE SEARCH RESULTS")
print("=" * 65)

for query in SAMPLE_QUERIES:
    response = retrieve_documents(query, top_k=3)

    print(f"\nQuery: {query}")
    print(f"Latency: {response['latency_ms']} ms")

    if not response["results"]:
        print("No matching chunks found.")
        continue

    for rank, result in enumerate(response["results"], start=1):
        print(
            f"  {rank}. {result['title']} "
            f"| score={result['similarity_score']:.4f}"
        )
        print(f"     {result['text'][:180]}...")

        SEARCH_RESULTS.append({
            "query": query,
            "rank": rank,
            "chunk_id": result["chunk_id"],
            "doc_id": result["doc_id"],
            "title": result["title"],
            "similarity_score": result["similarity_score"],
            "latency_ms": response["latency_ms"],
        })


# ------------------------------------------------------------
# 8. BASIC RETRIEVAL QUALITY CHECKS
# ------------------------------------------------------------

# Known relevant documents for our synthetic test queries.
# These are labels for evaluation, not model-generated answers.

EVALUATION_DATASET = [
    {
        "query": "How do we protect internal API endpoints?",
        "relevant_doc_ids": ["DOC001"],
    },
    {
        "query": "What steps should engineers follow during an outage?",
        "relevant_doc_ids": ["DOC002"],
    },
    {
        "query": "How can we restore a production database?",
        "relevant_doc_ids": ["DOC003"],
    },
    {
        "query": "How does semantic search use text embeddings?",
        "relevant_doc_ids": ["DOC008"],
    },
    {
        "query": "How can we measure RAG retrieval quality?",
        "relevant_doc_ids": ["DOC007"],
    },
    {
        "query": "How do we reduce API response latency?",
        "relevant_doc_ids": ["DOC009"],
    },
    {
        "query": "What checks are required before deploying software?",
        "relevant_doc_ids": ["DOC005", "DOC012"],
    },
]


def precision_at_k(retrieved_ids, relevant_ids, k=3):
    """Fraction of top-k retrieved document IDs that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")

    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    hits = sum(doc_id in relevant_ids for doc_id in retrieved)
    return hits / len(retrieved)


def recall_at_k(retrieved_ids, relevant_ids, k=3):
    """Fraction of known relevant document IDs found in top-k."""
    if not relevant_ids:
        return 0.0

    retrieved = set(retrieved_ids[:k])
    return len(retrieved.intersection(relevant_ids)) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """Reciprocal rank of the first relevant retrieved document."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(dataset, k=3):
    """Evaluate retrieval against the labeled synthetic dataset."""
    rows = []

    for case in dataset:
        response = retrieve_documents(case["query"], top_k=k)

        # Deduplicate document IDs because a document may have
        # multiple chunks in the ranked results.
        retrieved_ids = list(dict.fromkeys(
            item["doc_id"] for item in response["results"]
        ))

        relevant_ids = case["relevant_doc_ids"]

        rows.append({
            "query": case["query"],
            "precision_at_k": precision_at_k(
                retrieved_ids, relevant_ids, k
            ),
            "recall_at_k": recall_at_k(
                retrieved_ids, relevant_ids, k
            ),
            "reciprocal_rank": reciprocal_rank(
                retrieved_ids, relevant_ids
            ),
            "retrieved_doc_ids": ", ".join(retrieved_ids),
            "relevant_doc_ids": ", ".join(relevant_ids),
            "latency_ms": response["latency_ms"],
        })

    return pd.DataFrame(rows)


RETRIEVAL_EVALUATION_DF = evaluate_retrieval(
    EVALUATION_DATASET,
    k=CONFIG["top_k"],
)

print("\n" + "=" * 65)
print("RETRIEVAL EVALUATION")
print("=" * 65)

print(RETRIEVAL_EVALUATION_DF.to_string(index=False))

METRICS_SUMMARY = {
    "evaluation_queries": len(RETRIEVAL_EVALUATION_DF),
    "precision_at_3": round(
        RETRIEVAL_EVALUATION_DF["precision_at_k"].mean(), 4
    ),
    "recall_at_3": round(
        RETRIEVAL_EVALUATION_DF["recall_at_k"].mean(), 4
    ),
    "mean_reciprocal_rank": round(
        RETRIEVAL_EVALUATION_DF["reciprocal_rank"].mean(), 4
    ),
    "average_retrieval_latency_ms": round(
        RETRIEVAL_EVALUATION_DF["latency_ms"].mean(), 3
    ),
}


# ------------------------------------------------------------
# 9. SAVE PROJECT ARTIFACTS
# ------------------------------------------------------------

DOCUMENTS_DF.to_csv(
    OUTPUT_DIR / "documents.csv",
    index=False
)

CHUNKS_DF.to_csv(
    OUTPUT_DIR / "document_chunks.csv",
    index=False
)

pd.DataFrame(SEARCH_RESULTS).to_csv(
    OUTPUT_DIR / "sample_search_results.csv",
    index=False
)

RETRIEVAL_EVALUATION_DF.to_csv(
    OUTPUT_DIR / "retrieval_evaluation.csv",
    index=False
)

pd.DataFrame(
    [{"metric": key, "value": value}
     for key, value in METRICS_SUMMARY.items()]
).to_csv(
    OUTPUT_DIR / "metrics_summary.csv",
    index=False
)

print("\n" + "=" * 65)
print("PART 1 COMPLETED — EXECUTION SUMMARY")
print("=" * 65)

print(f"Documents: {len(DOCUMENTS_DF)}")
print(f"Chunks: {len(CHUNKS_DF)}")
print(f"Vocabulary terms: {len(vectorizer.vocabulary_)}")
print(f"TF-IDF matrix shape: {DOCUMENT_VECTORS.shape}")
print("\nMetrics summary:")

for metric, value in METRICS_SUMMARY.items():
    print(f"  {metric}: {value}")

print(f"\nArtifacts saved to: {OUTPUT_DIR.resolve()}")
print("\nAvailable variables for Part 2:")
print("  DOCUMENTS")
print("  DOCUMENTS_DF")
print("  CHUNKS_DF")
print("  vectorizer")
print("  DOCUMENT_VECTORS")
print("  retrieve_documents")
print("  EVALUATION_DATASET")
print("  precision_at_k")
print("  recall_at_k")
print("  reciprocal_rank")
print("  RETRIEVAL_EVALUATION_DF")
print("  METRICS_SUMMARY")


# ============================================================
# DAY 69/100 — PART 2
# Lightweight Latent-Semantic Search & Vector Similarity
# CPU-friendly | No model download | Continues from Part 1
# ============================================================

import re
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1. CONFIGURATION AND INPUT VALIDATION
# ------------------------------------------------------------

PART2_CONFIG = {
    "output_dir": "day69_hybrid_retrieval/part2",
    "embedding_dimensions": 32,
    "top_k": 3,
    "random_state": 42,
    "max_features": 5000,
}

OUTPUT_DIR = Path(PART2_CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if "CHUNKS_DF" not in globals():
    raise RuntimeError(
        "CHUNKS_DF is missing. Run the complete Part 1 cell first."
    )

required_columns = {"chunk_id", "doc_id", "title", "text"}
missing = required_columns - set(CHUNKS_DF.columns)

if missing:
    raise ValueError(f"CHUNKS_DF is missing columns: {missing}")

if CHUNKS_DF.empty:
    raise ValueError("CHUNKS_DF is empty. Check Part 1.")

chunks = CHUNKS_DF.copy().reset_index(drop=True)
chunks["text"] = chunks["text"].fillna("").astype(str)

print("=" * 65)
print("DAY 69 — PART 2: LATENT-SEMANTIC VECTOR SEARCH")
print("=" * 65)
print(f"Input chunks: {len(chunks)}")


# ------------------------------------------------------------
# 2. CREATE A LATENT-SEMANTIC REPRESENTATION
# ------------------------------------------------------------
# TF-IDF represents words and phrases.
# Truncated SVD compresses these vectors into a smaller
# latent space that can capture some broader term relationships.
#
# Note: this is NOT a pretrained neural embedding model.
# It is an efficient, local latent-semantic baseline.

semantic_vectorizer = TfidfVectorizer(
    max_features=PART2_CONFIG["max_features"],
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
)

tfidf_matrix = semantic_vectorizer.fit_transform(chunks["text"])

n_samples, n_features = tfidf_matrix.shape

if n_features == 0:
    raise ValueError(
        "The corpus produced no vocabulary. Check document text."
    )

# SVD needs at least two features and enough samples to reduce
# dimensions meaningfully. For very small corpora, retain the
# normalized TF-IDF vectors as a safe fallback.
use_svd = n_samples >= 2 and n_features >= 2

if use_svd:
    n_components = min(
        PART2_CONFIG["embedding_dimensions"],
        n_samples - 1,
        n_features - 1,
    )

    if n_components >= 1:
        svd_model = TruncatedSVD(
            n_components=n_components,
            random_state=PART2_CONFIG["random_state"],
            n_iter=5,
        )

        chunk_vectors = svd_model.fit_transform(tfidf_matrix)
        chunk_vectors = normalize(chunk_vectors, norm="l2")

        representation_type = "TF-IDF + Truncated SVD"
        explained_variance = float(
            svd_model.explained_variance_ratio_.sum()
        )
    else:
        chunk_vectors = normalize(
            tfidf_matrix.toarray(), norm="l2"
        )
        svd_model = None
        representation_type = "Normalized TF-IDF fallback"
        explained_variance = None
else:
    chunk_vectors = normalize(
        tfidf_matrix.toarray(), norm="l2"
    )
    svd_model = None
    representation_type = "Normalized TF-IDF fallback"
    explained_variance = None

print(f"Representation: {representation_type}")
print(f"Vector matrix shape: {chunk_vectors.shape}")

if explained_variance is not None:
    print(
        "SVD explained variance ratio sum:",
        round(explained_variance, 4),
    )


# ------------------------------------------------------------
# 3. IMPLEMENT SEMANTIC-SPACE RETRIEVAL
# ------------------------------------------------------------

def normalize_query(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s./-]", " ", text)
    return text.strip()


def encode_query(query):
    """Transform a query into the same space as chunk vectors."""
    query = normalize_query(query)

    query_tfidf = semantic_vectorizer.transform([query])

    if svd_model is not None:
        query_vector = svd_model.transform(query_tfidf)
    else:
        query_vector = query_tfidf.toarray()

    return normalize(query_vector, norm="l2")


def semantic_search(query, top_k=None, min_similarity=0.0):
    """
    Retrieve chunks using cosine similarity in the reduced
    latent-semantic space.

    Returns a dictionary compatible with Part 1's result format.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if top_k is None:
        top_k = PART2_CONFIG["top_k"]

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer")

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1")

    start_time = time.perf_counter()

    query_vector = encode_query(query)
    scores = cosine_similarity(
        query_vector, chunk_vectors
    ).ravel()

    ranked_indices = np.argsort(scores)[::-1]
    results = []

    for index in ranked_indices:
        score = float(scores[index])

        # Skip empty or non-matching vectors.
        if score <= 0 or score < min_similarity:
            continue

        chunk = chunks.iloc[int(index)]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "category": chunk["category"]
                if "category" in chunks.columns else "Unknown",
            "text": chunk["text"],
            "similarity_score": round(score, 6),
        })

        if len(results) >= top_k:
            break

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return {
        "query": query,
        "result_count": len(results),
        "latency_ms": round(latency_ms, 3),
        "results": results,
        "representation": representation_type,
    }


# ------------------------------------------------------------
# 4. COMPARE KEYWORD SEARCH WITH LATENT-SEMANTIC SEARCH
# ------------------------------------------------------------
# Uses Part 1's retriever as the keyword baseline.
# If it is unavailable, the comparison is skipped gracefully.

comparison_queries = [
    "How do I recover a service after a failure?",
    "How can confidential information be protected?",
    "How can I find passages with the same meaning?",
    "What helps restore a database after data loss?",
    "How can we make an API respond faster?",
    "What must happen before software is released?",
]

comparison_rows = []

for query in comparison_queries:
    semantic_result = semantic_search(query, top_k=3)

    keyword_result = None

    if "retrieve_documents" in globals():
        try:
            keyword_result = retrieve_documents(query, top_k=3)
        except Exception as exc:
            print(
                f"Keyword baseline failed for query "
                f"{query!r}: {exc}"
            )

    semantic_ids = [
        item["doc_id"] for item in semantic_result["results"]
    ]

    keyword_ids = (
        [item["doc_id"] for item in keyword_result["results"]]
        if keyword_result is not None else []
    )

    comparison_rows.append({
        "query": query,
        "keyword_top_results": " | ".join(keyword_ids),
        "latent_semantic_top_results": " | ".join(semantic_ids),
        "keyword_latency_ms": (
            keyword_result["latency_ms"]
            if keyword_result is not None else np.nan
        ),
        "latent_semantic_latency_ms": semantic_result["latency_ms"],
    })

COMPARISON_DF = pd.DataFrame(comparison_rows)

print("\n" + "=" * 65)
print("KEYWORD VS LATENT-SEMANTIC SEARCH")
print("=" * 65)
print(COMPARISON_DF.to_string(index=False))


# ------------------------------------------------------------
# 5. EVALUATE SEMANTIC RETRIEVAL
# ------------------------------------------------------------
# Reuse the ground-truth dataset from Part 1 if available.
# Otherwise, use the known labels defined below.

if "EVALUATION_DATASET" not in globals():
    EVALUATION_DATASET = [
        {
            "query": "How do we protect internal API endpoints?",
            "relevant_doc_ids": ["DOC001"],
        },
        {
            "query": "What steps should engineers follow during an outage?",
            "relevant_doc_ids": ["DOC002"],
        },
        {
            "query": "How can we restore a production database?",
            "relevant_doc_ids": ["DOC003"],
        },
        {
            "query": "How does semantic search use text embeddings?",
            "relevant_doc_ids": ["DOC008"],
        },
        {
            "query": "How can we measure RAG retrieval quality?",
            "relevant_doc_ids": ["DOC007"],
        },
        {
            "query": "How do we reduce API response latency?",
            "relevant_doc_ids": ["DOC009"],
        },
        {
            "query": "What checks are required before deploying software?",
            "relevant_doc_ids": ["DOC005", "DOC012"],
        },
    ]


def semantic_precision_at_k(retrieved_ids, relevant_ids, k=3):
    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    hits = sum(
        doc_id in relevant_ids for doc_id in retrieved
    )
    return hits / len(retrieved)


def semantic_recall_at_k(retrieved_ids, relevant_ids, k=3):
    if not relevant_ids:
        return 0.0

    retrieved = set(retrieved_ids[:k])
    return (
        len(retrieved.intersection(relevant_ids))
        / len(relevant_ids)
    )


def semantic_reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def evaluate_semantic_search(dataset, k=3):
    rows = []

    for case in dataset:
        response = semantic_search(case["query"], top_k=k)

        # Deduplicate documents so multiple chunks from the same
        # document do not inflate document-level evaluation.
        retrieved_ids = list(dict.fromkeys(
            item["doc_id"] for item in response["results"]
        ))

        relevant_ids = case["relevant_doc_ids"]

        rows.append({
            "query": case["query"],
            "precision_at_k": semantic_precision_at_k(
                retrieved_ids, relevant_ids, k
            ),
            "recall_at_k": semantic_recall_at_k(
                retrieved_ids, relevant_ids, k
            ),
            "reciprocal_rank": semantic_reciprocal_rank(
                retrieved_ids, relevant_ids
            ),
            "retrieved_doc_ids": ", ".join(retrieved_ids),
            "relevant_doc_ids": ", ".join(relevant_ids),
            "latency_ms": response["latency_ms"],
        })

    return pd.DataFrame(rows)


SEMANTIC_EVALUATION_DF = evaluate_semantic_search(
    EVALUATION_DATASET,
    k=PART2_CONFIG["top_k"],
)

SEMANTIC_METRICS = {
    "representation": representation_type,
    "evaluation_queries": len(SEMANTIC_EVALUATION_DF),
    "precision_at_3": round(
        SEMANTIC_EVALUATION_DF["precision_at_k"].mean(), 4
    ),
    "recall_at_3": round(
        SEMANTIC_EVALUATION_DF["recall_at_k"].mean(), 4
    ),
    "mean_reciprocal_rank": round(
        SEMANTIC_EVALUATION_DF["reciprocal_rank"].mean(), 4
    ),
    "average_latency_ms": round(
        SEMANTIC_EVALUATION_DF["latency_ms"].mean(), 3
    ),
}

print("\n" + "=" * 65)
print("LATENT-SEMANTIC RETRIEVAL EVALUATION")
print("=" * 65)
print(SEMANTIC_EVALUATION_DF.to_string(index=False))

print("\nMetric summary:")
for metric, value in SEMANTIC_METRICS.items():
    print(f"  {metric}: {value}")


# ------------------------------------------------------------
# 6. SAVE PART 2 ARTIFACTS
# ------------------------------------------------------------

vector_table = pd.DataFrame(
    chunk_vectors,
    columns=[
        f"dimension_{i + 1}"
        for i in range(chunk_vectors.shape[1])
    ],
)

vector_table.insert(0, "chunk_id", chunks["chunk_id"].values)
vector_table.insert(1, "doc_id", chunks["doc_id"].values)

vector_table.to_csv(
    OUTPUT_DIR / "latent_semantic_vectors.csv",
    index=False,
)

COMPARISON_DF.to_csv(
    OUTPUT_DIR / "keyword_vs_semantic_comparison.csv",
    index=False,
)

SEMANTIC_EVALUATION_DF.to_csv(
    OUTPUT_DIR / "semantic_retrieval_evaluation.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "semantic_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(SEMANTIC_METRICS, file, indent=2)

with open(
    OUTPUT_DIR / "part2_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            **PART2_CONFIG,
            "representation_type": representation_type,
            "tfidf_features": int(n_features),
            "vector_dimensions": int(chunk_vectors.shape[1]),
            "explained_variance_ratio_sum": explained_variance,
            "note": (
                "Truncated SVD provides a lightweight "
                "latent-semantic baseline; it is not a "
                "pretrained neural sentence embedding model."
            ),
        },
        file,
        indent=2,
    )

print("\n" + "=" * 65)
print("PART 2 EXECUTION SUMMARY")
print("=" * 65)
print(f"Chunks indexed: {len(chunks)}")
print(f"Representation: {representation_type}")
print(f"Vector dimensions: {chunk_vectors.shape[1]}")
print(f"Artifacts saved to: {OUTPUT_DIR.resolve()}")
print("\nMain function available for Part 3:")
print("  semantic_search(query, top_k=3)")
print("\nMain variables:")
print("  chunks")
print("  semantic_vectorizer")
print("  svd_model")
print("  chunk_vectors")
print("  COMPARISON_DF")
print("  SEMANTIC_EVALUATION_DF")
print("  SEMANTIC_METRICS")


# ============================================================
# DAY 69/100 — PART 3
# Hybrid Retrieval, Score Fusion & Lightweight Reranking
# CPU-friendly | No additional model downloads
# ============================================================

import json
import time
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ------------------------------------------------------------
# 1. CONFIGURATION AND VALIDATION
# ------------------------------------------------------------

PART3_CONFIG = {
    "output_dir": "day69_hybrid_retrieval/part3",
    "top_k": 3,
    "keyword_weight": 0.50,
    "semantic_weight": 0.50,
    "rrf_constant": 60,
    "lexical_rerank_weight": 0.15,
    "evaluation_k": 3,
    "random_state": 42,
}

OUTPUT_DIR = Path(PART3_CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if "CHUNKS_DF" not in globals() or CHUNKS_DF.empty:
    raise RuntimeError("Run Part 1 first to create CHUNKS_DF.")

chunks = CHUNKS_DF.copy().reset_index(drop=True)
chunks["text"] = chunks["text"].fillna("").astype(str)

required_columns = {"chunk_id", "doc_id", "title", "text"}
missing_columns = required_columns - set(chunks.columns)

if missing_columns:
    raise ValueError(f"Missing columns in CHUNKS_DF: {missing_columns}")

if not (
    0 <= PART3_CONFIG["keyword_weight"] <= 1
    and 0 <= PART3_CONFIG["semantic_weight"] <= 1
):
    raise ValueError("Fusion weights must be between 0 and 1.")

weight_total = (
    PART3_CONFIG["keyword_weight"]
    + PART3_CONFIG["semantic_weight"]
)

if weight_total <= 0:
    raise ValueError("At least one fusion weight must be positive.")

print("=" * 68)
print("DAY 69 — PART 3: HYBRID RETRIEVAL & RERANKING")
print("=" * 68)
print(f"Indexed chunks: {len(chunks)}")


# ------------------------------------------------------------
# 2. BUILD COMPATIBLE KEYWORD AND SEMANTIC INDEXES
# ------------------------------------------------------------
# Rebuild indexes against the current chunks to ensure all score
# arrays refer to the same chunk order.
#
# This also makes Part 3 robust to variables from earlier cells
# having been overwritten or created in a different order.

keyword_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
)

keyword_matrix = keyword_vectorizer.fit_transform(chunks["text"])

semantic_vectorizer_p3 = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
)

semantic_tfidf_matrix = semantic_vectorizer_p3.fit_transform(
    chunks["text"]
)

n_samples, n_features = semantic_tfidf_matrix.shape

if n_samples >= 2 and n_features >= 2:
    n_components = min(32, n_samples - 1, n_features - 1)

    from sklearn.decomposition import TruncatedSVD

    svd_model_p3 = TruncatedSVD(
        n_components=n_components,
        random_state=PART3_CONFIG["random_state"],
        n_iter=5,
    )

    semantic_matrix = svd_model_p3.fit_transform(
        semantic_tfidf_matrix
    )

    semantic_matrix = normalize(semantic_matrix, norm="l2")
    semantic_method = "TF-IDF + Truncated SVD"
else:
    svd_model_p3 = None
    semantic_matrix = normalize(
        semantic_tfidf_matrix.toarray(),
        norm="l2",
    )
    semantic_method = "Normalized TF-IDF fallback"

print("Keyword index:", keyword_matrix.shape)
print("Semantic index:", semantic_matrix.shape)
print("Semantic representation:", semantic_method)


# ------------------------------------------------------------
# 3. QUERY PROCESSING AND RAW SIMILARITY SCORES
# ------------------------------------------------------------

def clean_query(query):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    query = query.lower()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"[^\w\s./-]", " ", query)

    return query.strip()


def get_similarity_scores(query):
    """Return keyword and latent-semantic scores for every chunk."""
    query = clean_query(query)

    keyword_query_vector = keyword_vectorizer.transform([query])

    keyword_scores = cosine_similarity(
        keyword_query_vector,
        keyword_matrix,
    ).ravel()

    semantic_query_tfidf = semantic_vectorizer_p3.transform([query])

    if svd_model_p3 is not None:
        semantic_query_vector = svd_model_p3.transform(
            semantic_query_tfidf
        )
    else:
        semantic_query_vector = semantic_query_tfidf.toarray()

    semantic_query_vector = normalize(
        semantic_query_vector,
        norm="l2",
    )

    semantic_scores = cosine_similarity(
        semantic_query_vector,
        semantic_matrix,
    ).ravel()

    return keyword_scores, semantic_scores


# ------------------------------------------------------------
# 4. SCORE NORMALIZATION AND WEIGHTED SCORE FUSION
# ------------------------------------------------------------

def minmax_normalize(scores):
    """Normalize scores into [0, 1] for within-query comparison."""
    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        return scores

    minimum = float(np.min(scores))
    maximum = float(np.max(scores))

    if maximum - minimum < 1e-12:
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


def weighted_score_fusion(
    keyword_scores,
    semantic_scores,
    keyword_weight=0.5,
    semantic_weight=0.5,
):
    """
    Normalize both score distributions and combine them.

    The weights are configurable and normalized to sum to one.
    """
    total = keyword_weight + semantic_weight

    if total <= 0:
        raise ValueError("Fusion weights must sum to a positive value.")

    keyword_weight = keyword_weight / total
    semantic_weight = semantic_weight / total

    normalized_keyword = minmax_normalize(keyword_scores)
    normalized_semantic = minmax_normalize(semantic_scores)

    fused_scores = (
        keyword_weight * normalized_keyword
        + semantic_weight * normalized_semantic
    )

    return fused_scores, normalized_keyword, normalized_semantic


# ------------------------------------------------------------
# 5. RECIPROCAL RANK FUSION (RRF)
# ------------------------------------------------------------

def reciprocal_rank_fusion(
    keyword_scores,
    semantic_scores,
    constant=60,
):
    """
    Fuse rankings without requiring the raw score scales to match.

    Each ranking contributes 1 / (constant + rank).
    """
    if constant < 0:
        raise ValueError("RRF constant must be non-negative.")

    keyword_order = np.argsort(keyword_scores)[::-1]
    semantic_order = np.argsort(semantic_scores)[::-1]

    fused = np.zeros(len(keyword_scores), dtype=float)

    for rank, index in enumerate(keyword_order, start=1):
        fused[index] += 1.0 / (constant + rank)

    for rank, index in enumerate(semantic_order, start=1):
        fused[index] += 1.0 / (constant + rank)

    return fused


# ------------------------------------------------------------
# 6. LIGHTWEIGHT LEXICAL RERANKING
# ------------------------------------------------------------
# This is a transparent heuristic, NOT a neural cross-encoder.
# It rewards candidates containing query terms after retrieval.

def content_terms(text):
    terms = re.findall(r"\b[a-z0-9]+\b", str(text).lower())

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "do", "does", "did", "how", "what", "when",
        "where", "why", "can", "could", "would", "should",
        "to", "of", "in", "on", "at", "for", "and", "or",
        "we", "i", "you", "it", "this", "that", "after",
        "before", "with", "from", "be", "have", "has",
    }

    return set(
        term for term in terms
        if term not in stop_words and len(term) > 1
    )


def lexical_rerank_score(query, text):
    query_terms = content_terms(query)
    text_terms = content_terms(text)

    if not query_terms:
        return 0.0

    return len(query_terms & text_terms) / len(query_terms)


# ------------------------------------------------------------
# 7. UNIFIED HYBRID RETRIEVAL FUNCTION
# ------------------------------------------------------------

def hybrid_search(
    query,
    top_k=None,
    method="weighted",
    rerank=True,
):
    """
    method:
      - 'weighted': normalized weighted score fusion
      - 'rrf': Reciprocal Rank Fusion

    Reranking uses a small lexical-overlap adjustment.
    Returns ranked chunks and per-component scores.
    """
    query = clean_query(query)

    if top_k is None:
        top_k = PART3_CONFIG["top_k"]

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer.")

    start_time = time.perf_counter()

    keyword_scores, semantic_scores = get_similarity_scores(query)

    weighted_scores, normalized_keyword, normalized_semantic = (
        weighted_score_fusion(
            keyword_scores,
            semantic_scores,
            PART3_CONFIG["keyword_weight"],
            PART3_CONFIG["semantic_weight"],
        )
    )

    if method == "weighted":
        base_scores = weighted_scores
    elif method == "rrf":
        base_scores = reciprocal_rank_fusion(
            keyword_scores,
            semantic_scores,
            PART3_CONFIG["rrf_constant"],
        )
    else:
        raise ValueError("method must be 'weighted' or 'rrf'.")

    lexical_scores = np.array([
        lexical_rerank_score(query, text)
        for text in chunks["text"]
    ])

    if rerank:
        rerank_weight = PART3_CONFIG["lexical_rerank_weight"]
        final_scores = (
            (1 - rerank_weight) * minmax_normalize(base_scores)
            + rerank_weight * lexical_scores
        )
    else:
        final_scores = base_scores.copy()

    ranked_indices = np.argsort(final_scores)[::-1]

    results = []

    for index in ranked_indices:
        if final_scores[index] <= 0:
            continue

        chunk = chunks.iloc[int(index)]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "category": (
                chunk["category"]
                if "category" in chunks.columns else "Unknown"
            ),
            "text": chunk["text"],
            "keyword_score": round(float(keyword_scores[index]), 6),
            "semantic_score": round(float(semantic_scores[index]), 6),
            "normalized_keyword_score": round(
                float(normalized_keyword[index]), 6
            ),
            "normalized_semantic_score": round(
                float(normalized_semantic[index]), 6
            ),
            "weighted_score": round(float(weighted_scores[index]), 6),
            "fusion_score": round(float(base_scores[index]), 6),
            "lexical_rerank_score": round(
                float(lexical_scores[index]), 6
            ),
            "final_score": round(float(final_scores[index]), 6),
        })

        if len(results) >= top_k:
            break

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "query": query,
        "method": method,
        "reranking_enabled": rerank,
        "result_count": len(results),
        "latency_ms": round(latency_ms, 3),
        "results": results,
    }


# ------------------------------------------------------------
# 8. COMPARE KEYWORD, SEMANTIC, WEIGHTED AND RRF RETRIEVAL
# ------------------------------------------------------------

COMPARISON_QUERIES = [
    "How do we protect internal API endpoints?",
    "How do I recover a service after a failure?",
    "How can confidential customer information be protected?",
    "How can I find passages with the same meaning?",
    "What helps restore a database after data loss?",
    "How can we make an API respond faster?",
    "What must happen before software is released?",
]

comparison_rows = []
search_result_rows = []

for query in COMPARISON_QUERIES:
    keyword_scores, semantic_scores = get_similarity_scores(query)

    weighted_response = hybrid_search(
        query, top_k=3, method="weighted", rerank=True
    )

    rrf_response = hybrid_search(
        query, top_k=3, method="rrf", rerank=True
    )

    keyword_order = np.argsort(keyword_scores)[::-1]
    semantic_order = np.argsort(semantic_scores)[::-1]

    def document_ids_from_order(order, scores, k=3):
        ids = []
        for index in order:
            if scores[index] <= 0:
                continue
            doc_id = chunks.iloc[int(index)]["doc_id"]
            if doc_id not in ids:
                ids.append(doc_id)
            if len(ids) >= k:
                break
        return ids

    keyword_ids = document_ids_from_order(
        keyword_order, keyword_scores
    )
    semantic_ids = document_ids_from_order(
        semantic_order, semantic_scores
    )

    weighted_ids = list(dict.fromkeys(
        item["doc_id"] for item in weighted_response["results"]
    ))

    rrf_ids = list(dict.fromkeys(
        item["doc_id"] for item in rrf_response["results"]
    ))

    comparison_rows.append({
        "query": query,
        "keyword_top3": " | ".join(keyword_ids),
        "semantic_top3": " | ".join(semantic_ids),
        "weighted_hybrid_top3": " | ".join(weighted_ids),
        "rrf_hybrid_top3": " | ".join(rrf_ids),
        "weighted_latency_ms": weighted_response["latency_ms"],
        "rrf_latency_ms": rrf_response["latency_ms"],
    })

    for method_name, response in [
        ("weighted", weighted_response),
        ("rrf", rrf_response),
    ]:
        for rank, item in enumerate(response["results"], start=1):
            search_result_rows.append({
                "query": query,
                "method": method_name,
                "rank": rank,
                "chunk_id": item["chunk_id"],
                "doc_id": item["doc_id"],
                "title": item["title"],
                "keyword_score": item["keyword_score"],
                "semantic_score": item["semantic_score"],
                "fusion_score": item["fusion_score"],
                "final_score": item["final_score"],
                "latency_ms": response["latency_ms"],
            })

COMPARISON_DF = pd.DataFrame(comparison_rows)
HYBRID_SEARCH_RESULTS_DF = pd.DataFrame(search_result_rows)

print("\n" + "=" * 68)
print("RETRIEVAL METHOD COMPARISON")
print("=" * 68)
print(COMPARISON_DF.to_string(index=False))


# ------------------------------------------------------------
# 9. EVALUATE BASELINES AND HYBRID METHODS
# ------------------------------------------------------------

if "EVALUATION_DATASET" not in globals():
    EVALUATION_DATASET = [
        {
            "query": "How do we protect internal API endpoints?",
            "relevant_doc_ids": ["DOC001"],
        },
        {
            "query": "What steps should engineers follow during an outage?",
            "relevant_doc_ids": ["DOC002"],
        },
        {
            "query": "How can we restore a production database?",
            "relevant_doc_ids": ["DOC003"],
        },
        {
            "query": "How does semantic search use text embeddings?",
            "relevant_doc_ids": ["DOC008"],
        },
        {
            "query": "How can we measure RAG retrieval quality?",
            "relevant_doc_ids": ["DOC007"],
        },
        {
            "query": "How do we reduce API response latency?",
            "relevant_doc_ids": ["DOC009"],
        },
        {
            "query": "What checks are required before deploying software?",
            "relevant_doc_ids": ["DOC005", "DOC012"],
        },
    ]


def ranked_unique_doc_ids(indices, scores):
    ids = []

    for index in indices:
        if scores[index] <= 0:
            continue

        doc_id = chunks.iloc[int(index)]["doc_id"]

        if doc_id not in ids:
            ids.append(doc_id)

    return ids


def evaluate_ranked_ids(retrieved_ids, relevant_ids, k=3):
    top_ids = retrieved_ids[:k]

    precision = (
        sum(doc_id in relevant_ids for doc_id in top_ids)
        / len(top_ids)
        if top_ids else 0.0
    )

    recall = (
        len(set(top_ids) & set(relevant_ids)) / len(relevant_ids)
        if relevant_ids else 0.0
    )

    rr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            rr = 1.0 / rank
            break

    return precision, recall, rr


def evaluate_all_methods(dataset, k=3):
    rows = []

    for case in dataset:
        query = case["query"]
        relevant_ids = case["relevant_doc_ids"]

        keyword_scores, semantic_scores = get_similarity_scores(query)

        weighted_response = hybrid_search(
            query, top_k=len(chunks), method="weighted", rerank=True
        )

        rrf_response = hybrid_search(
            query, top_k=len(chunks), method="rrf", rerank=True
        )

        keyword_ids = ranked_unique_doc_ids(
            np.argsort(keyword_scores)[::-1], keyword_scores
        )

        semantic_ids = ranked_unique_doc_ids(
            np.argsort(semantic_scores)[::-1], semantic_scores
        )

        weighted_ids = list(dict.fromkeys(
            item["doc_id"] for item in weighted_response["results"]
        ))

        rrf_ids = list(dict.fromkeys(
            item["doc_id"] for item in rrf_response["results"]
        ))

        methods = {
            "keyword": keyword_ids,
            "latent_semantic": semantic_ids,
            "weighted_hybrid": weighted_ids,
            "rrf_hybrid": rrf_ids,
        }

        for method_name, retrieved_ids in methods.items():
            precision, recall, rr = evaluate_ranked_ids(
                retrieved_ids, relevant_ids, k
            )

            rows.append({
                "query": query,
                "method": method_name,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "reciprocal_rank": rr,
                "retrieved_doc_ids": ", ".join(retrieved_ids[:k]),
                "relevant_doc_ids": ", ".join(relevant_ids),
            })

    return pd.DataFrame(rows)


HYBRID_EVALUATION_DF = evaluate_all_methods(
    EVALUATION_DATASET,
    k=PART3_CONFIG["evaluation_k"],
)

METRICS_SUMMARY_DF = (
    HYBRID_EVALUATION_DF
    .groupby("method", as_index=False)
    .agg(
        precision_at_k=("precision_at_k", "mean"),
        recall_at_k=("recall_at_k", "mean"),
        mean_reciprocal_rank=("reciprocal_rank", "mean"),
    )
)

for metric in [
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
]:
    METRICS_SUMMARY_DF[metric] = (
        METRICS_SUMMARY_DF[metric].round(4)
    )

print("\n" + "=" * 68)
print("AGGREGATE EVALUATION RESULTS")
print("=" * 68)
print(METRICS_SUMMARY_DF.to_string(index=False))


# ------------------------------------------------------------
# 10. CONFIGURATION SENSITIVITY EXPERIMENT
# ------------------------------------------------------------
# Compare keyword weight settings. This is a small offline
# experiment, not a guarantee that any weight will generalize.

weight_experiment_rows = []

for keyword_weight in [0.25, 0.50, 0.75]:
    semantic_weight = 1.0 - keyword_weight
    query_precisions = []
    query_recalls = []
    query_rrs = []

    for case in EVALUATION_DATASET:
        query = clean_query(case["query"])
        relevant_ids = case["relevant_doc_ids"]

        keyword_scores, semantic_scores = get_similarity_scores(query)

        fused_scores, _, _ = weighted_score_fusion(
            keyword_scores,
            semantic_scores,
            keyword_weight,
            semantic_weight,
        )

        lexical_scores = np.array([
            lexical_rerank_score(query, text)
            for text in chunks["text"]
        ])

        final_scores = (
            (1 - PART3_CONFIG["lexical_rerank_weight"])
            * minmax_normalize(fused_scores)
            + PART3_CONFIG["lexical_rerank_weight"]
            * lexical_scores
        )

        order = np.argsort(final_scores)[::-1]
        retrieved_ids = ranked_unique_doc_ids(order, final_scores)

        precision, recall, rr = evaluate_ranked_ids(
            retrieved_ids,
            relevant_ids,
            PART3_CONFIG["evaluation_k"],
        )

        query_precisions.append(precision)
        query_recalls.append(recall)
        query_rrs.append(rr)

    weight_experiment_rows.append({
        "keyword_weight": keyword_weight,
        "semantic_weight": semantic_weight,
        "mean_precision_at_k": round(
            float(np.mean(query_precisions)), 4
        ),
        "mean_recall_at_k": round(
            float(np.mean(query_recalls)), 4
        ),
        "mean_reciprocal_rank": round(
            float(np.mean(query_rrs)), 4
        ),
    })

WEIGHT_EXPERIMENT_DF = pd.DataFrame(weight_experiment_rows)

print("\n" + "=" * 68)
print("WEIGHT SENSITIVITY EXPERIMENT")
print("=" * 68)
print(WEIGHT_EXPERIMENT_DF.to_string(index=False))


# ------------------------------------------------------------
# 11. SAVE ARTIFACTS
# ------------------------------------------------------------

COMPARISON_DF.to_csv(
    OUTPUT_DIR / "retrieval_method_comparison.csv",
    index=False,
)

HYBRID_SEARCH_RESULTS_DF.to_csv(
    OUTPUT_DIR / "hybrid_search_results.csv",
    index=False,
)

HYBRID_EVALUATION_DF.to_csv(
    OUTPUT_DIR / "hybrid_evaluation_by_query.csv",
    index=False,
)

METRICS_SUMMARY_DF.to_csv(
    OUTPUT_DIR / "hybrid_metrics_summary.csv",
    index=False,
)

WEIGHT_EXPERIMENT_DF.to_csv(
    OUTPUT_DIR / "fusion_weight_experiment.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "part3_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            **PART3_CONFIG,
            "semantic_method": semantic_method,
            "reranker": "lexical term overlap heuristic",
            "limitations": [
                "Not a neural cross-encoder reranker",
                "SVD is a corpus-dependent latent-semantic baseline",
                "Evaluation dataset is small and synthetic",
                "Score normalization is query-dependent",
            ],
        },
        file,
        indent=2,
    )


# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 68)
print("PART 3 EXECUTION SUMMARY")
print("=" * 68)
print(f"Chunks: {len(chunks)}")
print(f"Evaluation queries: {len(EVALUATION_DATASET)}")
print(f"Semantic method: {semantic_method}")
print("Fusion methods: weighted score fusion and RRF")
print("Reranking: lexical-overlap heuristic")
print(f"Artifacts: {OUTPUT_DIR.resolve()}")

print("\nUse the hybrid engine:")
print(
    "hybrid_search('How can we protect an API?', "
    "top_k=3, method='weighted')"
)

print("\nAvailable functions:")
print("  get_similarity_scores(query)")
print("  weighted_score_fusion(...)")
print("  reciprocal_rank_fusion(...)")
print("  hybrid_search(query, top_k=3, method='weighted')")
print("  evaluate_all_methods(dataset, k=3)")


# ============================================================
# DAY 69/100 — PART 4
# FastAPI + Automated Tests + Performance Checks + Scorecard
# CPU-friendly | Local prototype | No LLM required
# ============================================================

import json
import time
import statistics
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.testclient import TestClient

# ------------------------------------------------------------
# 1. CONFIGURATION AND PRE-FLIGHT VALIDATION
# ------------------------------------------------------------

PART4_CONFIG = {
    "project_name": "Day 69 - Hybrid Retrieval Engine",
    "version": "1.0.0",
    "output_dir": "day69_hybrid_retrieval/part4",
    "default_top_k": 3,
    "max_top_k": 10,
    "max_query_length": 1000,
    "latency_smoke_requests": 10,
}

OUTPUT_DIR = Path(PART4_CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

required_functions = [
    "retrieve_documents",
    "semantic_search",
    "hybrid_search",
]

for function_name in required_functions:
    if function_name not in globals():
        raise RuntimeError(
            f"Missing {function_name}. Run Parts 1, 2, and 3 first."
        )

if "CHUNKS_DF" not in globals() or CHUNKS_DF.empty:
    raise RuntimeError("CHUNKS_DF is missing or empty.")

print("=" * 68)
print("DAY 69 — PART 4: API, TESTS AND FINAL VALIDATION")
print("=" * 68)

# ------------------------------------------------------------
# 2. CREATE FASTAPI APPLICATION
# ------------------------------------------------------------

app = FastAPI(
    title="Hybrid Retrieval Engine API",
    description=(
        "Local search API supporting keyword, latent-semantic, "
        "weighted hybrid, and RRF retrieval."
    ),
    version=PART4_CONFIG["version"],
)

API_STATE = {
    "total_requests": 0,
    "endpoint_counts": Counter(),
    "errors": 0,
    "latencies_ms": [],
    "method_counts": Counter(),
}


def track_request(endpoint, latency_ms, failed=False, method=None):
    API_STATE["total_requests"] += 1
    API_STATE["endpoint_counts"][endpoint] += 1
    API_STATE["latencies_ms"].append(float(latency_ms))

    if failed:
        API_STATE["errors"] += 1

    if method:
        API_STATE["method_counts"][method] += 1


# ------------------------------------------------------------
# 3. DEFINE VALIDATED REQUEST MODELS
# ------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=PART4_CONFIG["max_query_length"],
    )
    top_k: int = Field(
        default=PART4_CONFIG["default_top_k"],
        ge=1,
        le=PART4_CONFIG["max_top_k"],
    )
    method: str = Field(
        default="weighted",
        description=(
            "One of keyword, semantic, weighted, or rrf."
        ),
    )
    rerank: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str
    chunks_indexed: int


# ------------------------------------------------------------
# 4. DEFINE SEARCH DISPATCHER
# ------------------------------------------------------------

def run_search(query, top_k=3, method="weighted", rerank=True):
    """Dispatch requests to the appropriate retrieval function."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")

    if len(query) > PART4_CONFIG["max_query_length"]:
        raise ValueError("query exceeds the maximum allowed length")

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer")

    if not 1 <= top_k <= PART4_CONFIG["max_top_k"]:
        raise ValueError(
            f"top_k must be between 1 and "
            f"{PART4_CONFIG['max_top_k']}"
        )

    method = method.lower().strip()

    if method == "keyword":
        response = retrieve_documents(query, top_k=top_k)

    elif method == "semantic":
        response = semantic_search(query, top_k=top_k)

    elif method == "weighted":
        response = hybrid_search(
            query,
            top_k=top_k,
            method="weighted",
            rerank=rerank,
        )

    elif method == "rrf":
        response = hybrid_search(
            query,
            top_k=top_k,
            method="rrf",
            rerank=rerank,
        )

    else:
        raise ValueError(
            "Unsupported method. Choose keyword, semantic, "
            "weighted, or rrf."
        )

    return {
        "query": query,
        "method": method,
        "top_k": top_k,
        "result_count": len(response.get("results", [])),
        "latency_ms": response.get("latency_ms", 0.0),
        "results": response.get("results", []),
    }


# ------------------------------------------------------------
# 5. API ENDPOINTS
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "project": PART4_CONFIG["project_name"],
        "version": PART4_CONFIG["version"],
        "message": "Hybrid Retrieval Engine API is ready.",
        "endpoints": [
            "GET /",
            "GET /health",
            "GET /info",
            "GET /metrics",
            "POST /search",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "version": PART4_CONFIG["version"],
        "chunks_indexed": int(len(CHUNKS_DF)),
    }


@app.get("/info")
def info():
    return {
        "project": PART4_CONFIG["project_name"],
        "version": PART4_CONFIG["version"],
        "chunks_indexed": int(len(CHUNKS_DF)),
        "supported_methods": [
            "keyword",
            "semantic",
            "weighted",
            "rrf",
        ],
        "semantic_representation": (
            globals().get("semantic_method", "SVD-based baseline")
        ),
        "reranker": "Lexical-overlap heuristic",
        "deployment": "local notebook prototype",
    }


@app.get("/metrics")
def metrics():
    latencies = API_STATE["latencies_ms"]

    return {
        "total_requests": API_STATE["total_requests"],
        "errors": API_STATE["errors"],
        "error_rate": (
            round(
                API_STATE["errors"]
                / API_STATE["total_requests"],
                4,
            )
            if API_STATE["total_requests"] else 0.0
        ),
        "endpoint_counts": dict(API_STATE["endpoint_counts"]),
        "method_counts": dict(API_STATE["method_counts"]),
        "average_latency_ms": (
            round(statistics.mean(latencies), 3)
            if latencies else 0.0
        ),
        "p95_latency_ms": (
            round(float(np.percentile(latencies, 95)), 3)
            if latencies else 0.0
        ),
    }


@app.post("/search")
def search(request: SearchRequest):
    start_time = time.perf_counter()

    try:
        response = run_search(
            query=request.query,
            top_k=request.top_k,
            method=request.method,
            rerank=request.rerank,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        track_request(
            endpoint="/search",
            latency_ms=elapsed_ms,
            method=request.method,
        )

        return response

    except ValueError as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        track_request(
            endpoint="/search",
            latency_ms=elapsed_ms,
            failed=True,
            method=request.method,
        )

        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        track_request(
            endpoint="/search",
            latency_ms=elapsed_ms,
            failed=True,
            method=request.method,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Internal search error: {type(exc).__name__}",
        )


# ------------------------------------------------------------
# 6. AUTOMATED API TESTS
# ------------------------------------------------------------

# TestClient runs the ASGI app in-process. This is an API-level
# test, not a test of a separately running production server.

client = TestClient(app)

test_rows = []


def run_api_test(test_name, request_callable, expected_status):
    start_time = time.perf_counter()

    try:
        response = request_callable()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        passed = response.status_code == expected_status

        try:
            response_body = response.json()
        except Exception:
            response_body = {"body": response.text[:200]}

        detail = (
            f"Expected HTTP {expected_status}; "
            f"received HTTP {response.status_code}"
        )

        test_rows.append({
            "test_name": test_name,
            "expected_status": expected_status,
            "actual_status": response.status_code,
            "passed": bool(passed),
            "latency_ms": round(elapsed_ms, 3),
            "detail": "PASS" if passed else detail,
        })

        return response

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        test_rows.append({
            "test_name": test_name,
            "expected_status": expected_status,
            "actual_status": None,
            "passed": False,
            "latency_ms": round(elapsed_ms, 3),
            "detail": f"{type(exc).__name__}: {exc}",
        })

        return None


run_api_test(
    "Root endpoint",
    lambda: client.get("/"),
    200,
)

run_api_test(
    "Health endpoint",
    lambda: client.get("/health"),
    200,
)

run_api_test(
    "Info endpoint",
    lambda: client.get("/info"),
    200,
)

run_api_test(
    "Metrics endpoint",
    lambda: client.get("/metrics"),
    200,
)

run_api_test(
    "Valid weighted search",
    lambda: client.post(
        "/search",
        json={
            "query": "How do we protect API endpoints?",
            "top_k": 3,
            "method": "weighted",
        },
    ),
    200,
)

run_api_test(
    "Keyword search",
    lambda: client.post(
        "/search",
        json={
            "query": "How can we reduce API latency?",
            "top_k": 3,
            "method": "keyword",
        },
    ),
    200,
)

run_api_test(
    "Semantic search",
    lambda: client.post(
        "/search",
        json={
            "query": "How can we restore a failed service?",
            "top_k": 3,
            "method": "semantic",
        },
    ),
    200,
)

run_api_test(
    "RRF search",
    lambda: client.post(
        "/search",
        json={
            "query": "How can we find similar text?",
            "top_k": 3,
            "method": "rrf",
        },
    ),
    200,
)

run_api_test(
    "Reject unsupported method",
    lambda: client.post(
        "/search",
        json={
            "query": "Find documents",
            "top_k": 3,
            "method": "unknown",
        },
    ),
    400,
)

run_api_test(
    "Reject top_k above maximum",
    lambda: client.post(
        "/search",
        json={
            "query": "Find documents",
            "top_k": 100,
            "method": "weighted",
        },
    ),
    422,
)

run_api_test(
    "Reject empty query",
    lambda: client.post(
        "/search",
        json={
            "query": "",
            "top_k": 3,
            "method": "weighted",
        },
    ),
    422,
)

run_api_test(
    "Reject excessively long query",
    lambda: client.post(
        "/search",
        json={
            "query": "x" * (
                PART4_CONFIG["max_query_length"] + 1
            ),
            "top_k": 3,
            "method": "weighted",
        },
    ),
    422,
)

API_TEST_DF = pd.DataFrame(test_rows)

print("\n" + "=" * 68)
print("AUTOMATED API TEST RESULTS")
print("=" * 68)
print(API_TEST_DF.to_string(index=False))


# ------------------------------------------------------------
# 7. SEQUENTIAL SMOKE TEST
# ------------------------------------------------------------

# Sends repeated requests in sequence. This is NOT a concurrent
# load test and should not be presented as one.

smoke_rows = []
smoke_query = "How do we protect internal API endpoints?"

for request_number in range(
    1, PART4_CONFIG["latency_smoke_requests"] + 1
):
    start_time = time.perf_counter()

    try:
        response = client.post(
            "/search",
            json={
                "query": smoke_query,
                "top_k": 3,
                "method": "weighted",
            },
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        smoke_rows.append({
            "request_number": request_number,
            "status_code": response.status_code,
            "passed": response.status_code == 200,
            "latency_ms": round(elapsed_ms, 3),
        })

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        smoke_rows.append({
            "request_number": request_number,
            "status_code": None,
            "passed": False,
            "latency_ms": round(elapsed_ms, 3),
            "error": f"{type(exc).__name__}: {exc}",
        })

SMOKE_TEST_DF = pd.DataFrame(smoke_rows)

smoke_latencies = SMOKE_TEST_DF["latency_ms"].tolist()

SMOKE_SUMMARY = {
    "requests_sent": len(SMOKE_TEST_DF),
    "requests_passed": int(SMOKE_TEST_DF["passed"].sum()),
    "requests_failed": int((~SMOKE_TEST_DF["passed"]).sum()),
    "average_latency_ms": round(
        float(np.mean(smoke_latencies)), 3
    ),
    "p50_latency_ms": round(
        float(np.percentile(smoke_latencies, 50)), 3
    ),
    "p95_latency_ms": round(
        float(np.percentile(smoke_latencies, 95)), 3
    ),
    "test_type": "sequential in-process smoke test",
}

print("\n" + "=" * 68)
print("SEQUENTIAL SMOKE TEST")
print("=" * 68)
print(SMOKE_TEST_DF.to_string(index=False))
print("\nSmoke summary:")
print(json.dumps(SMOKE_SUMMARY, indent=2))


# ------------------------------------------------------------
# 8. FINAL PROJECT SCORECARD
# ------------------------------------------------------------

total_tests = len(API_TEST_DF)
passed_tests = (
    int(API_TEST_DF["passed"].sum())
    if total_tests else 0
)

test_pass_rate = (
    round(passed_tests / total_tests, 4)
    if total_tests else 0.0
)

FINAL_SCORECARD = {
    "project": PART4_CONFIG["project_name"],
    "version": PART4_CONFIG["version"],
    "generated_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "documents": int(
        len(DOCUMENTS_DF)
        if "DOCUMENTS_DF" in globals()
        else chunks["doc_id"].nunique()
    ),
    "chunks_indexed": int(len(CHUNKS_DF)),
    "api_endpoints": [
        "GET /",
        "GET /health",
        "GET /info",
        "GET /metrics",
        "POST /search",
    ],
    "retrieval_methods": [
        "TF-IDF keyword retrieval",
        "TF-IDF + SVD latent-semantic retrieval",
        "Weighted hybrid score fusion",
        "Reciprocal Rank Fusion",
    ],
    "reranker": "Lexical overlap heuristic",
    "api_tests_total": total_tests,
    "api_tests_passed": passed_tests,
    "api_tests_failed": total_tests - passed_tests,
    "api_test_pass_rate": test_pass_rate,
    "smoke_test": SMOKE_SUMMARY,
    "deployment_status": "LOCAL_NOTEBOOK_PROTOTYPE",
    "limitations": [
        "Synthetic enterprise corpus",
        "Latent-semantic SVD is not a pretrained embedding model",
        "Lexical reranker is not a neural cross-encoder",
        "No persistent vector database",
        "No authentication or authorization",
        "No cloud deployment",
        "Smoke test is sequential, not concurrent",
        "Evaluation metrics are limited by the small labeled dataset",
    ],
}

print("\n" + "=" * 68)
print("FINAL PROJECT SCORECARD")
print("=" * 68)
print(json.dumps(FINAL_SCORECARD, indent=2))


# ------------------------------------------------------------
# 9. EXPORT REPORTS AND PROJECT DOCUMENTATION
# ------------------------------------------------------------

API_TEST_DF.to_csv(
    OUTPUT_DIR / "api_test_results.csv",
    index=False,
)

SMOKE_TEST_DF.to_csv(
    OUTPUT_DIR / "api_smoke_test.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "api_smoke_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(SMOKE_SUMMARY, file, indent=2)

with open(
    OUTPUT_DIR / "final_project_scorecard.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(FINAL_SCORECARD, file, indent=2)

with open(
    OUTPUT_DIR / "api_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(PART4_CONFIG, file, indent=2)

architecture = """
DAY 69 — HYBRID RETRIEVAL ENGINE

Client
  |
  v
FastAPI request validation (Pydantic)
  |
  v
Search dispatcher
  |-- Keyword: TF-IDF cosine similarity
  |-- Semantic: TF-IDF + Truncated SVD
  |-- Weighted: normalized score fusion
  |-- RRF: reciprocal rank fusion
  |
  v
Ranked document chunks
  |
  v
API response and in-memory metrics

Testing:
  - API endpoint tests
  - Request validation tests
  - Sequential smoke test
  - Retrieval evaluation from Part 3

Limitations:
  - Synthetic data
  - No pretrained neural embeddings
  - No persistent vector database
  - No authentication
  - No cloud deployment
"""

with open(
    OUTPUT_DIR / "architecture.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(architecture.strip() + "\n")

requirements = """numpy
pandas
scikit-learn
fastapi
pydantic
uvicorn
httpx
"""

with open(
    OUTPUT_DIR / "requirements.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(requirements)

run_instructions = """
DAY 69 — LOCAL API RUN INSTRUCTIONS

1. Run Parts 1, 2, 3 and 4 in the same notebook session.
2. Save the FastAPI app definition in a Python file named app.py.
3. Ensure the file builds or imports the document indexes and
   search functions before exposing the API.
4. Install dependencies from requirements.txt.
5. Start the server with:

   uvicorn app:app --host 127.0.0.1 --port 8000

6. Open the local API documentation at:

   http://127.0.0.1:8000/docs

Security:
This is a local prototype. Do not expose it publicly without
authentication, rate limiting, proper error handling, and deployment
hardening.

Note:
The notebook's in-memory API_STATE is not persistent and will reset
when the process restarts.
"""

with open(
    OUTPUT_DIR / "RUN_INSTRUCTIONS.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(run_instructions.strip() + "\n")

# Export a final validation checklist.
validation_rows = [
    {
        "check": "CHUNKS_DF available",
        "passed": "CHUNKS_DF" in globals()
        and not CHUNKS_DF.empty,
    },
    {
        "check": "Keyword search callable",
        "passed": callable(retrieve_documents),
    },
    {
        "check": "Semantic search callable",
        "passed": callable(semantic_search),
    },
    {
        "check": "Hybrid search callable",
        "passed": callable(hybrid_search),
    },
    {
        "check": "API tests all passed",
        "passed": total_tests > 0 and passed_tests == total_tests,
    },
    {
        "check": "Smoke test all passed",
        "passed": (
            len(SMOKE_TEST_DF) > 0
            and bool(SMOKE_TEST_DF["passed"].all())
        ),
    },
]

FINAL_VALIDATION_DF = pd.DataFrame(validation_rows)

FINAL_VALIDATION_DF.to_csv(
    OUTPUT_DIR / "final_validation.csv",
    index=False,
)

print("\n" + "=" * 68)
print("FINAL VALIDATION CHECKLIST")
print("=" * 68)
print(FINAL_VALIDATION_DF.to_string(index=False))

print("\nReports saved to:")
print(OUTPUT_DIR.resolve())

print("\nImportant:")
print("- The test outcomes above come from this actual execution.")
print("- Passing notebook tests do not prove production readiness.")
print("- The API object is available as `app` in this notebook.")
print("- Use a standalone app.py for a separate Uvicorn process.")

