# ============================================================
# DAY 53/100 — LANGCHAIN VECTOR DATABASE
# PART 1 : DOCUMENT CREATION + TEXT CHUNKING
#
# CPU FRIENDLY
# VERY LOW STORAGE
# NO EXTERNAL DATASET
# ============================================================


# ============================================================
# SECTION 1 : IMPORTS
# ============================================================

from langchain_core.documents import Document

print("Libraries Imported Successfully")


# ============================================================
# SECTION 2 : CREATE SMALL PREDEFINED DATASET
# ============================================================

documents = [

    Document(
        page_content="""
        Artificial Intelligence is the field of building
        computer systems that can perform tasks that normally
        require human intelligence. AI is used in automation,
        natural language processing, computer vision,
        recommendation systems, and robotics.
        """,
        metadata={
            "topic": "Artificial Intelligence"
        }
    ),

    Document(
        page_content="""
        Machine Learning is a branch of Artificial Intelligence.
        Machine Learning algorithms learn patterns from data
        and use those patterns to make predictions or decisions.
        Common approaches include supervised learning,
        unsupervised learning, and reinforcement learning.
        """,
        metadata={
            "topic": "Machine Learning"
        }
    ),

    Document(
        page_content="""
        Deep Learning is a subset of Machine Learning that uses
        neural networks with multiple layers. Deep Learning is
        widely used for image recognition, speech recognition,
        natural language processing, and generative AI.
        """,
        metadata={
            "topic": "Deep Learning"
        }
    ),

    Document(
        page_content="""
        Retrieval-Augmented Generation, also called RAG,
        combines information retrieval with language generation.
        A RAG system retrieves relevant documents from a knowledge
        base and provides the retrieved information to a language
        model to generate a grounded response.
        """,
        metadata={
            "topic": "RAG"
        }
    ),

    Document(
        page_content="""
        Vector databases store vector representations of data.
        They allow applications to perform similarity searches.
        Vector databases are commonly used in RAG systems,
        semantic search, recommendation systems, and AI
        knowledge bases.
        """,
        metadata={
            "topic": "Vector Database"
        }
    ),

    Document(
        page_content="""
        LangChain is a framework for developing applications
        powered by language models. It provides components for
        prompts, document processing, retrieval, vector stores,
        chains, agents, and other LLM application workflows.
        """,
        metadata={
            "topic": "LangChain"
        }
    )

]


# ============================================================
# SECTION 3 : CHECK DATASET
# ============================================================

print()
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(
    "Number of documents:",
    len(documents)
)


# ============================================================
# SECTION 4 : DISPLAY DOCUMENTS
# ============================================================

print()
print("=" * 60)
print("DOCUMENTS")
print("=" * 60)


for index, document in enumerate(
    documents
):

    print()
    print(
        "Document:",
        index + 1
    )

    print(
        "Topic:",
        document.metadata["topic"]
    )

    print(
        "Content:",
        document.page_content.strip()
    )


# ============================================================
# SECTION 5 : TEXT CLEANING
# ============================================================

def clean_document_text(text):

    # Remove unnecessary whitespace

    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# SECTION 6 : CLEAN DOCUMENTS
# ============================================================

cleaned_documents = []


for document in documents:

    cleaned_text = clean_document_text(
        document.page_content
    )


    cleaned_document = Document(

        page_content=cleaned_text,

        metadata=document.metadata

    )


    cleaned_documents.append(
        cleaned_document
    )


print()
print("=" * 60)
print("TEXT CLEANING COMPLETED")
print("=" * 60)


# ============================================================
# SECTION 7 : CHECK CLEANED DOCUMENT
# ============================================================

for document in cleaned_documents:

    print()
    print(
        "Topic:",
        document.metadata["topic"]
    )

    print(
        document.page_content
    )


# ============================================================
# SECTION 8 : SIMPLE TEXT CHUNKING
# ============================================================

CHUNK_SIZE = 120

CHUNK_OVERLAP = 20


def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []


    start = 0


    while start < len(text):

        end = start + chunk_size


        chunk = text[
            start:end
        ].strip()


        if chunk:

            chunks.append(
                chunk
            )


        start = end - overlap


        if start < 0:

            start = 0


    return chunks


# ============================================================
# SECTION 9 : CREATE CHUNKS
# ============================================================

chunked_documents = []


for document in cleaned_documents:

    chunks = split_text(
        document.page_content
    )


    for chunk_index, chunk in enumerate(
        chunks
    ):

        chunked_documents.append(

            Document(

                page_content=chunk,

                metadata={

                    "topic":
                    document.metadata["topic"],

                    "chunk":
                    chunk_index + 1

                }

            )

        )


# ============================================================
# SECTION 10 : DISPLAY CHUNKS
# ============================================================

print()
print("=" * 60)
print("DOCUMENT CHUNKS")
print("=" * 60)


for index, document in enumerate(
    chunked_documents
):

    print()

    print(
        "Chunk:",
        index + 1
    )

    print(
        "Topic:",
        document.metadata["topic"]
    )

    print(
        "Text:",
        document.page_content
    )


# ============================================================
# SECTION 11 : CHUNK STATISTICS
# ============================================================

print()
print("=" * 60)
print("CHUNK STATISTICS")
print("=" * 60)


print(
    "Original Documents:",
    len(documents)
)


print(
    "Cleaned Documents:",
    len(cleaned_documents)
)


print(
    "Total Chunks:",
    len(chunked_documents)
)


print(
    "Chunk Size:",
    CHUNK_SIZE
)


print(
    "Chunk Overlap:",
    CHUNK_OVERLAP
)


# ============================================================
# SECTION 12 : VERIFY METADATA
# ============================================================

print()
print("=" * 60)
print("METADATA VERIFICATION")
print("=" * 60)


for document in chunked_documents:

    print(
        document.metadata
    )


# ============================================================
# SECTION 13 : PART 1 SUMMARY
# ============================================================

print()
print("=" * 60)
print("DAY 53 — PART 1 COMPLETED")
print("=" * 60)


print("""
✔ Created a predefined AI knowledge base
✔ Used LangChain Document objects
✔ Added document metadata
✔ Cleaned document text
✔ Split documents into chunks
✔ Added chunk metadata
✔ No external dataset downloaded
✔ No large files created
✔ CPU-friendly
✔ Minimal storage usage

NEXT:
PART 2 — EMBEDDINGS + VECTOR DATABASE
""")
# ============================================================
# DAY 53/100 — LANGCHAIN VECTOR DATABASE
# PART 2 : EMBEDDINGS + VECTOR DATABASE
#
# CONTINUES FROM PART-1
#
# HARDWARE:
# - SLOW CPU
# - VERY LIMITED STORAGE
# - NO GPU
# ============================================================


# ============================================================
# SECTION 1 : IMPORTS
# ============================================================

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


print("Libraries Imported Successfully")


# ============================================================
# SECTION 2 : EXTRACT TEXT FROM DOCUMENTS
# ============================================================

texts = [

    document.page_content

    for document in chunked_documents

]


print()
print("=" * 60)
print("TEXT EXTRACTION")
print("=" * 60)

print(
    "Number of chunks:",
    len(texts)
)


# ============================================================
# SECTION 3 : CREATE EMBEDDING MODEL
# ============================================================

vectorizer = TfidfVectorizer(

    lowercase=True,

    stop_words="english"

)


print()
print("=" * 60)
print("CREATING EMBEDDING MODEL")
print("=" * 60)


# ============================================================
# SECTION 4 : CREATE VECTORS
# ============================================================

document_vectors = vectorizer.fit_transform(
    texts
)


print()
print("=" * 60)
print("EMBEDDINGS CREATED")
print("=" * 60)

print(
    "Number of documents:",
    document_vectors.shape[0]
)

print(
    "Vector dimensions:",
    document_vectors.shape[1]
)


# ============================================================
# SECTION 5 : CONVERT TO NUMPY
# ============================================================

document_vectors = document_vectors.toarray()


print()
print(
    "Vector matrix shape:",
    document_vectors.shape
)


# ============================================================
# SECTION 6 : SIMPLE VECTOR DATABASE
# ============================================================

class SimpleVectorDatabase:

    def __init__(
        self,
        documents,
        vectors
    ):

        self.documents = documents

        self.vectors = vectors


    # --------------------------------------------------------
    # SIMILARITY SEARCH
    # --------------------------------------------------------

    def similarity_search(
        self,
        query,
        k=3
    ):

        # Convert query into vector

        query_vector = vectorizer.transform(
            [query]
        ).toarray()


        # Calculate cosine similarity

        similarities = cosine_similarity(

            query_vector,

            self.vectors

        )[0]


        # Get highest similarity indexes

        top_indexes = np.argsort(
            similarities
        )[::-1][:k]


        results = []


        for index in top_indexes:

            results.append({

                "document":
                self.documents[index],

                "score":
                similarities[index]

            })


        return results


# ============================================================
# SECTION 7 : CREATE VECTOR DATABASE
# ============================================================

vector_db = SimpleVectorDatabase(

    documents=chunked_documents,

    vectors=document_vectors

)


print()
print("=" * 60)
print("VECTOR DATABASE CREATED")
print("=" * 60)

print(
    "Stored vectors:",
    len(vector_db.vectors)
)


# ============================================================
# SECTION 8 : TEST VECTOR SEARCH
# ============================================================

query = "What is machine learning?"


results = vector_db.similarity_search(
    query,
    k=3
)


print()
print("=" * 60)
print("VECTOR SEARCH TEST")
print("=" * 60)

print()
print("Query:")
print(query)


# ============================================================
# SECTION 9 : DISPLAY SEARCH RESULTS
# ============================================================

for index, result in enumerate(
    results
):

    document = result["document"]

    score = result["score"]


    print()
    print(
        "Result:",
        index + 1
    )


    print(
        "Topic:",
        document.metadata["topic"]
    )


    print(
        "Similarity Score:",
        round(score, 4)
    )


    print(
        "Content:"
    )

    print(
        document.page_content
    )


# ============================================================
# SECTION 10 : TEST MULTIPLE QUERIES
# ============================================================

test_queries = [

    "What is artificial intelligence?",

    "Explain deep learning",

    "What is RAG?",

    "What is a vector database?",

    "What is LangChain?"

]


print()
print("=" * 60)
print("MULTIPLE VECTOR SEARCH TEST")
print("=" * 60)


for query in test_queries:

    results = vector_db.similarity_search(
        query,
        k=2
    )


    print()
    print("-" * 60)

    print(
        "QUERY:",
        query
    )


    for index, result in enumerate(
        results
    ):

        document = result["document"]

        score = result["score"]


        print()
        print(
            f"Result {index + 1}:"
        )

        print(
            "Topic:",
            document.metadata["topic"]
        )

        print(
            "Score:",
            round(score, 4)
        )


# ============================================================
# SECTION 11 : VECTOR DATABASE STATISTICS
# ============================================================

print()
print("=" * 60)
print("VECTOR DATABASE STATISTICS")
print("=" * 60)


print(
    "Documents:",
    len(vector_db.documents)
)


print(
    "Vector dimensions:",
    vector_db.vectors.shape[1]
)


print(
    "Vector matrix:",
    vector_db.vectors.shape
)


# ============================================================
# SECTION 12 : INSPECT ONE VECTOR
# ============================================================

print()
print("=" * 60)
print("SAMPLE VECTOR")
print("=" * 60)


sample_vector = vector_db.vectors[0]


print(
    "Vector length:",
    len(sample_vector)
)


print(
    "First 20 values:"
)


print(
    sample_vector[:20]
)


# ============================================================
# SECTION 13 : PART 2 SUMMARY
# ============================================================

print()
print("=" * 60)
print("DAY 53 — PART 2 COMPLETED")
print("=" * 60)


print("""
✔ Converted documents into numerical vectors
✔ Created lightweight TF-IDF embeddings
✔ Created a vector database
✔ Implemented cosine similarity search
✔ Tested semantic-style retrieval
✔ Retrieved top-k documents
✔ Displayed similarity scores
✔ No external embedding model downloaded
✔ No large dataset
✔ No GPU required
✔ Very low storage usage
✔ CPU-friendly

NEXT:
PART 3 — LANGCHAIN RETRIEVER + QUERY PIPELINE
""")
# ============================================================
# DAY 53/100 — LANGCHAIN VECTOR DATABASE
# PART 3 : LANGCHAIN RETRIEVER + QUERY PIPELINE
#
# CONTINUES FROM PART-1 AND PART-2
# ============================================================


# ============================================================
# SECTION 1 : IMPORTS
# ============================================================

from langchain_core.documents import Document

print("LangChain Components Imported Successfully")


# ============================================================
# SECTION 2 : CREATE LANGCHAIN RETRIEVER
# ============================================================

class LangChainVectorRetriever:

    def __init__(
        self,
        vector_database
    ):

        self.vector_database = vector_database


    # --------------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------------------------

    def retrieve(
        self,
        query,
        k=3
    ):

        results = self.vector_database.similarity_search(
            query,
            k=k
        )


        documents = []


        for result in results:

            document = result["document"]

            score = result["score"]


            # Store similarity score in metadata

            document_copy = Document(

                page_content=document.page_content,

                metadata={

                    **document.metadata,

                    "similarity_score":
                    float(score)

                }

            )


            documents.append(
                document_copy
            )


        return documents


# ============================================================
# SECTION 3 : CREATE RETRIEVER
# ============================================================

retriever = LangChainVectorRetriever(
    vector_db
)


print()
print("=" * 60)
print("LANGCHAIN RETRIEVER CREATED")
print("=" * 60)


# ============================================================
# SECTION 4 : TEST RETRIEVER
# ============================================================

query = "What is machine learning?"


retrieved_documents = retriever.retrieve(
    query,
    k=3
)


print()
print("=" * 60)
print("RETRIEVER TEST")
print("=" * 60)


print()
print("Query:")
print(query)


# ============================================================
# SECTION 5 : DISPLAY RETRIEVED DOCUMENTS
# ============================================================

for index, document in enumerate(
    retrieved_documents
):

    print()
    print(
        "Retrieved Document:",
        index + 1
    )


    print(
        "Topic:",
        document.metadata["topic"]
    )


    print(
        "Similarity Score:",
        round(
            document.metadata["similarity_score"],
            4
        )
    )


    print(
        "Content:"
    )

    print(
        document.page_content
    )


# ============================================================
# SECTION 6 : CREATE CONTEXT FROM DOCUMENTS
# ============================================================

def create_context(
    documents
):

    context_parts = []


    for document in documents:

        topic = document.metadata[
            "topic"
        ]

        content = document.page_content


        context_parts.append(

            f"Topic: {topic}\n"
            f"Information: {content}"

        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# SECTION 7 : TEST CONTEXT CREATION
# ============================================================

context = create_context(
    retrieved_documents
)


print()
print("=" * 60)
print("RETRIEVED CONTEXT")
print("=" * 60)


print(context)


# ============================================================
# SECTION 8 : BUILD RAG-STYLE PROMPT
# ============================================================

def build_rag_prompt(
    query,
    context
):

    prompt = f"""
You are an AI assistant.

Answer the user's question using the
provided context.

If the context does not contain enough
information, clearly say that the
information is not available.

Context:
{context}

User Question:
{query}

Answer:
"""


    return prompt.strip()


# ============================================================
# SECTION 9 : CREATE RAG PROMPT
# ============================================================

rag_prompt = build_rag_prompt(
    query,
    context
)


print()
print("=" * 60)
print("RAG-STYLE PROMPT")
print("=" * 60)


print(rag_prompt)


# ============================================================
# SECTION 10 : COMPLETE RETRIEVAL PIPELINE
# ============================================================

def retrieve_context(
    query,
    k=3
):

    documents = retriever.retrieve(
        query,
        k=k
    )


    context = create_context(
        documents
    )


    return documents, context


# ============================================================
# SECTION 11 : TEST COMPLETE RETRIEVAL
# ============================================================

test_queries = [

    "What is artificial intelligence?",

    "Explain deep learning",

    "What is RAG?",

    "What is a vector database?",

    "What is LangChain?"

]


print()
print("=" * 60)
print("COMPLETE RETRIEVAL PIPELINE TEST")
print("=" * 60)


for query in test_queries:

    documents, context = retrieve_context(
        query,
        k=2
    )


    print()
    print("-" * 60)

    print(
        "USER QUERY:"
    )

    print(query)


    print()
    print(
        "RETRIEVED TOPICS:"
    )


    for document in documents:

        print(
            "-",
            document.metadata["topic"],
            "| Score:",
            round(
                document.metadata[
                    "similarity_score"
                ],
                4
            )
        )


    print()
    print(
        "CONTEXT:"
    )

    print(context)


# ============================================================
# SECTION 12 : RETRIEVAL QUALITY CHECK
# ============================================================

print()
print("=" * 60)
print("RETRIEVAL QUALITY CHECK")
print("=" * 60)


quality_queries = {

    "machine learning": "What is machine learning?",

    "deep learning": "What is deep learning?",

    "rag": "What is RAG?",

    "vector database": "What is a vector database?",

    "langchain": "What is LangChain?"

}


correct = 0

total = len(
    quality_queries
)


for expected_topic, query in quality_queries.items():

    documents, context = retrieve_context(
        query,
        k=1
    )


    retrieved_topic = documents[0].metadata[
        "topic"
    ].lower()


    if expected_topic in retrieved_topic:

        correct += 1


    print()
    print(
        "Query:",
        query
    )


    print(
        "Retrieved:",
        retrieved_topic
    )


print()
print(
    "Correct Top-1 Retrieval:",
    correct,
    "/",
    total
)


print(
    "Retrieval Accuracy:",
    round(
        (correct / total) * 100,
        2
    ),
    "%"
)


# ============================================================
# SECTION 13 : PART 3 SUMMARY
# ============================================================

print()
print("=" * 60)
print("DAY 53 — PART 3 COMPLETED")
print("=" * 60)


print("""
✔ LangChain Document objects used
✔ Retriever layer created
✔ Vector database connected
✔ Similarity search integrated
✔ Top-K document retrieval implemented
✔ Retrieved context created
✔ RAG-style prompt created
✔ Multiple queries tested
✔ Retrieval quality checked
✔ No large model downloaded
✔ No external dataset
✔ No GPU required
✔ Minimal storage usage
✔ CPU-friendly

NEXT:
PART 4 — COMPLETE LANGCHAIN VECTOR DB + RAG APPLICATION
""")
# ============================================================
# DAY 53/100 — LANGCHAIN VECTOR DATABASE
# PART 4 : COMPLETE VECTOR DB + RAG APPLICATION
#
# CONTINUES FROM PART-1, PART-2 AND PART-3
#
# CPU FRIENDLY
# LOW STORAGE
# NO EXTERNAL DATASET
# NO LARGE MODEL
# ============================================================


# ============================================================
# SECTION 1 : COMPLETE QUERY PIPELINE
# ============================================================

def answer_query(
    query,
    k=2
):

    # --------------------------------------------------------
    # STEP 1 : RETRIEVE DOCUMENTS
    # --------------------------------------------------------

    documents, context = retrieve_context(
        query,
        k=k
    )


    # --------------------------------------------------------
    # STEP 2 : CHECK WHETHER INFORMATION EXISTS
    # --------------------------------------------------------

    if len(documents) == 0:

        return {
            "query": query,
            "answer": "I could not find relevant information.",
            "documents": [],
            "context": ""
        }


    # --------------------------------------------------------
    # STEP 3 : GET BEST DOCUMENT
    # --------------------------------------------------------

    best_document = documents[0]


    best_score = best_document.metadata[
        "similarity_score"
    ]


    # --------------------------------------------------------
    # STEP 4 : GENERATE LIGHTWEIGHT ANSWER
    #
    # We use the retrieved knowledge base directly
    # instead of loading a large LLM.
    # --------------------------------------------------------

    answer = best_document.page_content


    # --------------------------------------------------------
    # STEP 5 : RETURN COMPLETE RESULT
    # --------------------------------------------------------

    return {

        "query": query,

        "answer": answer,

        "documents": documents,

        "context": context,

        "best_score": best_score

    }


# ============================================================
# SECTION 2 : TEST COMPLETE PIPELINE
# ============================================================

print()
print("=" * 60)
print("COMPLETE VECTOR DB + RAG PIPELINE")
print("=" * 60)


query = "What is retrieval augmented generation?"


result = answer_query(
    query,
    k=2
)


print()
print("USER QUERY:")
print(
    result["query"]
)


print()
print("GENERATED ANSWER:")
print(
    result["answer"]
)


print()
print("BEST SIMILARITY SCORE:")
print(
    round(
        result["best_score"],
        4
    )
)


# ============================================================
# SECTION 3 : SHOW SOURCES
# ============================================================

print()
print("=" * 60)
print("RETRIEVED SOURCES")
print("=" * 60)


for index, document in enumerate(
    result["documents"]
):

    print()

    print(
        "Source:",
        index + 1
    )

    print(
        "Topic:",
        document.metadata["topic"]
    )

    print(
        "Similarity:",
        round(
            document.metadata[
                "similarity_score"
            ],
            4
        )
    )


# ============================================================
# SECTION 4 : MULTIPLE QUESTIONS
# ============================================================

questions = [

    "What is artificial intelligence?",

    "What is machine learning?",

    "Explain deep learning.",

    "What is RAG?",

    "What is a vector database?",

    "What is LangChain?",

    "What is a transformer?"

]


print()
print("=" * 60)
print("MULTI-QUERY TEST")
print("=" * 60)


for question in questions:

    result = answer_query(
        question,
        k=2
    )


    print()
    print("-" * 60)

    print(
        "QUESTION:"
    )

    print(question)


    print()
    print(
        "ANSWER:"
    )

    print(
        result["answer"]
    )


    print()
    print(
        "SOURCE:"
    )

    for document in result["documents"]:

        print(
            "-",
            document.metadata["topic"]
        )


# ============================================================
# SECTION 5 : INTERACTIVE RAG CHATBOT
# ============================================================

def run_vector_chatbot():

    print()
    print("=" * 60)
    print("      DAY 53 — VECTOR DATABASE CHATBOT")
    print("=" * 60)


    print()
    print(
        "Ask questions about the AI knowledge base."
    )


    print()
    print("Commands:")

    print(
        "  sources → Show retrieved sources"
    )

    print(
        "  stats   → Show vector database statistics"
    )

    print(
        "  exit    → Exit chatbot"
    )


    print()


    while True:

        query = input(
            "You: "
        )


        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if query.lower().strip() == "exit":

            print()
            print(
                "Vector chatbot stopped."
            )

            break


        # ----------------------------------------------------
        # EMPTY QUERY
        # ----------------------------------------------------

        if not query.strip():

            print(
                "Please enter a question."
            )

            continue


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        if query.lower().strip() == "stats":

            print()
            print(
                "Documents:",
                len(vector_db.documents)
            )

            print(
                "Vector Dimensions:",
                vector_db.vectors.shape[1]
            )

            print(
                "Stored Vectors:",
                len(vector_db.vectors)
            )

            continue


        # ----------------------------------------------------
        # RETRIEVE ANSWER
        # ----------------------------------------------------

        result = answer_query(
            query,
            k=2
        )


        print()
        print(
            "MiniRAG:"
        )

        print(
            result["answer"]
        )


        # ----------------------------------------------------
        # SHOW SOURCES
        # ----------------------------------------------------

        print()

        print(
            "Retrieved Sources:"
        )


        for document in result["documents"]:

            print(
                "-",
                document.metadata["topic"],
                "| Score:",
                round(
                    document.metadata[
                        "similarity_score"
                    ],
                    4
                )
            )


        print()


# ============================================================
# SECTION 6 : VECTOR DATABASE INFORMATION
# ============================================================

print()
print("=" * 60)
print("VECTOR DATABASE INFORMATION")
print("=" * 60)


print(
    "Documents stored:",
    len(vector_db.documents)
)


print(
    "Vectors stored:",
    len(vector_db.vectors)
)


print(
    "Vector dimensions:",
    vector_db.vectors.shape[1]
)


print(
    "Chunk size:",
    CHUNK_SIZE
)


print(
    "Chunk overlap:",
    CHUNK_OVERLAP
)


# ============================================================
# SECTION 7 : FINAL ARCHITECTURE
# ============================================================

print()
print("=" * 60)
print("FINAL PROJECT ARCHITECTURE")
print("=" * 60)


print("""
                 USER
                   |
                   v
              USER QUERY
                   |
                   v
          QUERY EMBEDDING
                   |
                   v
        +--------------------+
        |   VECTOR DATABASE  |
        +--------------------+
                   |
                   v
          SIMILARITY SEARCH
                   |
                   v
             TOP-K DOCS
                   |
                   v
          CONTEXT GENERATION
                   |
                   v
          RESPONSE GENERATOR
                   |
                   v
                ANSWER
""")


# ============================================================
# SECTION 8 : PROJECT SUMMARY
# ============================================================

print()
print("=" * 60)
print("DAY 53/100 — PROJECT COMPLETED")
print("=" * 60)


print("""
PROJECT:
LangChain Vector Database

PART 1:
✔ Predefined AI knowledge base
✔ LangChain Documents
✔ Text cleaning
✔ Text chunking
✔ Metadata

PART 2:
✔ Lightweight TF-IDF embeddings
✔ Vector representations
✔ Vector database
✔ Cosine similarity
✔ Top-K search

PART 3:
✔ LangChain-style retriever
✔ Query processing
✔ Context creation
✔ RAG-style prompt

PART 4:
✔ Complete retrieval pipeline
✔ Answer generation
✔ Source tracking
✔ Multi-query testing
✔ Interactive chatbot
✔ Vector database statistics

HARDWARE OPTIMIZATION:
✔ No external dataset
✔ No large embedding model
✔ No LLM download
✔ No GPU
✔ CPU friendly
✔ Very low storage requirement

DAY 53/100 COMPLETE ✔
""")


# ============================================================
# SECTION 9 : START INTERACTIVE CHATBOT
# ============================================================

# Uncomment the line below to start chatting.

# run_vector_chatbot()
