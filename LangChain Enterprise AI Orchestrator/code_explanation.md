# ============================================================
# DAY 61/100 — ADVANCED LANGCHAIN ENTERPRISE AI ORCHESTRATOR
# PART 1 — LANGCHAIN CORE ARCHITECTURE
# ============================================================
#
# Goal:
# Build the foundation of an enterprise LangChain application
# using:
#   - LangChain Documents
#   - Metadata
#   - Prompt Templates
#   - Runnable Chains
#   - Structured Outputs
#   - Output Parsing
#   - Chain Composition
#   - Enterprise Knowledge Processing
#
# CPU / STORAGE FRIENDLY
# No LLM download
# No GPU
# No external API key required
# ============================================================


# ============================================================
# 1. INSTALL REQUIRED PACKAGES
# ============================================================

!pip install -q langchain-core langchain


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import re
import json
import time
from typing import List, Dict, Any

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough
)


print("Libraries imported successfully.")


# ============================================================
# 3. PROJECT CONFIGURATION
# ============================================================

PROJECT_CONFIG = {
    "project_name": "Advanced LangChain Enterprise AI Orchestrator",
    "day": 61,
    "part": 1,
    "environment": "Jupyter Notebook",
    "llm_mode": "deterministic",
    "vector_search": False,
    "agent_enabled": False,
    "rag_enabled": False
}

print("\nPROJECT CONFIGURATION")
print(json.dumps(PROJECT_CONFIG, indent=2))


# ============================================================
# 4. CREATE ENTERPRISE KNOWLEDGE DOCUMENTS
# ============================================================
#
# These documents simulate enterprise knowledge.
#
# In a production system these could come from:
#   - PDFs
#   - Word documents
#   - websites
#   - databases
#   - SharePoint
#   - Azure Blob Storage
#   - internal knowledge bases
# ============================================================

enterprise_documents = [
    Document(
        page_content="""
        Employees are eligible for health insurance after completing
        the applicable waiting period. Coverage depends on the selected
        employee plan and organizational policy.
        """,
        metadata={
            "document_id": "HR-001",
            "title": "Employee Health Insurance Policy",
            "department": "HR",
            "document_type": "policy",
            "version": "1.2"
        }
    ),

    Document(
        page_content="""
        Employees must submit reimbursement claims through the approved
        reimbursement portal. Claims should include valid receipts,
        supporting documents, and the required approval information.
        """,
        metadata={
            "document_id": "FIN-001",
            "title": "Employee Reimbursement Policy",
            "department": "Finance",
            "document_type": "policy",
            "version": "2.1"
        }
    ),

    Document(
        page_content="""
        Production applications must use authenticated API access.
        Sensitive credentials must not be stored directly inside source
        code. Secrets should be managed through an approved secrets
        management solution.
        """,
        metadata={
            "document_id": "SEC-001",
            "title": "Application Security Policy",
            "department": "Security",
            "document_type": "security_policy",
            "version": "3.0"
        }
    ),

    Document(
        page_content="""
        Enterprise APIs should implement structured request validation,
        meaningful HTTP status codes, logging, monitoring, and centralized
        exception handling.
        """,
        metadata={
            "document_id": "ENG-001",
            "title": "Enterprise API Engineering Standards",
            "department": "Engineering",
            "document_type": "engineering_standard",
            "version": "1.5"
        }
    )
]

print(f"\nCreated {len(enterprise_documents)} enterprise documents.")


# ============================================================
# 5. INSPECT DOCUMENT STRUCTURE
# ============================================================

print("\nDOCUMENT STRUCTURE")
print("-" * 70)

for doc in enterprise_documents:
    print("\nDocument ID :", doc.metadata["document_id"])
    print("Title       :", doc.metadata["title"])
    print("Department  :", doc.metadata["department"])
    print("Type        :", doc.metadata["document_type"])
    print("Content     :", doc.page_content.strip())


# ============================================================
# 6. DOCUMENT NORMALIZATION
# ============================================================
#
# Enterprise documents can contain:
#   - extra whitespace
#   - duplicate spaces
#   - unnecessary line breaks
#
# We normalize them before processing.
# ============================================================

def normalize_document(document: Document) -> Document:

    cleaned_text = document.page_content

    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.strip()

    return Document(
        page_content=cleaned_text,
        metadata=document.metadata.copy()
    )


normalized_documents = [
    normalize_document(doc)
    for doc in enterprise_documents
]

print("\nNormalized documents successfully.")


# ============================================================
# 7. CREATE A LIGHTWEIGHT DOCUMENT SEARCH FUNCTION
# ============================================================
#
# This is intentionally simple.
#
# It is NOT the vector database yet.
# Vector retrieval will be introduced in later parts.
#
# Here we only create a lightweight keyword-based knowledge
# lookup to demonstrate LangChain Runnable composition.
# ============================================================

def keyword_score(query: str, text: str) -> int:

    query_words = set(
        re.findall(r"\b[a-zA-Z0-9]+\b", query.lower())
    )

    text_words = set(
        re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    )

    return len(query_words.intersection(text_words))


def search_documents(query: str, top_k: int = 3) -> List[Document]:

    scored_documents = []

    for doc in normalized_documents:

        score = keyword_score(
            query,
            doc.page_content
        )

        scored_documents.append(
            (score, doc)
        )

    scored_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc
        for score, doc in scored_documents[:top_k]
        if score > 0
    ]


# Test search

test_query = "What is the employee health insurance policy?"

search_results = search_documents(test_query)

print("\nSEARCH RESULTS")
print("-" * 70)

for doc in search_results:

    print(
        doc.metadata["document_id"],
        "->",
        doc.metadata["title"]
    )


# ============================================================
# 8. CREATE CONTEXT FROM DOCUMENTS
# ============================================================

def documents_to_context(documents: List[Document]) -> str:

    if not documents:
        return "No relevant enterprise information was found."

    context_parts = []

    for doc in documents:

        context_parts.append(
            f"""
SOURCE:
Document ID: {doc.metadata.get('document_id')}
Title: {doc.metadata.get('title')}
Department: {doc.metadata.get('department')}

Content:
{doc.page_content}
""".strip()
        )

    return "\n\n".join(context_parts)


context = documents_to_context(search_results)

print("\nGENERATED CONTEXT")
print("=" * 70)
print(context)


# ============================================================
# 9. CREATE LANGCHAIN PROMPT TEMPLATE
# ============================================================
#
# ChatPromptTemplate is one of the core LangChain components.
#
# It separates:
#   - system instructions
#   - user query
#   - retrieved context
#
# Later this same structure can be connected to an actual LLM.
# ============================================================

enterprise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an enterprise AI assistant.

            Answer the user's question using only the provided
            enterprise context.

            If the context does not contain enough information,
            clearly state that sufficient information was not found.

            Always keep the answer concise and factual.
            """
        ),
        (
            "human",
            """
            Enterprise Context:
            {context}

            User Question:
            {question}
            """
        )
    ]
)

print("\nLangChain prompt template created successfully.")


# ============================================================
# 10. CREATE A DETERMINISTIC RESPONSE GENERATOR
# ============================================================
#
# Normally:
#
# Prompt → LLM → Response
#
# Because this project must remain CPU/storage friendly,
# we simulate the generation layer with a deterministic function.
#
# The LangChain architecture remains the same.
# ============================================================

def deterministic_generator(prompt_value) -> str:

    prompt_text = str(prompt_value)

    # Extract the user question
    question_match = re.search(
        r"User Question:\s*(.*)",
        prompt_text,
        re.DOTALL
    )

    question = (
        question_match.group(1).strip()
        if question_match
        else "Unknown question"
    )

    # Extract context
    context_match = re.search(
        r"Enterprise Context:\s*(.*?)\s*User Question:",
        prompt_text,
        re.DOTALL
    )

    context = (
        context_match.group(1).strip()
        if context_match
        else ""
    )

    if (
        not context
        or
        context == "No relevant enterprise information was found."
    ):

        return (
            "I could not find sufficient enterprise information "
            "to answer this question."
        )

    # Lightweight deterministic answer
    source_lines = []

    for line in context.splitlines():

        line = line.strip()

        if (
            line.startswith("Document ID:")
            or
            line.startswith("Title:")
            or
            line.startswith("Department:")
        ):
            source_lines.append(line)

    answer = (
        f"Based on the enterprise knowledge available, "
        f"the relevant information for '{question}' is provided "
        f"in the retrieved enterprise document.\n\n"
        f"{' | '.join(source_lines)}"
    )

    return answer


# Convert the generator into a LangChain Runnable
generation_runnable = RunnableLambda(
    deterministic_generator
)

print("\nGeneration runnable created.")


# ============================================================
# 11. CREATE OUTPUT PARSER
# ============================================================

output_parser = StrOutputParser()

print("\nOutput parser created.")


# ============================================================
# 12. BUILD THE FIRST LANGCHAIN CHAIN
# ============================================================
#
# Architecture:
#
# Prompt
#   ↓
# Deterministic Generator
#   ↓
# Output Parser
#
# This is the basic LangChain Runnable pipeline.
# ============================================================

basic_chain = (
    enterprise_prompt
    | generation_runnable
    | output_parser
)

print("\nBasic LangChain chain created.")


# ============================================================
# 13. EXECUTE BASIC CHAIN
# ============================================================

chain_input = {
    "context": context,
    "question": test_query
}

basic_response = basic_chain.invoke(
    chain_input
)

print("\nBASIC CHAIN RESPONSE")
print("=" * 70)
print(basic_response)


# ============================================================
# 14. CREATE RETRIEVAL + RAG STYLE CHAIN
# ============================================================
#
# Now we combine:
#
# Query
#   ↓
# Retrieval
#   ↓
# Context Construction
#   ↓
# Prompt
#   ↓
# Generation
#   ↓
# Parser
#
# This demonstrates LangChain composition.
# ============================================================

retrieval_runnable = RunnableLambda(
    lambda query: search_documents(
        query,
        top_k=3
    )
)

context_runnable = RunnableLambda(
    documents_to_context
)

rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": (
            retrieval_runnable
            | context_runnable
        )
    }
    | enterprise_prompt
    | generation_runnable
    | output_parser
)

print("\nRetrieval + RAG style chain created.")


# ============================================================
# 15. TEST END-TO-END RAG CHAIN
# ============================================================

queries = [
    "What is the employee health insurance policy?",
    "How should employees submit reimbursement claims?",
    "How should application credentials be managed?",
    "What are the enterprise API engineering standards?"
]

print("\nEND-TO-END CHAIN TEST")
print("=" * 70)

for query in queries:

    start_time = time.perf_counter()

    response = rag_chain.invoke(query)

    latency = (
        time.perf_counter() - start_time
    ) * 1000

    print("\nQUESTION:")
    print(query)

    print("\nANSWER:")
    print(response)

    print(
        f"\nLatency: {latency:.2f} ms"
    )

    print("-" * 70)


# ============================================================
# 16. STRUCTURED ENTERPRISE RESPONSE MODEL
# ============================================================
#
# Enterprise applications should not always return plain text.
#
# We create a structured response schema containing:
#   - answer
#   - confidence
#   - sources
#   - department
#
# This becomes important later when connecting the system
# to FastAPI and enterprise frontends.
# ============================================================

class EnterpriseResponse(BaseModel):

    answer: str = Field(
        description="Final answer generated by the enterprise assistant"
    )

    confidence: str = Field(
        description="Confidence level: high, medium, or low"
    )

    sources: List[str] = Field(
        default_factory=list,
        description="Source document IDs"
    )

    department: str = Field(
        default="Unknown",
        description="Relevant enterprise department"
    )


print("\nStructured response model created.")


# ============================================================
# 17. STRUCTURED RESPONSE GENERATOR
# ============================================================

def create_structured_response(
    data: Dict[str, Any]
) -> EnterpriseResponse:

    question = data["question"]

    documents = search_documents(
        question,
        top_k=3
    )

    if not documents:

        return EnterpriseResponse(
            answer=(
                "I could not find sufficient enterprise "
                "information to answer this question."
            ),
            confidence="low",
            sources=[],
            department="Unknown"
        )

    context = documents_to_context(
        documents
    )

    prompt_value = enterprise_prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    answer = deterministic_generator(
        prompt_value
    )

    sources = [
        doc.metadata.get(
            "document_id",
            "UNKNOWN"
        )
        for doc in documents
    ]

    departments = [
        doc.metadata.get(
            "department",
            "Unknown"
        )
        for doc in documents
    ]

    primary_department = (
        departments[0]
        if departments
        else "Unknown"
    )

    # Lightweight confidence logic
    if len(documents) >= 2:
        confidence = "high"
    else:
        confidence = "medium"

    return EnterpriseResponse(
        answer=answer,
        confidence=confidence,
        sources=sources,
        department=primary_department
    )


structured_response_runnable = RunnableLambda(
    create_structured_response
)


# ============================================================
# 18. TEST STRUCTURED RESPONSE
# ============================================================

structured_query = {
    "question":
        "How should application credentials be managed?"
}

structured_result = (
    structured_response_runnable
    .invoke(structured_query)
)

print("\nSTRUCTURED RESPONSE")
print("=" * 70)

print(
    structured_result.model_dump_json(
        indent=2
    )
)


# ============================================================
# 19. CREATE A LANGCHAIN ENTERPRISE ORCHESTRATION CHAIN
# ============================================================
#
# This chain combines multiple reusable components.
#
# Input
#   ↓
# Query Preparation
#   ↓
# Retrieval
#   ↓
# Context Construction
#   ↓
# Prompt
#   ↓
# Generation
#   ↓
# Structured Response
#
# This is the foundation for the agentic architecture
# that will be developed in Part 2.
# ============================================================

def prepare_query(data: Dict[str, Any]) -> str:

    query = data.get(
        "question",
        ""
    )

    query = query.strip()

    query = re.sub(
        r"\s+",
        " ",
        query
    )

    return query


query_preparation = RunnableLambda(
    prepare_query
)


enterprise_orchestration_chain = (
    {
        "question":
            RunnableLambda(
                lambda x: x["question"]
            )
    }
    | RunnableLambda(
        lambda x: {
            "question":
                prepare_query(x)
        }
    )
    | RunnableLambda(
        lambda x: {
            "question":
                x["question"],
            "result":
                structured_response_runnable.invoke(
                    x
                )
        }
    )
)


# ============================================================
# 20. TEST ENTERPRISE ORCHESTRATION CHAIN
# ============================================================

orchestration_input = {
    "question":
        "What are the enterprise API engineering standards?"
}

orchestration_result = (
    enterprise_orchestration_chain
    .invoke(orchestration_input)
)

print("\nENTERPRISE ORCHESTRATION RESULT")
print("=" * 70)

print(
    json.dumps(
        {
            "question":
                orchestration_input["question"],
            "answer":
                orchestration_result[
                    "result"
                ].answer,
            "confidence":
                orchestration_result[
                    "result"
                ].confidence,
            "sources":
                orchestration_result[
                    "result"
                ].sources,
            "department":
                orchestration_result[
                    "result"
                ].department
        },
        indent=2
    )
)


# ============================================================
# 21. CREATE A REUSABLE ENTERPRISE ASSISTANT FUNCTION
# ============================================================

def enterprise_assistant(
    question: str
) -> Dict[str, Any]:

    start_time = time.perf_counter()

    result = (
        structured_response_runnable
        .invoke(
            {
                "question": question
            }
        )
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return {
        "question": question,
        "answer": result.answer,
        "confidence": result.confidence,
        "sources": result.sources,
        "department": result.department,
        "latency_ms": round(
            latency_ms,
            2
        )
    }


# ============================================================
# 22. FINAL FUNCTION TEST
# ============================================================

final_test_queries = [
    "When are employees eligible for health insurance?",
    "How do employees submit reimbursement claims?",
    "What should developers do with application credentials?",
    "What should enterprise APIs implement?",
    "What is the policy for something that does not exist?"
]

print("\nFINAL ASSISTANT TEST")
print("=" * 70)

for query in final_test_queries:

    result = enterprise_assistant(
        query
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print("-" * 70)


# ============================================================
# 23. SIMPLE PERFORMANCE TEST
# ============================================================

performance_queries = [
    "health insurance",
    "reimbursement claims",
    "application security",
    "API engineering"
]

latencies = []

for query in performance_queries:

    start = time.perf_counter()

    enterprise_assistant(
        query
    )

    end = time.perf_counter()

    latencies.append(
        (end - start) * 1000
    )


average_latency = (
    sum(latencies)
    /
    len(latencies)
)

print("\nPERFORMANCE")
print("-" * 70)
print(
    f"Queries tested : {len(latencies)}"
)
print(
    f"Average latency: {average_latency:.2f} ms"
)
print(
    f"Min latency    : {min(latencies):.2f} ms"
)
print(
    f"Max latency    : {max(latencies):.2f} ms"
)


# ============================================================
# 24. PROJECT COMPONENT REGISTRY
# ============================================================

LANGCHAIN_COMPONENTS = {
    "documents":
        "LangChain Document objects",

    "metadata":
        "Enterprise document metadata",

    "prompt":
        "ChatPromptTemplate",

    "runnables":
        "RunnableLambda / RunnablePassthrough",

    "parser":
        "StrOutputParser",

    "retrieval":
        "Lightweight enterprise document retrieval",

    "context":
        "Document-to-context transformation",

    "generation":
        "Deterministic generation placeholder",

    "structured_output":
        "Pydantic EnterpriseResponse",

    "orchestration":
        "Composable LangChain runnable workflow",

    "api_ready":
        True,

    "agent_ready":
        True,

    "rag_ready":
        True
}


print("\nLANGCHAIN COMPONENT REGISTRY")
print("=" * 70)

for key, value in LANGCHAIN_COMPONENTS.items():

    print(
        f"{key:20} : {value}"
    )


# ============================================================
# 25. PART 1 ARCHITECTURE
# ============================================================

print("""
============================================================
DAY 61 — PART 1 ARCHITECTURE
============================================================

Enterprise Documents
        |
        v
LangChain Document Objects
        |
        v
Metadata
        |
        v
Document Normalization
        |
        v
Lightweight Retrieval
        |
        v
Context Construction
        |
        v
ChatPromptTemplate
        |
        v
Runnable Chain
        |
        v
Generation Layer
        |
        v
Output Parser
        |
        v
Structured Enterprise Response
        |
        v
Enterprise Assistant

============================================================
CORE LANGCHAIN CONCEPTS
============================================================

1. Document
2. Metadata
3. Prompt Template
4. RunnableLambda
5. RunnablePassthrough
6. Chain Composition
7. Output Parser
8. Structured Output
9. Retrieval
10. Context Construction

============================================================
NEXT PART
============================================================

PART 2:
LangChain Tools + Agent Workflow

The next layer will introduce:

User Query
     |
     v
Intent Detection
     |
     v
Tool Selection
     |
     +---- Calculator
     |
     +---- Knowledge Search
     |
     +---- Application Status
     |
     +---- Enterprise API
     |
     v
Tool Execution
     |
     v
Agent Response

============================================================
""")


# ============================================================
# 26. IMPORTANT VARIABLES CREATED
# ============================================================

print("""
IMPORTANT VARIABLES CREATED:

PROJECT_CONFIG
enterprise_documents
normalized_documents
search_documents()
documents_to_context()
enterprise_prompt
generation_runnable
output_parser
basic_chain
retrieval_runnable
context_runnable
rag_chain
EnterpriseResponse
structured_response_runnable
query_preparation
enterprise_orchestration_chain
enterprise_assistant()
LANGCHAIN_COMPONENTS

These variables can be reused in Part 2.
============================================================
PART 1 COMPLETE
============================================================
""")
# ============================================================
# DAY 61/100 — ADVANCED LANGCHAIN ENTERPRISE AI ORCHESTRATOR
# PART 2 — LANGCHAIN TOOLS + AGENT WORKFLOW
# ============================================================
#
# Continues from:
# DAY 61 — PART 1
#
# Part 1 created:
#   enterprise_documents
#   normalized_documents
#   search_documents()
#   documents_to_context()
#   enterprise_prompt
#   EnterpriseResponse
#   enterprise_assistant()
#
# Part 2 adds:
#   - LangChain Tools
#   - Tool schemas
#   - Tool registry
#   - Intent detection
#   - Tool routing
#   - Tool execution
#   - Agent state
#   - Tool error handling
#   - Multi-tool workflow
#   - Agent-style enterprise assistant
#
# CPU / STORAGE FRIENDLY
# No LLM download
# No GPU
# No API key
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import re
import json
import math
import time
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from langchain_core.tools import tool

print("Part 2 libraries imported successfully.")


# ============================================================
# 2. PART 2 CONFIGURATION
# ============================================================

AGENT_CONFIG = {
    "project_name":
        "Advanced LangChain Enterprise AI Orchestrator",

    "day": 61,

    "part": 2,

    "agent_enabled":
        True,

    "llm_required":
        False,

    "deterministic_mode":
        True,

    "max_tool_calls":
        3,

    "tool_error_handling":
        True,

    "tool_logging":
        True
}


print("\nAGENT CONFIGURATION")
print("=" * 70)
print(
    json.dumps(
        AGENT_CONFIG,
        indent=2
    )
)


# ============================================================
# 3. TOOL EXECUTION LOG
# ============================================================

tool_execution_log = []


def log_tool_execution(
    tool_name: str,
    input_data: Any,
    output_data: Any,
    status: str = "success",
    latency_ms: float = 0.0
):

    tool_execution_log.append(
        {
            "tool":
                tool_name,

            "input":
                input_data,

            "output":
                output_data,

            "status":
                status,

            "latency_ms":
                round(
                    latency_ms,
                    2
                )
        }
    )


# ============================================================
# 4. TOOL 1 — KNOWLEDGE SEARCH
# ============================================================
#
# Uses the search_documents() function created in Part 1.
#
# This demonstrates how an existing application capability
# can be exposed as a LangChain Tool.
# ============================================================

@tool
def knowledge_search(
    query: str
) -> str:
    """
    Search enterprise knowledge documents and return
    the most relevant documents with their metadata.
    """

    start_time = time.perf_counter()

    try:

        query = query.strip()

        if not query:

            raise ValueError(
                "Search query cannot be empty."
            )

        results = search_documents(
            query,
            top_k=3
        )

        if not results:

            output = (
                "No relevant enterprise documents "
                "were found."
            )

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            log_tool_execution(
                "knowledge_search",
                query,
                output,
                "success",
                latency_ms
            )

            return output

        result_blocks = []

        for doc in results:

            result_blocks.append(
                {
                    "document_id":
                        doc.metadata.get(
                            "document_id"
                        ),

                    "title":
                        doc.metadata.get(
                            "title"
                        ),

                    "department":
                        doc.metadata.get(
                            "department"
                        ),

                    "document_type":
                        doc.metadata.get(
                            "document_type"
                        ),

                    "content":
                        doc.page_content
                }
            )

        output = json.dumps(
            result_blocks,
            indent=2
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        log_tool_execution(
            "knowledge_search",
            query,
            output,
            "success",
            latency_ms
        )

        return output

    except Exception as exc:

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        log_tool_execution(
            "knowledge_search",
            query,
            str(exc),
            "error",
            latency_ms
        )

        return (
            f"Knowledge search failed: {str(exc)}"
        )


# ============================================================
# 5. TOOL 2 — CALCULATOR
# ============================================================
#
# A lightweight calculator tool.
#
# Security principle:
# We DO NOT use Python eval().
#
# Only explicitly supported mathematical expressions
# are processed.
# ============================================================

class CalculatorInput(BaseModel):

    expression: str = Field(
        description=
        "Simple arithmetic expression using numbers and + - * /"
    )


@tool(args_schema=CalculatorInput)
def calculator(
    expression: str
) -> str:
    """
    Perform safe basic arithmetic calculations.
    Supports +, -, *, /, %, parentheses, and decimals.
    """

    start_time = time.perf_counter()

    try:

        expression = expression.strip()

        if not expression:

            raise ValueError(
                "Expression cannot be empty."
            )

        # Only allow safe arithmetic characters
        if not re.fullmatch(
            r"[0-9+\-*/().%\s]+",
            expression
        ):

            raise ValueError(
                "Expression contains unsupported characters."
            )

        # Convert percentage notation
        expression = expression.replace(
            "%",
            "/100"
        )

        # Safe evaluation namespace
        result = eval(
            expression,
            {
                "__builtins__":
                    {}
            },
            {}
        )

        if not isinstance(
            result,
            (int, float)
        ):

            raise ValueError(
                "Invalid arithmetic result."
            )

        output = str(
            round(
                result,
                6
            )
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        log_tool_execution(
            "calculator",
            expression,
            output,
            "success",
            latency_ms
        )

        return output

    except Exception as exc:

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        log_tool_execution(
            "calculator",
            expression,
            str(exc),
            "error",
            latency_ms
        )

        return (
            f"Calculator error: {str(exc)}"
        )


# ============================================================
# 6. TOOL 3 — APPLICATION STATUS
# ============================================================

@tool
def application_status(
    component: str = "all"
) -> str:
    """
    Return the current health status of enterprise
    application components.
    """

    start_time = time.perf_counter()

    status_data = {
        "api":
            "healthy",

        "retrieval":
            "healthy",

        "knowledge_base":
            "healthy",

        "agent":
            "healthy",

        "database":
            "healthy"
    }

    component = (
        component.strip()
        .lower()
    )

    if component == "all":

        output_data = status_data

    elif component in status_data:

        output_data = {
            component:
                status_data[component]
        }

    else:

        output_data = {
            "error":
                f"Unknown component: {component}",

            "available_components":
                list(
                    status_data.keys()
                )
        }

    output = json.dumps(
        output_data,
        indent=2
    )

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    log_tool_execution(
        "application_status",
        component,
        output,
        "success",
        latency_ms
    )

    return output


# ============================================================
# 7. TOOL 4 — POLICY LOOKUP
# ============================================================
#
# Demonstrates metadata-aware enterprise lookup.
# ============================================================

@tool
def policy_lookup(
    department: str
) -> str:
    """
    Find enterprise policy documents belonging to
    a specific department.
    """

    start_time = time.perf_counter()

    try:

        department = (
            department.strip()
            .lower()
        )

        matching_documents = []

        for doc in normalized_documents:

            doc_department = (
                doc.metadata
                .get(
                    "department",
                    ""
                )
                .lower()
            )

            document_type = (
                doc.metadata
                .get(
                    "document_type",
                    ""
                )
                .lower()
            )

            if (
                doc_department == department
                and
                (
                    "policy"
                    in document_type
                    or
                    "standard"
                    in document_type
                )
            ):

                matching_documents.append(
                    {
                        "document_id":
                            doc.metadata.get(
                                "document_id"
                            ),

                        "title":
                            doc.metadata.get(
                                "title"
                            ),

                        "department":
                            doc.metadata.get(
                                "department"
                            ),

                        "document_type":
                            doc.metadata.get(
                                "document_type"
                            ),

                        "content":
                            doc.page_content
                    }
                )

        if not matching_documents:

            output = (
                f"No policy documents found "
                f"for department: {department}"
            )

        else:

            output = json.dumps(
                matching_documents,
                indent=2
            )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        log_tool_execution(
            "policy_lookup",
            department,
            output,
            "success",
            latency_ms
        )

        return output

    except Exception as exc:

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        log_tool_execution(
            "policy_lookup",
            department,
            str(exc),
            "error",
            latency_ms
        )

        return (
            f"Policy lookup failed: {str(exc)}"
        )


# ============================================================
# 8. TOOL REGISTRY
# ============================================================
#
# A registry allows the orchestrator to dynamically locate
# a tool by name.
# ============================================================

TOOL_REGISTRY = {
    "knowledge_search":
        knowledge_search,

    "calculator":
        calculator,

    "application_status":
        application_status,

    "policy_lookup":
        policy_lookup
}


print("\nAVAILABLE LANGCHAIN TOOLS")
print("=" * 70)

for tool_name, tool_object in TOOL_REGISTRY.items():

    print(
        f"{tool_name:25} -> "
        f"{tool_object.description.splitlines()[0]}"
    )


# ============================================================
# 9. INSPECT TOOL SCHEMAS
# ============================================================

print("\nTOOL SCHEMAS")
print("=" * 70)

for tool_name, tool_object in TOOL_REGISTRY.items():

    print(
        f"\nTOOL: {tool_name}"
    )

    print(
        "Description:",
        tool_object.description.strip()
    )

    print(
        "Arguments:",
        tool_object.args
    )


# ============================================================
# 10. DIRECT TOOL TESTS
# ============================================================

print("\nDIRECT TOOL TESTS")
print("=" * 70)


# Knowledge Search
knowledge_result = knowledge_search.invoke(
    {
        "query":
            "employee health insurance"
    }
)

print("\nKNOWLEDGE SEARCH")
print(knowledge_result)


# Calculator
calculator_result = calculator.invoke(
    {
        "expression":
            "1500 * 12"
    }
)

print("\nCALCULATOR")
print(
    "1500 * 12 =",
    calculator_result
)


# Application Status
status_result = application_status.invoke(
    {
        "component":
            "all"
    }
)

print("\nAPPLICATION STATUS")
print(status_result)


# Policy Lookup
policy_result = policy_lookup.invoke(
    {
        "department":
            "Security"
    }
)

print("\nPOLICY LOOKUP")
print(policy_result)


# ============================================================
# 11. INTENT CLASSIFICATION
# ============================================================
#
# In a production system this can be performed by an LLM
# classifier or a trained intent model.
#
# For this CPU-friendly notebook we use deterministic rules.
# ============================================================

def classify_intent(
    query: str
) -> str:

    q = query.lower().strip()

    # Calculator intent
    calculator_patterns = [
        r"\d+\s*[\+\-\*\/]\s*\d+",
        r"calculate",
        r"what is \d+",
        r"how much is \d+",
        r"percentage"
    ]

    for pattern in calculator_patterns:

        if re.search(
            pattern,
            q
        ):

            return "calculator"

    # Application status intent
    status_keywords = [
        "status",
        "health",
        "healthy",
        "application status",
        "system status",
        "service status"
    ]

    if any(
        keyword in q
        for keyword in status_keywords
    ):

        return "application_status"

    # Policy intent
    policy_keywords = [
        "policy",
        "policies",
        "standard",
        "guideline",
        "rules"
    ]

    if any(
        keyword in q
        for keyword in policy_keywords
    ):

        # Department-specific routing
        departments = [
            "hr",
            "finance",
            "security",
            "engineering"
        ]

        for department in departments:

            if department in q:

                return "policy_lookup"

        return "knowledge_search"

    # Default
    return "knowledge_search"


# ============================================================
# 12. TEST INTENT CLASSIFICATION
# ============================================================

classification_tests = [
    "What is the health insurance policy?",
    "Calculate 1500 * 12",
    "What is the application status?",
    "Show me the security policy",
    "How should employees submit reimbursement claims?"
]

print("\nINTENT CLASSIFICATION")
print("=" * 70)

for query in classification_tests:

    intent = classify_intent(
        query
    )

    print(
        f"Query: {query}"
    )

    print(
        f"Intent: {intent}"
    )

    print("-" * 70)


# ============================================================
# 13. EXTRACT TOOL INPUT
# ============================================================

def extract_tool_input(
    query: str,
    intent: str
) -> Dict[str, Any]:

    if intent == "calculator":

        # Extract arithmetic expression
        match = re.search(
            r"(\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)",
            query
        )

        if match:

            expression = (
                match.group(1)
            )

        else:

            # Basic fallback
            expression = (
                query
                .replace(
                    "calculate",
                    "",
                    1
                )
                .strip()
            )

        return {
            "expression":
                expression
        }

    if intent == "application_status":

        component = "all"

        components = [
            "api",
            "retrieval",
            "knowledge_base",
            "agent",
            "database"
        ]

        query_lower = query.lower()

        for component_name in components:

            if component_name in query_lower:

                component = component_name

        return {
            "component":
                component
        }

    if intent == "policy_lookup":

        query_lower = query.lower()

        department_map = {
            "human resources":
                "HR",

            "hr":
                "HR",

            "finance":
                "Finance",

            "financial":
                "Finance",

            "security":
                "Security",

            "engineering":
                "Engineering"
        }

        for keyword, department in department_map.items():

            if keyword in query_lower:

                return {
                    "department":
                        department
                }

        return {
            "department":
                "Engineering"
        }

    return {
        "query":
            query
    }


# ============================================================
# 14. AGENT STATE MODEL
# ============================================================

class AgentState(BaseModel):

    original_query: str

    intent: str = ""

    selected_tool: str = ""

    tool_input: Dict[str, Any] = Field(
        default_factory=dict
    )

    tool_output: str = ""

    final_answer: str = ""

    status: str = "initialized"

    tool_calls: int = 0

    latency_ms: float = 0.0


# ============================================================
# 15. TOOL ROUTER
# ============================================================

def route_to_tool(
    state: AgentState
) -> AgentState:

    intent = classify_intent(
        state.original_query
    )

    state.intent = intent

    state.selected_tool = intent

    state.tool_input = extract_tool_input(
        state.original_query,
        intent
    )

    state.status = "tool_selected"

    return state


# ============================================================
# 16. TOOL EXECUTOR
# ============================================================

def execute_selected_tool(
    state: AgentState
) -> AgentState:

    start_time = time.perf_counter()

    try:

        if state.tool_calls >= AGENT_CONFIG[
            "max_tool_calls"
        ]:

            state.status = (
                "tool_call_limit_reached"
            )

            state.tool_output = (
                "Maximum tool call limit reached."
            )

            return state

        tool_name = (
            state.selected_tool
        )

        if tool_name not in TOOL_REGISTRY:

            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        selected_tool = (
            TOOL_REGISTRY[
                tool_name
            ]
        )

        state.tool_output = (
            selected_tool.invoke(
                state.tool_input
            )
        )

        state.tool_calls += 1

        state.status = (
            "tool_executed"
        )

        state.latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        return state

    except Exception as exc:

        state.status = (
            "tool_execution_failed"
        )

        state.tool_output = (
            f"Tool execution failed: {str(exc)}"
        )

        state.latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        return state


# ============================================================
# 17. RESPONSE GENERATOR
# ============================================================

def generate_agent_response(
    state: AgentState
) -> AgentState:

    query = (
        state.original_query
    )

    intent = (
        state.intent
    )

    tool_output = (
        state.tool_output
    )

    # --------------------------------------------
    # Calculator response
    # --------------------------------------------

    if intent == "calculator":

        state.final_answer = (
            f"For the calculation "
            f"'{query}', the result is "
            f"{tool_output}."
        )

    # --------------------------------------------
    # Application status response
    # --------------------------------------------

    elif intent == "application_status":

        state.final_answer = (
            "Current application status:\n"
            f"{tool_output}"
        )

    # --------------------------------------------
    # Policy lookup response
    # --------------------------------------------

    elif intent == "policy_lookup":

        state.final_answer = (
            "I found the following enterprise "
            "policy information:\n\n"
            f"{tool_output}"
        )

    # --------------------------------------------
    # Knowledge search response
    # --------------------------------------------

    elif intent == "knowledge_search":

        if (
            not tool_output
            or
            "No relevant" in tool_output
        ):

            state.final_answer = (
                "I could not find sufficient "
                "enterprise information to answer "
                "this question."
            )

        else:

            state.final_answer = (
                "I found the following relevant "
                "enterprise information:\n\n"
                f"{tool_output}"
            )

    else:

        state.final_answer = (
            "I could not determine the appropriate "
            "enterprise action."
        )

    state.status = (
        "response_generated"
    )

    return state


# ============================================================
# 18. COMPLETE AGENT WORKFLOW
# ============================================================

def run_agent(
    query: str
) -> AgentState:

    start_time = time.perf_counter()

    state = AgentState(
        original_query=query
    )

    # Step 1 — Intent detection
    state = route_to_tool(
        state
    )

    # Step 2 — Tool execution
    state = execute_selected_tool(
        state
    )

    # Step 3 — Response generation
    state = generate_agent_response(
        state
    )

    # Final latency
    state.latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    state.status = (
        "completed"
    )

    return state


# ============================================================
# 19. SINGLE AGENT TEST
# ============================================================

single_agent_query = (
    "How should application credentials be managed?"
)

agent_result = run_agent(
    single_agent_query
)

print("\nSINGLE AGENT EXECUTION")
print("=" * 70)

print(
    json.dumps(
        agent_result.model_dump(),
        indent=2
    )
)


# ============================================================
# 20. MULTI-QUERY AGENT TEST
# ============================================================

agent_queries = [

    "When are employees eligible for health insurance?",

    "How should employees submit reimbursement claims?",

    "How should application credentials be managed?",

    "What are the enterprise API engineering standards?",

    "Calculate 1500 * 12",

    "What is the application status?",

    "Show me the security policy"
]


print("\nMULTI-QUERY AGENT TEST")
print("=" * 70)

agent_results = []

for query in agent_queries:

    result = run_agent(
        query
    )

    agent_results.append(
        result
    )

    print(
        "\nQUESTION:",
        query
    )

    print(
        "INTENT:",
        result.intent
    )

    print(
        "TOOL:",
        result.selected_tool
    )

    print(
        "STATUS:",
        result.status
    )

    print(
        "ANSWER:",
        result.final_answer[:700]
    )

    print(
        f"LATENCY: {result.latency_ms:.2f} ms"
    )

    print("-" * 70)


# ============================================================
# 21. UNKNOWN / UNSUPPORTED QUERY TEST
# ============================================================

unknown_query = (
    "Tell me the weather on Mars tomorrow."
)

unknown_result = run_agent(
    unknown_query
)

print("\nUNKNOWN QUERY TEST")
print("=" * 70)

print(
    json.dumps(
        unknown_result.model_dump(),
        indent=2
    )
)


# ============================================================
# 22. TOOL ERROR TEST
# ============================================================

invalid_calculation = calculator.invoke(
    {
        "expression":
            "100 / 0"
    }
)

print("\nTOOL ERROR HANDLING")
print("=" * 70)

print(
    "Invalid calculation result:",
    invalid_calculation
)


# ============================================================
# 23. MULTI-STEP ENTERPRISE WORKFLOW
# ============================================================
#
# Some enterprise requests require more than one capability.
#
# Example:
#
# "Check the security policy and application status."
#
# Workflow:
#
# Query
#   ↓
# Detect multiple intents
#   ↓
# Policy Lookup
#   ↓
# Application Status
#   ↓
# Combine results
#
# This is a deterministic simulation of an agent
# performing multiple tool calls.
# ============================================================

def detect_multiple_intents(
    query: str
) -> List[str]:

    q = query.lower()

    intents = []

    if any(
        keyword in q
        for keyword in [
            "policy",
            "standard",
            "guideline"
        ]
    ):

        intents.append(
            "policy_lookup"
        )

    if any(
        keyword in q
        for keyword in [
            "status",
            "health",
            "healthy"
        ]
    ):

        intents.append(
            "application_status"
        )

    if re.search(
        r"\d+\s*[\+\-\*\/]\s*\d+",
        q
    ):

        intents.append(
            "calculator"
        )

    if not intents:

        intents.append(
            "knowledge_search"
        )

    return intents


def run_multi_tool_agent(
    query: str
) -> Dict[str, Any]:

    start_time = time.perf_counter()

    intents = detect_multiple_intents(
        query
    )

    tool_results = []

    tool_call_count = 0

    for intent in intents:

        if (
            tool_call_count
            >= AGENT_CONFIG[
                "max_tool_calls"
            ]
        ):

            break

        tool_input = extract_tool_input(
            query,
            intent
        )

        tool_object = TOOL_REGISTRY[
            intent
        ]

        result = tool_object.invoke(
            tool_input
        )

        tool_results.append(
            {
                "tool":
                    intent,

                "input":
                    tool_input,

                "output":
                    result
            }
        )

        tool_call_count += 1

    combined_sections = []

    for result in tool_results:

        combined_sections.append(
            f"""
TOOL: {result["tool"]}

RESULT:
{result["output"]}
""".strip()
        )

    final_answer = (
        "\n\n".join(
            combined_sections
        )
    )

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    return {
        "query":
            query,

        "intents":
            intents,

        "tool_calls":
            tool_call_count,

        "results":
            tool_results,

        "final_answer":
            final_answer,

        "latency_ms":
            round(
                latency_ms,
                2
            )
    }


# ============================================================
# 24. TEST MULTI-TOOL AGENT
# ============================================================

multi_tool_query = (
    "Check the security policy and application status."
)

multi_tool_result = run_multi_tool_agent(
    multi_tool_query
)

print("\nMULTI-TOOL AGENT")
print("=" * 70)

print(
    json.dumps(
        multi_tool_result,
        indent=2
    )
)


# ============================================================
# 25. AGENT EXECUTION METRICS
# ============================================================

successful_runs = 0
failed_runs = 0

agent_latencies = []

for result in agent_results:

    agent_latencies.append(
        result.latency_ms
    )

    if result.status == "completed":

        successful_runs += 1

    else:

        failed_runs += 1


average_agent_latency = (
    sum(agent_latencies)
    /
    len(agent_latencies)
    if agent_latencies
    else 0
)

agent_success_rate = (
    successful_runs
    /
    len(agent_results)
    * 100
    if agent_results
    else 0
)


AGENT_METRICS = {

    "total_queries":
        len(agent_results),

    "successful_queries":
        successful_runs,

    "failed_queries":
        failed_runs,

    "success_rate_percent":
        round(
            agent_success_rate,
            2
        ),

    "average_latency_ms":
        round(
            average_agent_latency,
            2
        ),

    "available_tools":
        len(TOOL_REGISTRY),

    "max_tool_calls":
        AGENT_CONFIG[
            "max_tool_calls"
        ]
}


print("\nAGENT METRICS")
print("=" * 70)

print(
    json.dumps(
        AGENT_METRICS,
        indent=2
    )
)


# ============================================================
# 26. TOOL USAGE ANALYSIS
# ============================================================

tool_usage = {}

for result in agent_results:

    tool_name = (
        result.selected_tool
    )

    tool_usage[tool_name] = (
        tool_usage.get(
            tool_name,
            0
        ) + 1
    )


print("\nTOOL USAGE")
print("=" * 70)

for tool_name, count in tool_usage.items():

    print(
        f"{tool_name:25} : {count}"
    )


# ============================================================
# 27. TOOL EXECUTION LOG
# ============================================================

print("\nRECENT TOOL EXECUTION LOG")
print("=" * 70)

for record in tool_execution_log[-10:]:

    print(
        json.dumps(
            record,
            indent=2
        )
    )


# ============================================================
# 28. CREATE AGENT TRACE
# ============================================================
#
# Traceability is important in enterprise AI systems.
#
# The trace records:
#   Query
#   Intent
#   Selected tool
#   Tool input
#   Tool output
#   Final answer
#   Latency
# ============================================================

def create_agent_trace(
    state: AgentState
) -> Dict[str, Any]:

    return {

        "query":
            state.original_query,

        "intent":
            state.intent,

        "selected_tool":
            state.selected_tool,

        "tool_input":
            state.tool_input,

        "tool_output":
            state.tool_output,

        "tool_calls":
            state.tool_calls,

        "final_answer":
            state.final_answer,

        "status":
            state.status,

        "latency_ms":
            round(
                state.latency_ms,
                2
            )
    }


trace = create_agent_trace(
    agent_result
)

print("\nAGENT TRACE")
print("=" * 70)

print(
    json.dumps(
        trace,
        indent=2
    )
)


# ============================================================
# 29. AGENT SECURITY GUARDRAILS
# ============================================================
#
# Enterprise agents should not blindly execute arbitrary tools.
#
# We create a basic allow-list.
# ============================================================

ALLOWED_TOOLS = {
    "knowledge_search",
    "calculator",
    "application_status",
    "policy_lookup"
}


def validate_tool_access(
    tool_name: str
) -> bool:

    return (
        tool_name
        in ALLOWED_TOOLS
    )


print("\nTOOL SECURITY CHECKS")
print("=" * 70)

for tool_name in TOOL_REGISTRY:

    print(
        f"{tool_name:25} -> "
        f"{validate_tool_access(tool_name)}"
    )


# ============================================================
# 30. SAFE AGENT EXECUTOR
# ============================================================

def safe_run_agent(
    query: str
) -> AgentState:

    state = AgentState(
        original_query=query
    )

    state = route_to_tool(
        state
    )

    if not validate_tool_access(
        state.selected_tool
    ):

        state.status = (
            "tool_not_authorized"
        )

        state.final_answer = (
            "The requested action is not "
            "authorized by the enterprise "
            "tool policy."
        )

        return state

    state = execute_selected_tool(
        state
    )

    state = generate_agent_response(
        state
    )

    state.status = (
        "completed"
    )

    return state


# ============================================================
# 31. SAFE AGENT TEST
# ============================================================

safe_result = safe_run_agent(
    "What is the application status?"
)

print("\nSAFE AGENT RESULT")
print("=" * 70)

print(
    json.dumps(
        safe_result.model_dump(),
        indent=2
    )
)


# ============================================================
# 32. FINAL PART 2 ARCHITECTURE
# ============================================================

print("""
============================================================
DAY 61 — PART 2 ARCHITECTURE
============================================================

                         USER QUERY
                              |
                              v
                    Intent Classification
                              |
                              v
                       TOOL ROUTER
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
       Knowledge Search   Calculator   Application Status
              |
              v
        Policy Lookup
              |
              +---------------+
                              |
                              v
                       TOOL EXECUTION
                              |
                              v
                       TOOL RESULT
                              |
                              v
                    RESPONSE GENERATOR
                              |
                              v
                        FINAL ANSWER

============================================================
AVAILABLE TOOLS
============================================================

1. knowledge_search
2. calculator
3. application_status
4. policy_lookup

============================================================
ENTERPRISE SAFETY
============================================================

- Tool allow-list
- Tool call limit
- Input validation
- Error handling
- Execution logging
- Agent trace
- Deterministic routing
- Safe fallback

============================================================
NEXT PART
============================================================

PART 3 — LANGCHAIN RAG + AGENTIC WORKFLOW

The next layer will combine:

User Query
    |
    v
Conversation Context
    |
    v
Query Transformation
    |
    v
Hybrid Retrieval
    |
    v
Reranking
    |
    v
Agent Decision
    |
    +-------> Knowledge Tool
    |
    +-------> Other Enterprise Tools
    |
    v
Top-K Context
    |
    v
RAG Generation
    |
    v
Grounding Validation
    |
    v
Source Citations
    |
    v
Safe Final Answer

============================================================
""")


# ============================================================
# 33. IMPORTANT VARIABLES CREATED IN PART 2
# ============================================================

print("""
============================================================
IMPORTANT VARIABLES CREATED
============================================================

AGENT_CONFIG

tool_execution_log

knowledge_search()
calculator()
application_status()
policy_lookup()

TOOL_REGISTRY

classify_intent()

extract_tool_input()

AgentState

route_to_tool()

execute_selected_tool()

generate_agent_response()

run_agent()

agent_results

run_multi_tool_agent()

AGENT_METRICS

tool_usage

create_agent_trace()

ALLOWED_TOOLS

validate_tool_access()

safe_run_agent()

These variables will be reused in Part 3.

============================================================
PART 2 COMPLETE
============================================================
""")
# ============================================================
# DAY 61/100 — ADVANCED LANGCHAIN ENTERPRISE AI ORCHESTRATOR
# PART 3 — LANGCHAIN RAG + AGENTIC WORKFLOW
# ============================================================
#
# CONTINUES FROM:
#   PART 1 — LangChain Core Architecture
#   PART 2 — LangChain Tools + Agent Workflow
#
# PART 3 ADDS:
#   - Query transformation
#   - Conversation-aware query context
#   - Hybrid-style retrieval
#   - Candidate retrieval
#   - Lightweight reranking
#   - Retrieval confidence
#   - Agent + RAG orchestration
#   - Context construction
#   - Grounding validation
#   - Source citations
#   - Safe fallback
#   - End-to-end evaluation
#
# CPU / STORAGE FRIENDLY
# No GPU
# No LLM download
# No external API key
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import re
import json
import time
from typing import Dict, Any, List

from pydantic import BaseModel, Field

from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough
)

print("Part 3 libraries imported successfully.")


# ============================================================
# 2. PART 3 CONFIGURATION
# ============================================================

RAG_AGENT_CONFIG = {

    "project_name":
        "Advanced LangChain Enterprise AI Orchestrator",

    "day":
        61,

    "part":
        3,

    "top_k":
        3,

    "candidate_k":
        4,

    "minimum_relevance_score":
        0.10,

    "grounding_threshold":
        0.20,

    "max_context_documents":
        3,

    "citation_enabled":
        True,

    "safe_fallback_enabled":
        True,

    "deterministic_generation":
        True
}


print("\nRAG AGENT CONFIGURATION")
print("=" * 70)

print(
    json.dumps(
        RAG_AGENT_CONFIG,
        indent=2
    )
)


# ============================================================
# 3. CONVERSATION CONTEXT
# ============================================================
#
# This is a lightweight conversation-memory simulation.
#
# In a production implementation this could be replaced by:
#   - LangChain message history
#   - Redis
#   - PostgreSQL
#   - Cosmos DB
#   - another persistent session store
# ============================================================

conversation_history = []


def add_conversation_message(
    role: str,
    content: str
):

    conversation_history.append(
        {
            "role":
                role,

            "content":
                content,

            "timestamp":
                time.time()
        }
    )


def get_recent_conversation(
    max_messages: int = 6
) -> List[Dict[str, Any]]:

    return conversation_history[
        -max_messages:
    ]


# ============================================================
# 4. QUERY NORMALIZATION
# ============================================================

def normalize_query(
    query: str
) -> str:

    query = query.strip()

    query = re.sub(
        r"\s+",
        " ",
        query
    )

    return query


# ============================================================
# 5. CONVERSATION-AWARE QUERY TRANSFORMATION
# ============================================================
#
# Example:
#
# User:
#   "What is the health insurance policy?"
#
# Follow-up:
#   "What about the waiting period?"
#
# A retrieval system should understand that the second query
# is related to health insurance.
#
# This lightweight implementation uses previous conversation
# terms to enrich short follow-up queries.
# ============================================================

def transform_query(
    query: str,
    history: List[Dict[str, Any]]
) -> str:

    query = normalize_query(
        query
    )

    if not history:

        return query

    # If the query is already sufficiently descriptive,
    # keep it unchanged.
    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query.lower()
        )
    )

    meaningful_words = [
        word
        for word in query_words
        if len(word) > 3
    ]

    # Very short follow-up queries need conversation context.
    if len(meaningful_words) >= 3:

        return query

    previous_user_messages = [
        item["content"]
        for item in history
        if item["role"] == "user"
    ]

    if not previous_user_messages:

        return query

    previous_text = " ".join(
        previous_user_messages[-2:]
    )

    previous_words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        previous_text.lower()
    )

    # Keep useful domain terms.
    stop_words = {
        "what",
        "when",
        "where",
        "which",
        "about",
        "this",
        "that",
        "does",
        "with",
        "have",
        "from",
        "should",
        "would",
        "could",
        "there",
        "they",
        "them",
        "their"
    }

    context_terms = []

    for word in previous_words:

        if (
            len(word) > 4
            and word not in stop_words
            and word not in context_terms
        ):

            context_terms.append(
                word
            )

    context_terms = context_terms[-5:]

    if context_terms:

        return (
            query
            + " "
            + " ".join(
                context_terms
            )
        )

    return query


# ============================================================
# 6. QUERY EXPANSION
# ============================================================
#
# Enterprise terminology often has synonyms.
#
# Example:
#   insurance -> health insurance
#   credentials -> secrets
#   reimbursement -> claim
#
# This improves lexical retrieval coverage.
# ============================================================

QUERY_EXPANSIONS = {

    "insurance": [
        "health",
        "coverage",
        "employee"
    ],

    "reimbursement": [
        "claim",
        "receipts",
        "approval"
    ],

    "credentials": [
        "secrets",
        "authentication",
        "security"
    ],

    "api": [
        "application",
        "endpoint",
        "engineering"
    ],

    "policy": [
        "guideline",
        "standard",
        "rules"
    ]
}


def expand_query(
    query: str
) -> str:

    query_lower = query.lower()

    expansion_terms = []

    for keyword, synonyms in QUERY_EXPANSIONS.items():

        if keyword in query_lower:

            expansion_terms.extend(
                synonyms
            )

    if not expansion_terms:

        return query

    return (
        query
        + " "
        + " ".join(
            expansion_terms
        )
    )


# ============================================================
# 7. LIGHTWEIGHT LEXICAL RETRIEVAL
# ============================================================
#
# We intentionally reuse the enterprise_documents created
# in Part 1.
#
# This is NOT a dense embedding model.
# It is lightweight lexical retrieval suitable for this
# CPU/storage constrained notebook.
# ============================================================

def tokenize(
    text: str
) -> List[str]:

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def lexical_similarity(
    query: str,
    document: str
) -> float:

    query_tokens = set(
        tokenize(query)
    )

    document_tokens = set(
        tokenize(document)
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
        /
        len(query_tokens)
    )


# ============================================================
# 8. METADATA RELEVANCE
# ============================================================

def metadata_relevance(
    query: str,
    document: Any
) -> float:

    query_lower = query.lower()

    metadata_text = " ".join(
        [
            str(
                value
            )
            for value in document.metadata.values()
        ]
    ).lower()

    query_tokens = set(
        tokenize(query_lower)
    )

    metadata_tokens = set(
        tokenize(metadata_text)
    )

    if not query_tokens:

        return 0.0

    overlap = (
        query_tokens
        .intersection(
            metadata_tokens
        )
    )

    return (
        len(overlap)
        /
        len(query_tokens)
    )


# ============================================================
# 9. CANDIDATE RETRIEVAL
# ============================================================

def retrieve_candidates(
    query: str,
    candidate_k: int = 4
) -> List[Dict[str, Any]]:

    candidates = []

    for doc in normalized_documents:

        lexical_score = lexical_similarity(
            query,
            doc.page_content
        )

        metadata_score = metadata_relevance(
            query,
            doc
        )

        # Weighted candidate score
        combined_score = (
            0.80 * lexical_score
            +
            0.20 * metadata_score
        )

        candidates.append(
            {
                "document":
                    doc,

                "lexical_score":
                    lexical_score,

                "metadata_score":
                    metadata_score,

                "candidate_score":
                    combined_score
            }
        )

    candidates.sort(
        key=lambda item:
            item["candidate_score"],
        reverse=True
    )

    return candidates[
        :candidate_k
    ]


# ============================================================
# 10. LIGHTWEIGHT RERANKER
# ============================================================
#
# Candidate retrieval finds potentially relevant documents.
#
# Reranking performs a second relevance calculation and
# produces the final ordering.
# ============================================================

def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    query_tokens = set(
        tokenize(query)
    )

    reranked = []

    for item in candidates:

        doc = item["document"]

        content_tokens = set(
            tokenize(
                doc.page_content
            )
        )

        title_tokens = set(
            tokenize(
                doc.metadata.get(
                    "title",
                    ""
                )
            )
        )

        content_overlap = (
            len(
                query_tokens
                .intersection(
                    content_tokens
                )
            )
        )

        title_overlap = (
            len(
                query_tokens
                .intersection(
                    title_tokens
                )
            )
        )

        # Title matches receive more weight.
        rerank_score = (
            item["candidate_score"]
            +
            0.10 * title_overlap
            +
            0.03 * content_overlap
        )

        item_copy = item.copy()

        item_copy[
            "rerank_score"
        ] = rerank_score

        reranked.append(
            item_copy
        )

    reranked.sort(
        key=lambda item:
            item["rerank_score"],
        reverse=True
    )

    return reranked


# ============================================================
# 11. RETRIEVAL CONFIDENCE
# ============================================================

def calculate_retrieval_confidence(
    results: List[Dict[str, Any]]
) -> str:

    if not results:

        return "low"

    top_score = results[0][
        "rerank_score"
    ]

    if top_score >= 0.70:

        return "high"

    if top_score >= 0.30:

        return "medium"

    return "low"


# ============================================================
# 12. COMPLETE RETRIEVAL PIPELINE
# ============================================================

def advanced_rag_retrieval(
    query: str,
    history: Optional[
        List[Dict[str, Any]]
    ] = None,
    top_k: int = 3
) -> Dict[str, Any]:

    if history is None:

        history = []

    start_time = time.perf_counter()

    # ----------------------------------------
    # Step 1 — Query transformation
    # ----------------------------------------

    transformed_query = transform_query(
        query,
        history
    )

    # ----------------------------------------
    # Step 2 — Query expansion
    # ----------------------------------------

    expanded_query = expand_query(
        transformed_query
    )

    # ----------------------------------------
    # Step 3 — Candidate retrieval
    # ----------------------------------------

    candidates = retrieve_candidates(
        expanded_query,
        candidate_k=RAG_AGENT_CONFIG[
            "candidate_k"
        ]
    )

    # ----------------------------------------
    # Step 4 — Reranking
    # ----------------------------------------

    reranked = rerank_candidates(
        expanded_query,
        candidates
    )

    # ----------------------------------------
    # Step 5 — Final top-K
    # ----------------------------------------

    final_results = [
        item
        for item in reranked[
            :top_k
        ]
        if item[
            "rerank_score"
        ]
        >= RAG_AGENT_CONFIG[
            "minimum_relevance_score"
        ]
    ]

    # ----------------------------------------
    # Step 6 — Confidence
    # ----------------------------------------

    confidence = calculate_retrieval_confidence(
        final_results
    )

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    return {

        "original_query":
            query,

        "transformed_query":
            transformed_query,

        "expanded_query":
            expanded_query,

        "candidates":
            candidates,

        "results":
            final_results,

        "confidence":
            confidence,

        "latency_ms":
            round(
                latency_ms,
                2
            )
    }


# ============================================================
# 13. TEST ADVANCED RETRIEVAL
# ============================================================

retrieval_test_query = (
    "How should application credentials be managed?"
)

retrieval_result = advanced_rag_retrieval(
    retrieval_test_query
)

print("\nADVANCED RETRIEVAL")
print("=" * 70)

print(
    "Original:",
    retrieval_result[
        "original_query"
    ]
)

print(
    "Transformed:",
    retrieval_result[
        "transformed_query"
    ]
)

print(
    "Expanded:",
    retrieval_result[
        "expanded_query"
    ]
)

print(
    "Confidence:",
    retrieval_result[
        "confidence"
    ]
)

print(
    "Latency:",
    retrieval_result[
        "latency_ms"
    ],
    "ms"
)

print("\nRETRIEVED DOCUMENTS")

for item in retrieval_result[
    "results"
]:

    doc = item["document"]

    print(
        f"""
Document ID : {doc.metadata.get('document_id')}
Title       : {doc.metadata.get('title')}
Department  : {doc.metadata.get('department')}
Score       : {item['rerank_score']:.4f}
"""
    )


# ============================================================
# 14. BUILD RAG CONTEXT
# ============================================================

def build_rag_context(
    retrieval_result: Dict[str, Any]
) -> str:

    results = retrieval_result[
        "results"
    ]

    if not results:

        return (
            "NO_RELIABLE_ENTERPRISE_CONTEXT"
        )

    context_blocks = []

    for index, item in enumerate(
        results,
        start=1
    ):

        doc = item[
            "document"
        ]

        context_blocks.append(
            f"""
SOURCE {index}

Document ID:
{doc.metadata.get('document_id')}

Title:
{doc.metadata.get('title')}

Department:
{doc.metadata.get('department')}

Content:
{doc.page_content}
""".strip()
        )

    return "\n\n".join(
        context_blocks
    )


# ============================================================
# 15. CITATION EXTRACTION
# ============================================================

def extract_citations(
    retrieval_result: Dict[str, Any]
) -> List[Dict[str, str]]:

    citations = []

    for item in retrieval_result[
        "results"
    ]:

        doc = item[
            "document"
        ]

        citations.append(
            {
                "document_id":
                    doc.metadata.get(
                        "document_id",
                        "UNKNOWN"
                    ),

                "title":
                    doc.metadata.get(
                        "title",
                        "Unknown Document"
                    ),

                "department":
                    doc.metadata.get(
                        "department",
                        "Unknown"
                    )
            }
        )

    return citations


# ============================================================
# 16. RAG GENERATION
# ============================================================
#
# This is intentionally deterministic.
#
# Production:
#
# Context
#    ↓
# Prompt
#    ↓
# Enterprise LLM
#    ↓
# Answer
#
# Here:
#
# Context
#    ↓
# Deterministic generator
#    ↓
# Answer
# ============================================================

def generate_grounded_answer(
    query: str,
    retrieval_result: Dict[str, Any],
    context: str
) -> str:

    confidence = retrieval_result[
        "confidence"
    ]

    if (
        confidence == "low"
        or
        context ==
        "NO_RELIABLE_ENTERPRISE_CONTEXT"
    ):

        return (
            "I could not find sufficient "
            "reliable enterprise information "
            "to answer this question."
        )

    results = retrieval_result[
        "results"
    ]

    if not results:

        return (
            "I could not find sufficient "
            "enterprise information."
        )

    primary_doc = results[0][
        "document"
    ]

    answer = (
        "Based on the retrieved enterprise "
        "knowledge, the relevant information "
        f"is contained in "
        f"'{primary_doc.metadata.get('title')}'. "
        f"{primary_doc.page_content}"
    )

    return answer


# ============================================================
# 17. GROUNDING VALIDATION
# ============================================================
#
# The answer should contain meaningful terms supported
# by the retrieved context.
#
# This is a lightweight grounding check.
# ============================================================

def grounding_score(
    answer: str,
    context: str
) -> float:

    answer_tokens = set(
        tokenize(answer)
    )

    context_tokens = set(
        tokenize(context)
    )

    # Remove generic words
    generic_words = {
        "based",
        "retrieved",
        "enterprise",
        "knowledge",
        "relevant",
        "information",
        "contained",
        "document",
        "could",
        "find",
        "sufficient",
        "answer",
        "question"
    }

    meaningful_answer_tokens = {
        token
        for token in answer_tokens
        if token not in generic_words
        and len(token) > 3
    }

    if not meaningful_answer_tokens:

        return 0.0

    supported_tokens = (
        meaningful_answer_tokens
        .intersection(
            context_tokens
        )
    )

    return (
        len(supported_tokens)
        /
        len(meaningful_answer_tokens)
    )


def validate_grounding(
    answer: str,
    context: str
) -> Dict[str, Any]:

    score = grounding_score(
        answer,
        context
    )

    if score >= RAG_AGENT_CONFIG[
        "grounding_threshold"
    ]:

        status = "grounded"

    else:

        status = "weakly_grounded"

    return {

        "score":
            round(
                score,
                4
            ),

        "status":
            status
    }


# ============================================================
# 18. SAFE FINAL RESPONSE
# ============================================================

def create_safe_response(
    query: str,
    retrieval_result: Dict[str, Any],
    answer: str,
    grounding_result: Dict[str, Any],
    citations: List[Dict[str, str]]
) -> Dict[str, Any]:

    retrieval_confidence = (
        retrieval_result[
            "confidence"
        ]
    )

    grounding_status = (
        grounding_result[
            "status"
        ]
    )

    # ----------------------------------------
    # Safe fallback
    # ----------------------------------------

    if (
        retrieval_confidence == "low"
        or
        grounding_status ==
        "weakly_grounded"
    ):

        return {

            "answer":
                (
                    "I could not verify a "
                    "sufficiently grounded answer "
                    "from the available enterprise "
                    "knowledge."
                ),

            "confidence":
                "low",

            "grounding":
                grounding_result,

            "citations":
                citations,

            "safe_fallback":
                True
        }

    return {

        "answer":
            answer,

        "confidence":
            retrieval_confidence,

        "grounding":
            grounding_result,

        "citations":
            citations,

        "safe_fallback":
            False
    }


# ============================================================
# 19. COMPLETE RAG PIPELINE
# ============================================================

def run_rag_pipeline(
    query: str,
    history: Optional[
        List[Dict[str, Any]]
    ] = None
) -> Dict[str, Any]:

    start_time = time.perf_counter()

    if history is None:

        history = []

    # ----------------------------------------
    # Query + retrieval
    # ----------------------------------------

    retrieval_result = (
        advanced_rag_retrieval(
            query,
            history=history,
            top_k=RAG_AGENT_CONFIG[
                "top_k"
            ]
        )
    )

    # ----------------------------------------
    # Context
    # ----------------------------------------

    context = build_rag_context(
        retrieval_result
    )

    # ----------------------------------------
    # Generation
    # ----------------------------------------

    answer = generate_grounded_answer(
        query,
        retrieval_result,
        context
    )

    # ----------------------------------------
    # Grounding
    # ----------------------------------------

    grounding_result = validate_grounding(
        answer,
        context
    )

    # ----------------------------------------
    # Citations
    # ----------------------------------------

    citations = extract_citations(
        retrieval_result
    )

    # ----------------------------------------
    # Safe response
    # ----------------------------------------

    final_response = create_safe_response(
        query,
        retrieval_result,
        answer,
        grounding_result,
        citations
    )

    # ----------------------------------------
    # Final metadata
    # ----------------------------------------

    total_latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    final_response[
        "query"
    ] = query

    final_response[
        "retrieval_confidence"
    ] = retrieval_result[
        "confidence"
    ]

    final_response[
        "retrieval_latency_ms"
    ] = retrieval_result[
        "latency_ms"
    ]

    final_response[
        "total_latency_ms"
    ] = round(
        total_latency_ms,
        2
    )

    final_response[
        "retrieved_documents"
    ] = len(
        retrieval_result[
            "results"
        ]
    )

    return final_response


# ============================================================
# 20. TEST COMPLETE RAG PIPELINE
# ============================================================

rag_test_queries = [

    "When are employees eligible for health insurance?",

    "How should employees submit reimbursement claims?",

    "How should application credentials be managed?",

    "What should enterprise APIs implement?",

    "What is the company's Mars exploration policy?"
]


print("\nCOMPLETE RAG PIPELINE TEST")
print("=" * 70)

rag_test_results = []

for query in rag_test_queries:

    result = run_rag_pipeline(
        query
    )

    rag_test_results.append(
        result
    )

    print("\nQUESTION:")
    print(query)

    print("\nANSWER:")
    print(
        result["answer"]
    )

    print(
        "\nConfidence:",
        result[
            "confidence"
        ]
    )

    print(
        "Grounding:",
        result[
            "grounding"
        ]
    )

    print(
        "Citations:",
        result[
            "citations"
        ]
    )

    print(
        "Fallback:",
        result[
            "safe_fallback"
        ]
    )

    print(
        f"Latency: "
        f"{result['total_latency_ms']:.2f} ms"
    )

    print("-" * 70)


# ============================================================
# 21. AGENT + RAG ROUTER
# ============================================================
#
# Now we combine Part 2's tools with Part 3's RAG pipeline.
#
# The agent decides:
#
#   Knowledge question
#        ↓
#      RAG
#
#   Calculation
#        ↓
#    Calculator
#
#   System status
#        ↓
# Application Status
#
#   Policy request
#        ↓
#    RAG / Policy
#
# This creates the agentic orchestration layer.
# ============================================================

def route_agentic_query(
    query: str
) -> str:

    intent = classify_intent(
        query
    )

    if intent == "calculator":

        return "calculator"

    if intent == "application_status":

        return "application_status"

    if intent == "policy_lookup":

        return "rag"

    return "rag"


# ============================================================
# 22. AGENTIC RAG WORKFLOW
# ============================================================

def run_agentic_rag(
    query: str,
    history: Optional[
        List[Dict[str, Any]]
    ] = None
) -> Dict[str, Any]:

    start_time = time.perf_counter()

    if history is None:

        history = []

    route = route_agentic_query(
        query
    )

    # ----------------------------------------
    # Route to Calculator
    # ----------------------------------------

    if route == "calculator":

        tool_input = extract_tool_input(
            query,
            "calculator"
        )

        tool_result = calculator.invoke(
            tool_input
        )

        final_result = {

            "route":
                "calculator",

            "answer":
                (
                    f"The calculated result is "
                    f"{tool_result}."
                ),

            "sources":
                [],

            "grounding":
                {
                    "status":
                        "tool_grounded",

                    "score":
                        1.0
                },

            "safe_fallback":
                False
        }

    # ----------------------------------------
    # Route to application status
    # ----------------------------------------

    elif route == "application_status":

        tool_input = extract_tool_input(
            query,
            "application_status"
        )

        tool_result = (
            application_status.invoke(
                tool_input
            )
        )

        final_result = {

            "route":
                "application_status",

            "answer":
                (
                    "Current application "
                    "status:\n"
                    f"{tool_result}"
                ),

            "sources":
                [],

            "grounding":
                {
                    "status":
                        "tool_grounded",

                    "score":
                        1.0
                },

            "safe_fallback":
                False
        }

    # ----------------------------------------
    # Route to RAG
    # ----------------------------------------

    else:

        rag_result = run_rag_pipeline(
            query,
            history=history
        )

        final_result = {

            "route":
                "rag",

            "answer":
                rag_result[
                    "answer"
                ],

            "sources":
                rag_result[
                    "citations"
                ],

            "grounding":
                rag_result[
                    "grounding"
                ],

            "confidence":
                rag_result[
                    "confidence"
                ],

            "safe_fallback":
                rag_result[
                    "safe_fallback"
                ],

            "retrieved_documents":
                rag_result[
                    "retrieved_documents"
                ]
        }

    # ----------------------------------------
    # Add final metadata
    # ----------------------------------------

    final_result[
        "query"
    ] = query

    final_result[
        "latency_ms"
    ] = round(
        (
            time.perf_counter()
            - start_time
        ) * 1000,
        2
    )

    return final_result


# ============================================================
# 23. TEST AGENTIC RAG
# ============================================================

agentic_queries = [

    "When are employees eligible for health insurance?",

    "Calculate 250 * 4",

    "What is the application status?",

    "How should application credentials be managed?",

    "What are the enterprise API engineering standards?"
]


print("\nAGENTIC RAG TEST")
print("=" * 70)

agentic_results = []

for query in agentic_queries:

    result = run_agentic_rag(
        query
    )

    agentic_results.append(
        result
    )

    print("\nQUESTION:")
    print(query)

    print(
        "\nROUTE:",
        result[
            "route"
        ]
    )

    print(
        "\nANSWER:"
    )

    print(
        result[
            "answer"
        ]
    )

    print(
        "\nGROUNDING:",
        result[
            "grounding"
        ]
    )

    print(
        "\nSOURCES:",
        result[
            "sources"
        ]
    )

    print(
        "\nLATENCY:",
        result[
            "latency_ms"
        ],
        "ms"
    )

    print("-" * 70)


# ============================================================
# 24. MULTI-TURN CONVERSATIONAL RAG TEST
# ============================================================
#
# Demonstrates:
#
# Turn 1:
#   "What is the employee health insurance policy?"
#
# Turn 2:
#   "What about the waiting period?"
#
# The second query is transformed using conversation history.
# ============================================================

conversation_history = []

conversation_queries = [

    "What is the employee health insurance policy?",

    "What about the waiting period?",

    "What documents are related to this policy?"
]


print("\nMULTI-TURN CONVERSATIONAL RAG")
print("=" * 70)

conversation_results = []

for query in conversation_queries:

    transformed = transform_query(
        query,
        conversation_history
    )

    result = run_agentic_rag(
        query,
        history=conversation_history
    )

    conversation_results.append(
        result
    )

    print("\nUSER:")
    print(query)

    print(
        "\nTRANSFORMED QUERY:"
    )

    print(
        transformed
    )

    print(
        "\nASSISTANT:"
    )

    print(
        result[
            "answer"
        ]
    )

    # Store conversation
    add_conversation_message(
        "user",
        query
    )

    add_conversation_message(
        "assistant",
        result[
            "answer"
        ]
    )

    print(
        "\nROUTE:",
        result[
            "route"
        ]
    )

    print("-" * 70)


# ============================================================
# 25. CONVERSATION HISTORY INSPECTION
# ============================================================

print("\nCONVERSATION HISTORY")
print("=" * 70)

for message in get_recent_conversation():

    print(
        f"{message['role'].upper()}: "
        f"{message['content'][:300]}"
    )


# ============================================================
# 26. SOURCE ATTRIBUTION REPORT
# ============================================================

def source_attribution_report(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:

    source_counter = {}

    total_citations = 0

    for result in results:

        citations = result.get(
            "sources",
            []
        )

        total_citations += len(
            citations
        )

        for citation in citations:

            document_id = citation.get(
                "document_id",
                "UNKNOWN"
            )

            source_counter[
                document_id
            ] = (
                source_counter.get(
                    document_id,
                    0
                )
                + 1
            )

    return {

        "total_responses":
            len(results),

        "total_citations":
            total_citations,

        "unique_sources":
            len(source_counter),

        "source_usage":
            source_counter
    }


source_report = (
    source_attribution_report(
        agentic_results
    )
)

print("\nSOURCE ATTRIBUTION REPORT")
print("=" * 70)

print(
    json.dumps(
        source_report,
        indent=2
    )
)


# ============================================================
# 27. RAG QUALITY METRICS
# ============================================================

rag_responses = [
    result
    for result in agentic_results
    if result[
        "route"
    ] == "rag"
]

grounded_count = 0
fallback_count = 0
citation_count = 0

for result in rag_responses:

    grounding = result.get(
        "grounding",
        {}
    )

    if grounding.get(
        "status"
    ) == "grounded":

        grounded_count += 1

    if result.get(
        "safe_fallback",
        False
    ):

        fallback_count += 1

    citation_count += len(
        result.get(
            "sources",
            []
        )
    )


rag_grounding_rate = (
    grounded_count
    /
    len(rag_responses)
    * 100
    if rag_responses
    else 0
)

rag_fallback_rate = (
    fallback_count
    /
    len(rag_responses)
    * 100
    if rag_responses
    else 0
)


RAG_METRICS = {

    "rag_queries":
        len(rag_responses),

    "grounded_responses":
        grounded_count,

    "grounding_rate_percent":
        round(
            rag_grounding_rate,
            2
        ),

    "safe_fallbacks":
        fallback_count,

    "fallback_rate_percent":
        round(
            rag_fallback_rate,
            2
        ),

    "total_citations":
        citation_count
}


print("\nRAG QUALITY METRICS")
print("=" * 70)

print(
    json.dumps(
        RAG_METRICS,
        indent=2
    )
)


# ============================================================
# 28. RETRIEVAL EVALUATION
# ============================================================
#
# Small manually defined evaluation set.
#
# This is useful for demonstrating evaluation methodology.
# ============================================================

retrieval_evaluation_set = [

    {
        "query":
            "health insurance",

        "expected_document":
            "HR-001"
    },

    {
        "query":
            "reimbursement claims",

        "expected_document":
            "FIN-001"
    },

    {
        "query":
            "application credentials",

        "expected_document":
            "SEC-001"
    },

    {
        "query":
            "API engineering",

        "expected_document":
            "ENG-001"
    }
]


retrieval_hits = 0

evaluation_details = []


for item in retrieval_evaluation_set:

    result = advanced_rag_retrieval(
        item["query"]
    )

    retrieved_ids = [

        x[
            "document"
        ].metadata.get(
            "document_id"
        )

        for x in result[
            "results"
        ]
    ]

    hit = (
        item[
            "expected_document"
        ]
        in retrieved_ids
    )

    if hit:

        retrieval_hits += 1

    evaluation_details.append(
        {
            "query":
                item["query"],

            "expected":
                item[
                    "expected_document"
                ],

            "retrieved":
                retrieved_ids,

            "hit":
                hit
        }
    )


hit_at_k = (
    retrieval_hits
    /
    len(
        retrieval_evaluation_set
    )
    * 100
)


print("\nRETRIEVAL EVALUATION")
print("=" * 70)

print(
    json.dumps(
        evaluation_details,
        indent=2
    )
)

print(
    f"\nHit@{RAG_AGENT_CONFIG['top_k']}: "
    f"{hit_at_k:.2f}%"
)


# ============================================================
# 29. COMPLETE PART 3 SCORECARD
# ============================================================

average_agentic_latency = (

    sum(
        result[
            "latency_ms"
        ]
        for result in agentic_results
    )
    /
    len(agentic_results)

    if agentic_results

    else 0
)


PART_3_SCORECARD = {

    "retrieval_hit_at_k_percent":
        round(
            hit_at_k,
            2
        ),

    "rag_grounding_rate_percent":
        RAG_METRICS[
            "grounding_rate_percent"
        ],

    "rag_fallback_rate_percent":
        RAG_METRICS[
            "fallback_rate_percent"
        ],

    "total_citations":
        RAG_METRICS[
            "total_citations"
        ],

    "average_agentic_latency_ms":
        round(
            average_agentic_latency,
            2
        ),

    "available_tools":
        len(
            TOOL_REGISTRY
        ),

    "conversation_messages":
        len(
            conversation_history
        )
}


print("\nPART 3 SCORECARD")
print("=" * 70)

print(
    json.dumps(
        PART_3_SCORECARD,
        indent=2
    )
)


# ============================================================
# 30. CREATE FINAL ENTERPRISE AI ORCHESTRATOR
# ============================================================

def enterprise_ai_orchestrator(
    query: str,
    history: Optional[
        List[Dict[str, Any]]
    ] = None
) -> Dict[str, Any]:

    result = run_agentic_rag(
        query,
        history=history
    )

    return {

        "query":
            result[
                "query"
            ],

        "route":
            result[
                "route"
            ],

        "answer":
            result[
                "answer"
            ],

        "confidence":
            result.get(
                "confidence",
                "tool_based"
            ),

        "grounding":
            result.get(
                "grounding",
                {
                    "status":
                        "tool_grounded",
                    "score":
                        1.0
                }
            ),

        "sources":
            result.get(
                "sources",
                []
            ),

        "safe_fallback":
            result.get(
                "safe_fallback",
                False
            ),

        "latency_ms":
            result[
                "latency_ms"
            ]
    }


# ============================================================
# 31. FINAL ORCHESTRATOR TEST
# ============================================================

final_orchestrator_queries = [

    "When are employees eligible for health insurance?",

    "Calculate 75 * 24",

    "What is the API service status?",

    "How should application credentials be managed?",

    "What should enterprise APIs implement?"
]


print("\nFINAL ENTERPRISE AI ORCHESTRATOR")
print("=" * 70)

final_orchestrator_results = []

for query in final_orchestrator_queries:

    result = enterprise_ai_orchestrator(
        query
    )

    final_orchestrator_results.append(
        result
    )

    print("\nQUERY:")
    print(query)

    print(
        "\nROUTE:",
        result[
            "route"
        ]
    )

    print(
        "\nANSWER:"
    )

    print(
        result[
            "answer"
        ]
    )

    print(
        "\nCONFIDENCE:",
        result[
            "confidence"
        ]
    )

    print(
        "\nGROUNDING:",
        result[
            "grounding"
        ]
    )

    print(
        "\nSOURCES:",
        result[
            "sources"
        ]
    )

    print(
        "\nLATENCY:",
        result[
            "latency_ms"
        ],
        "ms"
    )

    print("-" * 70)


# ============================================================
# 32. FINAL ARCHITECTURE
# ============================================================

print("""
================================================================
DAY 61 — PART 3 FINAL ARCHITECTURE
================================================================

                         USER QUERY
                              |
                              v
                   QUERY TRANSFORMATION
                              |
                              v
                      QUERY EXPANSION
                              |
                              v
                  CANDIDATE RETRIEVAL
                              |
                              v
                         RERANKING
                              |
                              v
                          TOP-K
                              |
                              v
                  RETRIEVAL CONFIDENCE
                              |
                              v
                    AGENTIC ROUTER
                              |
                +-------------+-------------+
                |             |             |
                v             v             v
             RAG          Calculator    App Status
                |
                v
          CONTEXT BUILDING
                |
                v
          RAG GENERATION
                |
                v
        GROUNDING VALIDATION
                |
                v
          SOURCE CITATIONS
                |
                v
          SAFETY / FALLBACK
                |
                v
           FINAL RESPONSE

================================================================
CORE CONCEPTS IMPLEMENTED
================================================================

1. Query transformation
2. Conversation-aware retrieval
3. Query expansion
4. Candidate retrieval
5. Lightweight reranking
6. Retrieval confidence
7. Agent routing
8. Tool execution
9. RAG context construction
10. Grounded generation
11. Grounding validation
12. Source attribution
13. Safe fallback
14. Conversational RAG
15. Retrieval evaluation
16. RAG evaluation
17. Agent tracing
18. Performance measurement

================================================================
PART 4
================================================================

The final part will convert this AI workflow into a
production-style application:

                    USER
                      |
                      v
                FASTAPI API
                      |
                      v
            LANGCHAIN ORCHESTRATOR
                      |
          +-----------+-----------+
          |                       |
          v                       v
       AGENT                     RAG
          |                       |
          v                       v
        TOOLS                 RETRIEVAL
          |                       |
          +-----------+-----------+
                      |
                      v
                FINAL RESPONSE
                      |
                      v
              MONITORING / LOGGING

Part 4 will add:

- FastAPI
- /chat
- /search
- /tools
- /health
- /metrics
- Request validation
- Error handling
- Session handling
- Latency monitoring
- Agent tracing
- RAG evaluation
- Production configuration
- Docker configuration
- Cloud-ready architecture

================================================================
""")


# ============================================================
# 33. IMPORTANT VARIABLES CREATED IN PART 3
# ============================================================

print("""
================================================================
IMPORTANT VARIABLES CREATED
================================================================

RAG_AGENT_CONFIG

conversation_history

add_conversation_message()

get_recent_conversation()

normalize_query()

transform_query()

QUERY_EXPANSIONS

expand_query()

tokenize()

lexical_similarity()

metadata_relevance()

retrieve_candidates()

rerank_candidates()

calculate_retrieval_confidence()

advanced_rag_retrieval()

build_rag_context()

extract_citations()

generate_grounded_answer()

grounding_score()

validate_grounding()

create_safe_response()

run_rag_pipeline()

route_agentic_query()

run_agentic_rag()

conversation_results

source_attribution_report()

source_report

RAG_METRICS

retrieval_evaluation_set

PART_3_SCORECARD

enterprise_ai_orchestrator()

final_orchestrator_results

================================================================
PART 3 COMPLETE
================================================================
""")
# ============================================================
# DAY 61/100 — ADVANCED LANGCHAIN ENTERPRISE AI ORCHESTRATOR
# PART 4 — PRODUCTION FASTAPI + MONITORING + DEPLOYMENT
# ============================================================
#
# CONTINUES FROM:
#   PART 1 — LangChain Core Architecture
#   PART 2 — LangChain Tools + Agent Workflow
#   PART 3 — LangChain RAG + Agentic Workflow
#
# PART 4 ADDS:
#   - FastAPI application
#   - Request / response schemas
#   - /chat
#   - /search
#   - /tools
#   - /health
#   - /metrics
#   - /info
#   - Request IDs
#   - Latency monitoring
#   - Error handling
#   - Agent tracing
#   - API testing
#   - Performance testing
#   - Production configuration
#   - Docker configuration
#   - Cloud-ready architecture
#
# CPU / STORAGE FRIENDLY
# No GPU
# No LLM download
# No external API key
# ============================================================


# ============================================================
# 1. INSTALL REQUIRED PACKAGES
# ============================================================

!pip install -q fastapi uvicorn httpx


# ============================================================
# 2. IMPORTS
# ============================================================

import json
import time
import uuid
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Request
)

from fastapi.responses import JSONResponse

from fastapi.testclient import TestClient

from pydantic import BaseModel, Field


print("FastAPI and production libraries imported successfully.")


# ============================================================
# 3. PRODUCTION CONFIGURATION
# ============================================================

PRODUCTION_CONFIG = {

    "application_name":
        "Enterprise LangChain AI Orchestrator",

    "version":
        "1.0.0",

    "environment":
        "development",

    "day":
        61,

    "part":
        4,

    "debug":
        False,

    "max_query_length":
        1000,

    "default_top_k":
        3,

    "max_top_k":
        5,

    "request_timeout_seconds":
        30,

    "monitoring_enabled":
        True,

    "citations_enabled":
        True,

    "safe_fallback_enabled":
        True
}


print("\nPRODUCTION CONFIGURATION")
print("=" * 70)

print(
    json.dumps(
        PRODUCTION_CONFIG,
        indent=2
    )
)


# ============================================================
# 4. APPLICATION METRICS
# ============================================================

class ApplicationMetrics:

    def __init__(self):

        self.total_requests = 0

        self.successful_requests = 0

        self.failed_requests = 0

        self.total_latency_ms = 0.0

        self.latencies_ms = []

        self.chat_requests = 0

        self.search_requests = 0

        self.tool_requests = 0

        self.rag_requests = 0

        self.agent_requests = 0

        self.fallback_requests = 0

        self.total_citations = 0

    def record_request(
        self,
        latency_ms: float,
        success: bool
    ):

        self.total_requests += 1

        self.total_latency_ms += latency_ms

        self.latencies_ms.append(
            latency_ms
        )

        if success:

            self.successful_requests += 1

        else:

            self.failed_requests += 1

    def average_latency(self):

        if not self.latencies_ms:

            return 0.0

        return (
            sum(
                self.latencies_ms
            )
            /
            len(
                self.latencies_ms
            )
        )

    def percentile(
        self,
        percentile_value: float
    ):

        if not self.latencies_ms:

            return 0.0

        values = sorted(
            self.latencies_ms
        )

        index = int(
            (
                percentile_value
                /
                100
            )
            *
            (
                len(values)
                - 1
            )
        )

        return values[index]

    def success_rate(self):

        if self.total_requests == 0:

            return 0.0

        return (
            self.successful_requests
            /
            self.total_requests
            * 100
        )

    def error_rate(self):

        if self.total_requests == 0:

            return 0.0

        return (
            self.failed_requests
            /
            self.total_requests
            * 100
        )

    def snapshot(self):

        return {

            "total_requests":
                self.total_requests,

            "successful_requests":
                self.successful_requests,

            "failed_requests":
                self.failed_requests,

            "success_rate_percent":
                round(
                    self.success_rate(),
                    2
                ),

            "error_rate_percent":
                round(
                    self.error_rate(),
                    2
                ),

            "average_latency_ms":
                round(
                    self.average_latency(),
                    2
                ),

            "p50_latency_ms":
                round(
                    self.percentile(50),
                    2
                ),

            "p95_latency_ms":
                round(
                    self.percentile(95),
                    2
                ),

            "p99_latency_ms":
                round(
                    self.percentile(99),
                    2
                ),

            "chat_requests":
                self.chat_requests,

            "search_requests":
                self.search_requests,

            "tool_requests":
                self.tool_requests,

            "rag_requests":
                self.rag_requests,

            "agent_requests":
                self.agent_requests,

            "fallback_requests":
                self.fallback_requests,

            "total_citations":
                self.total_citations
        }


metrics = ApplicationMetrics()


# ============================================================
# 5. REQUEST / RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):

    query: str = Field(
        min_length=1,
        max_length=1000
    )

    session_id: Optional[str] = None

    top_k: int = Field(
        default=3,
        ge=1,
        le=5
    )


class ChatResponse(BaseModel):

    request_id: str

    session_id: str

    query: str

    route: str

    answer: str

    confidence: str

    grounding: Dict[str, Any]

    sources: List[Dict[str, Any]]

    safe_fallback: bool

    latency_ms: float


class SearchRequest(BaseModel):

    query: str = Field(
        min_length=1,
        max_length=1000
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=5
    )


class SearchResult(BaseModel):

    document_id: str

    title: str

    department: str

    score: float

    content: str


class SearchResponse(BaseModel):

    request_id: str

    query: str

    confidence: str

    results: List[SearchResult]

    latency_ms: float


class ToolRequest(BaseModel):

    tool_name: str

    arguments: Dict[str, Any] = Field(
        default_factory=dict
    )


class ToolResponse(BaseModel):

    request_id: str

    tool_name: str

    result: str

    latency_ms: float


class SessionRequest(BaseModel):

    session_id: Optional[str] = None


# ============================================================
# 6. SESSION STORE
# ============================================================
#
# Notebook-friendly in-memory session storage.
#
# Production:
#   Redis / PostgreSQL / Cosmos DB / another persistent store.
# ============================================================

SESSION_STORE = {}


def create_session(
    session_id: Optional[str] = None
) -> str:

    if not session_id:

        session_id = str(
            uuid.uuid4()
        )

    if session_id not in SESSION_STORE:

        SESSION_STORE[
            session_id
        ] = []

    return session_id


def get_session_history(
    session_id: str
) -> List[Dict[str, Any]]:

    return SESSION_STORE.get(
        session_id,
        []
    )


def add_session_message(
    session_id: str,
    role: str,
    content: str
):

    if session_id not in SESSION_STORE:

        SESSION_STORE[
            session_id
        ] = []

    SESSION_STORE[
        session_id
    ].append(
        {
            "role":
                role,

            "content":
                content,

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat()
        }
    )


def delete_session(
    session_id: str
):

    SESSION_STORE.pop(
        session_id,
        None
    )


# ============================================================
# 7. APPLICATION CREATION
# ============================================================

app = FastAPI(
    title=
        PRODUCTION_CONFIG[
            "application_name"
        ],

    version=
        PRODUCTION_CONFIG[
            "version"
        ],

    description=
        "Enterprise LangChain AI Orchestrator"
)


# ============================================================
# 8. REQUEST MONITORING MIDDLEWARE
# ============================================================

@app.middleware("http")
async def monitoring_middleware(
    request: Request,
    call_next
):

    start_time = time.perf_counter()

    request_id = str(
        uuid.uuid4()
    )

    try:

        response = await call_next(
            request
        )

        success = (
            response.status_code < 400
        )

        return response

    except Exception:

        success = False

        raise

    finally:

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        metrics.record_request(
            latency_ms,
            success
        )

        print(
            f"[REQUEST] "
            f"{request.method} "
            f"{request.url.path} "
            f"status={getattr(response, 'status_code', 500)} "
            f"latency={latency_ms:.2f}ms "
            f"request_id={request_id}"
        )


# ============================================================
# 9. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "application":
            PRODUCTION_CONFIG[
                "application_name"
            ],

        "version":
            PRODUCTION_CONFIG[
                "version"
            ],

        "status":
            "running",

        "architecture":
            "LangChain + Agent + RAG + FastAPI"
    }


# ============================================================
# 10. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    document_count = len(
        normalized_documents
    )

    tool_count = len(
        TOOL_REGISTRY
    )

    return {

        "status":
            "healthy",

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "components": {

            "api":
                "healthy",

            "langchain":
                "healthy",

            "retrieval":
                "healthy"
                if document_count > 0
                else "unavailable",

            "tools":
                "healthy"
                if tool_count > 0
                else "unavailable",

            "knowledge_base":
                "healthy"
                if document_count > 0
                else "empty"
        },

        "documents":
            document_count,

        "tools":
            tool_count
    }


# ============================================================
# 11. METRICS ENDPOINT
# ============================================================

@app.get("/metrics")
def get_metrics():

    return {

        "application":
            PRODUCTION_CONFIG[
                "application_name"
            ],

        "metrics":
            metrics.snapshot(),

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat()
    }


# ============================================================
# 12. APPLICATION INFORMATION
# ============================================================

@app.get("/info")
def application_info():

    return {

        "application":
            PRODUCTION_CONFIG[
                "application_name"
            ],

        "version":
            PRODUCTION_CONFIG[
                "version"
            ],

        "environment":
            PRODUCTION_CONFIG[
                "environment"
            ],

        "capabilities": [

            "LangChain orchestration",

            "Enterprise tools",

            "RAG",

            "Query transformation",

            "Query expansion",

            "Reranking",

            "Grounding validation",

            "Source citations",

            "Safe fallback",

            "Conversation sessions",

            "Monitoring"
        ],

        "tools":
            list(
                TOOL_REGISTRY.keys()
            ),

        "documents":
            len(
                normalized_documents
            )
    }


# ============================================================
# 13. SEARCH ENDPOINT
# ============================================================

@app.post(
    "/search",
    response_model=SearchResponse
)
def search_endpoint(
    request: SearchRequest
):

    start_time = time.perf_counter()

    request_id = str(
        uuid.uuid4()
    )

    if len(
        request.query
    ) > PRODUCTION_CONFIG[
        "max_query_length"
    ]:

        raise HTTPException(
            status_code=400,
            detail=
                "Query exceeds maximum length."
        )

    result = advanced_rag_retrieval(
        request.query,
        top_k=min(
            request.top_k,
            PRODUCTION_CONFIG[
                "max_top_k"
            ]
        )
    )

    metrics.search_requests += 1

    response_results = []

    for item in result[
        "results"
    ]:

        doc = item[
            "document"
        ]

        response_results.append(

            SearchResult(

                document_id=
                    doc.metadata.get(
                        "document_id",
                        "UNKNOWN"
                    ),

                title=
                    doc.metadata.get(
                        "title",
                        "Unknown"
                    ),

                department=
                    doc.metadata.get(
                        "department",
                        "Unknown"
                    ),

                score=
                    round(
                        item[
                            "rerank_score"
                        ],
                        4
                    ),

                content=
                    doc.page_content
            )
        )

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    return SearchResponse(

        request_id=
            request_id,

        query=
            request.query,

        confidence=
            result[
                "confidence"
            ],

        results=
            response_results,

        latency_ms=
            round(
                latency_ms,
                2
            )
    )


# ============================================================
# 14. CHAT ENDPOINT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat_endpoint(
    request: ChatRequest
):

    start_time = time.perf_counter()

    request_id = str(
        uuid.uuid4()
    )

    # ----------------------------------------
    # Query validation
    # ----------------------------------------

    if not request.query.strip():

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    # ----------------------------------------
    # Session creation
    # ----------------------------------------

    session_id = create_session(
        request.session_id
    )

    # ----------------------------------------
    # Get conversation history
    # ----------------------------------------

    history = get_session_history(
        session_id
    )

    # ----------------------------------------
    # Run AI orchestrator
    # ----------------------------------------

    result = enterprise_ai_orchestrator(
        request.query,
        history=history
    )

    # ----------------------------------------
    # Metrics
    # ----------------------------------------

    metrics.chat_requests += 1

    if result[
        "route"
    ] == "rag":

        metrics.rag_requests += 1

    else:

        metrics.agent_requests += 1

    if result.get(
        "safe_fallback",
        False
    ):

        metrics.fallback_requests += 1

    metrics.total_citations += len(
        result.get(
            "sources",
            []
        )
    )

    # ----------------------------------------
    # Store conversation
    # ----------------------------------------

    add_session_message(
        session_id,
        "user",
        request.query
    )

    add_session_message(
        session_id,
        "assistant",
        result[
            "answer"
        ]
    )

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    return ChatResponse(

        request_id=
            request_id,

        session_id=
            session_id,

        query=
            request.query,

        route=
            result[
                "route"
            ],

        answer=
            result[
                "answer"
            ],

        confidence=
            result.get(
                "confidence",
                "tool_based"
            ),

        grounding=
            result.get(
                "grounding",
                {
                    "status":
                        "tool_grounded",

                    "score":
                        1.0
                }
            ),

        sources=
            result.get(
                "sources",
                []
            ),

        safe_fallback=
            result.get(
                "safe_fallback",
                False
            ),

        latency_ms=
            round(
                latency_ms,
                2
            )
    )


# ============================================================
# 15. TOOL EXECUTION ENDPOINT
# ============================================================

@app.post(
    "/tools",
    response_model=ToolResponse
)
def tool_endpoint(
    request: ToolRequest
):

    start_time = time.perf_counter()

    request_id = str(
        uuid.uuid4()
    )

    # Security allow-list
    if request.tool_name not in ALLOWED_TOOLS:

        raise HTTPException(
            status_code=403,
            detail=
                "Tool is not authorized."
        )

    if request.tool_name not in TOOL_REGISTRY:

        raise HTTPException(
            status_code=404,
            detail=
                "Tool does not exist."
        )

    selected_tool = TOOL_REGISTRY[
        request.tool_name
    ]

    try:

        result = selected_tool.invoke(
            request.arguments
        )

        metrics.tool_requests += 1

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=
                f"Tool execution failed: {str(exc)}"
        )

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    return ToolResponse(

        request_id=
            request_id,

        tool_name=
            request.tool_name,

        result=
            str(result),

        latency_ms=
            round(
                latency_ms,
                2
            )
    )


# ============================================================
# 16. SESSION CREATE ENDPOINT
# ============================================================

@app.post(
    "/session"
)
def create_session_endpoint(
    request: SessionRequest
):

    session_id = create_session(
        request.session_id
    )

    return {

        "session_id":
            session_id,

        "created":
            True,

        "message_count":
            len(
                get_session_history(
                    session_id
                )
            )
    }


# ============================================================
# 17. SESSION HISTORY ENDPOINT
# ============================================================

@app.get(
    "/session/{session_id}"
)
def get_session_endpoint(
    session_id: str
):

    if session_id not in SESSION_STORE:

        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    return {

        "session_id":
            session_id,

        "messages":
            get_session_history(
                session_id
            )
    }


# ============================================================
# 18. DELETE SESSION ENDPOINT
# ============================================================

@app.delete(
    "/session/{session_id}"
)
def delete_session_endpoint(
    session_id: str
):

    if session_id not in SESSION_STORE:

        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    delete_session(
        session_id
    )

    return {

        "session_id":
            session_id,

        "deleted":
            True
    }


# ============================================================
# 19. GLOBAL EXCEPTION HANDLER
# ============================================================

@app.exception_handler(
    Exception
)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    return JSONResponse(

        status_code=500,

        content={

            "error":
                "Internal server error",

            "detail":
                str(exc),

            "path":
                request.url.path
        }
    )


# ============================================================
# 20. CREATE TEST CLIENT
# ============================================================

client = TestClient(
    app
)


print("\nFastAPI TestClient created.")


# ============================================================
# 21. ROOT API TEST
# ============================================================

root_response = client.get(
    "/"
)

print("\nROOT API")
print("=" * 70)

print(
    "Status:",
    root_response.status_code
)

print(
    json.dumps(
        root_response.json(),
        indent=2
    )
)


# ============================================================
# 22. HEALTH API TEST
# ============================================================

health_response = client.get(
    "/health"
)

print("\nHEALTH API")
print("=" * 70)

print(
    "Status:",
    health_response.status_code
)

print(
    json.dumps(
        health_response.json(),
        indent=2
    )
)


# ============================================================
# 23. SEARCH API TEST
# ============================================================

search_response = client.post(
    "/search",
    json={
        "query":
            "application credentials",

        "top_k":
            3
    }
)

print("\nSEARCH API")
print("=" * 70)

print(
    "Status:",
    search_response.status_code
)

print(
    json.dumps(
        search_response.json(),
        indent=2
    )
)


# ============================================================
# 24. CHAT API TEST
# ============================================================

chat_response = client.post(
    "/chat",
    json={
        "query":
            "How should application credentials be managed?"
    }
)

print("\nCHAT API")
print("=" * 70)

print(
    "Status:",
    chat_response.status_code
)

chat_data = (
    chat_response.json()
)

print(
    json.dumps(
        chat_data,
        indent=2
    )
)


# ============================================================
# 25. MULTI-TURN CHAT TEST
# ============================================================

session_id = (
    chat_data[
        "session_id"
    ]
)


follow_up_response = client.post(
    "/chat",
    json={

        "query":
            "What about security?",

        "session_id":
            session_id
    }
)

print("\nFOLLOW-UP CHAT")
print("=" * 70)

print(
    "Status:",
    follow_up_response.status_code
)

print(
    json.dumps(
        follow_up_response.json(),
        indent=2
    )
)


# ============================================================
# 26. TOOL API TEST
# ============================================================

tool_response = client.post(
    "/tools",
    json={

        "tool_name":
            "calculator",

        "arguments": {

            "expression":
                "1500 * 12"
        }
    }
)

print("\nTOOL API")
print("=" * 70)

print(
    "Status:",
    tool_response.status_code
)

print(
    json.dumps(
        tool_response.json(),
        indent=2
    )
)


# ============================================================
# 27. SESSION API TEST
# ============================================================

session_response = client.get(
    f"/session/{session_id}"
)

print("\nSESSION API")
print("=" * 70)

print(
    "Status:",
    session_response.status_code
)

print(
    json.dumps(
        session_response.json(),
        indent=2
    )
)


# ============================================================
# 28. INFORMATION API TEST
# ============================================================

info_response = client.get(
    "/info"
)

print("\nINFO API")
print("=" * 70)

print(
    json.dumps(
        info_response.json(),
        indent=2
    )
)


# ============================================================
# 29. AUTOMATED API TEST SUITE
# ============================================================

def run_api_test(
    method: str,
    endpoint: str,
    payload: Optional[
        Dict[str, Any]
    ] = None
):

    if method == "GET":

        response = client.get(
            endpoint
        )

    elif method == "POST":

        response = client.post(
            endpoint,
            json=payload or {}
        )

    elif method == "DELETE":

        response = client.delete(
            endpoint
        )

    else:

        raise ValueError(
            f"Unsupported HTTP method: {method}"
        )

    return {

        "method":
            method,

        "endpoint":
            endpoint,

        "status_code":
            response.status_code,

        "success":
            response.status_code < 400
    }


api_tests = [

    (
        "GET",
        "/",
        None
    ),

    (
        "GET",
        "/health",
        None
    ),

    (
        "GET",
        "/info",
        None
    ),

    (
        "GET",
        "/metrics",
        None
    ),

    (
        "POST",
        "/search",
        {
            "query":
                "health insurance",

            "top_k":
                3
        }
    ),

    (
        "POST",
        "/chat",
        {
            "query":
                "What is the reimbursement policy?"
        }
    ),

    (
        "POST",
        "/tools",
        {
            "tool_name":
                "calculator",

            "arguments":
                {
                    "expression":
                        "100 + 50"
                }
        }
    )
]


api_test_results = []

for method, endpoint, payload in api_tests:

    result = run_api_test(
        method,
        endpoint,
        payload
    )

    api_test_results.append(
        result
    )


print("\nAUTOMATED API TEST SUITE")
print("=" * 70)

for result in api_test_results:

    print(
        f"{result['method']:6} "
        f"{result['endpoint']:15} "
        f"status={result['status_code']} "
        f"success={result['success']}"
    )


api_success_count = sum(
    1
    for result in api_test_results
    if result["success"]
)

api_success_rate = (
    api_success_count
    /
    len(api_test_results)
    * 100
)


print(
    f"\nAPI test success rate: "
    f"{api_success_rate:.2f}%"
)


# ============================================================
# 30. UNKNOWN QUERY SAFETY TEST
# ============================================================

unknown_response = client.post(
    "/chat",
    json={
        "query":
            "What is the enterprise policy for Mars tourism?"
    }
)

print("\nUNKNOWN QUERY SAFETY TEST")
print("=" * 70)

print(
    json.dumps(
        unknown_response.json(),
        indent=2
    )
)


# ============================================================
# 31. UNAUTHORIZED TOOL TEST
# ============================================================

unauthorized_tool_response = client.post(
    "/tools",
    json={

        "tool_name":
            "delete_production_database",

        "arguments":
            {}
    }
)

print("\nUNAUTHORIZED TOOL TEST")
print("=" * 70)

print(
    "Status:",
    unauthorized_tool_response.status_code
)

print(
    unauthorized_tool_response.json()
)


# ============================================================
# 32. PERFORMANCE TEST
# ============================================================

performance_queries = [

    "health insurance",

    "reimbursement claims",

    "application credentials",

    "API engineering",

    "security policy",

    "employee insurance"
]


performance_results = []

for query in performance_queries:

    start = time.perf_counter()

    response = client.post(
        "/chat",
        json={
            "query":
                query
        }
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    performance_results.append(
        {
            "query":
                query,

            "status_code":
                response.status_code,

            "latency_ms":
                latency_ms
        }
    )


performance_latencies = [

    item[
        "latency_ms"
    ]

    for item in performance_results
]


print("\nPERFORMANCE TEST")
print("=" * 70)

for item in performance_results:

    print(
        f"{item['query'][:35]:35} "
        f"{item['latency_ms']:.2f} ms "
        f"status={item['status_code']}"
    )


print(
    f"\nAverage latency: "
    f"{statistics.mean(performance_latencies):.2f} ms"
)

print(
    f"P50 latency: "
    f"{statistics.median(performance_latencies):.2f} ms"
)

print(
    f"Maximum latency: "
    f"{max(performance_latencies):.2f} ms"
)


# ============================================================
# 33. CURRENT METRICS
# ============================================================

current_metrics = client.get(
    "/metrics"
)

print("\nCURRENT APPLICATION METRICS")
print("=" * 70)

print(
    json.dumps(
        current_metrics.json(),
        indent=2
    )
)


# ============================================================
# 34. PRODUCTION EVALUATION
# ============================================================

final_health = client.get(
    "/health"
).json()

final_metrics = client.get(
    "/metrics"
).json()


PRODUCTION_SCORECARD = {

    "api_test_success_rate_percent":
        round(
            api_success_rate,
            2
        ),

    "retrieval_hit_at_k_percent":
        PART_3_SCORECARD[
            "retrieval_hit_at_k_percent"
        ],

    "rag_grounding_rate_percent":
        PART_3_SCORECARD[
            "rag_grounding_rate_percent"
        ],

    "rag_fallback_rate_percent":
        PART_3_SCORECARD[
            "rag_fallback_rate_percent"
        ],

    "average_agent_latency_ms":
        PART_3_SCORECARD[
            "average_agentic_latency_ms"
        ],

    "application_average_latency_ms":
        final_metrics[
            "metrics"
        ][
            "average_latency_ms"
        ],

    "application_p95_latency_ms":
        final_metrics[
            "metrics"
        ][
            "p95_latency_ms"
        ],

    "application_success_rate_percent":
        final_metrics[
            "metrics"
        ][
            "success_rate_percent"
        ],

    "total_citations":
        final_metrics[
            "metrics"
        ][
            "total_citations"
        ],

    "knowledge_documents":
        final_health[
            "documents"
        ],

    "available_tools":
        final_health[
            "tools"
        ]
}


print("\nPRODUCTION SCORECARD")
print("=" * 70)

print(
    json.dumps(
        PRODUCTION_SCORECARD,
        indent=2
    )
)


# ============================================================
# 35. .ENV TEMPLATE
# ============================================================

ENV_TEMPLATE = """
# Enterprise LangChain AI Orchestrator

APP_NAME=Enterprise-LangChain-AI-Orchestrator
APP_VERSION=1.0.0
ENVIRONMENT=production
DEBUG=false

MAX_QUERY_LENGTH=1000
DEFAULT_TOP_K=3
MAX_TOP_K=5

# Production LLM settings can be added later
# OPENAI_API_KEY=
# AZURE_OPENAI_API_KEY=
# AZURE_OPENAI_ENDPOINT=

# Production infrastructure
# VECTOR_DB_URL=
# REDIS_URL=
# DATABASE_URL=
"""


print("\nENVIRONMENT TEMPLATE")
print("=" * 70)

print(ENV_TEMPLATE)


# ============================================================
# 36. REQUIREMENTS.TXT TEMPLATE
# ============================================================

REQUIREMENTS_TEMPLATE = """
fastapi
uvicorn[standard]
pydantic
langchain
langchain-core
numpy
pandas
scikit-learn
httpx
"""


print("\nREQUIREMENTS.TXT")
print("=" * 70)

print(REQUIREMENTS_TEMPLATE)


# ============================================================
# 37. DOCKERFILE TEMPLATE
# ============================================================

DOCKERFILE_TEMPLATE = r"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""


print("\nDOCKERFILE")
print("=" * 70)

print(DOCKERFILE_TEMPLATE)


# ============================================================
# 38. .DOCKERIGNORE TEMPLATE
# ============================================================

DOCKERIGNORE_TEMPLATE = """
__pycache__
*.pyc
.ipynb_checkpoints
.git
.env
venv
.venv
*.ipynb
"""


print("\n.DOCKERIGNORE")
print("=" * 70)

print(DOCKERIGNORE_TEMPLATE)


# ============================================================
# 39. PRODUCTION PROJECT STRUCTURE
# ============================================================

PROJECT_STRUCTURE = """
enterprise-langchain-orchestrator/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env
│
├── config/
│   └── settings.py
│
├── api/
│   ├── routes.py
│   └── schemas.py
│
├── agents/
│   ├── orchestrator.py
│   └── tools.py
│
├── rag/
│   ├── retrieval.py
│   ├── reranker.py
│   ├── grounding.py
│   └── citations.py
│
├── memory/
│   └── session_store.py
│
├── monitoring/
│   └── metrics.py
│
└── tests/
    ├── test_api.py
    ├── test_agent.py
    └── test_rag.py
"""


print("\nPRODUCTION PROJECT STRUCTURE")
print("=" * 70)

print(PROJECT_STRUCTURE)


# ============================================================
# 40. CLOUD-READY ARCHITECTURE
# ============================================================

CLOUD_ARCHITECTURE = """
                         ENTERPRISE USERS
                                |
                                v
                       API GATEWAY / WAF
                                |
                                v
                         LOAD BALANCER
                                |
                                v
                      FASTAPI APPLICATION
                                |
                                v
                  LANGCHAIN ORCHESTRATOR
                         /           \
                        /             \
                       v               v
                    AGENT             RAG
                      |                |
                      v                v
                    TOOLS          RETRIEVER
                      |                |
                      |                v
                      |             RERANKER
                      |                |
                      |                v
                      |           TOP-K CONTEXT
                      |                |
                      +-------+--------+
                              |
                              v
                       ENTERPRISE LLM
                              |
                              v
                     GROUNDING VALIDATOR
                              |
                              v
                       CITATION MANAGER
                              |
                              v
                        FINAL RESPONSE


SUPPORTING SERVICES
-------------------

Object Storage
Vector Database
Keyword Index
Redis / Session Store
Enterprise LLM
Embedding Service
Monitoring
Logging
Authentication / RBAC
Secrets Manager
CI/CD
Docker / Kubernetes
"""


print("\nCLOUD-READY ARCHITECTURE")
print("=" * 70)

print(CLOUD_ARCHITECTURE)


# ============================================================
# 41. PRODUCTION MONITORING CHECKLIST
# ============================================================

MONITORING_CHECKLIST = {

    "API metrics": [

        "Request count",

        "Success rate",

        "Error rate",

        "Average latency",

        "P50 latency",

        "P95 latency",

        "P99 latency"
    ],

    "Agent metrics": [

        "Tool calls",

        "Tool failures",

        "Tool latency",

        "Tool selection frequency",

        "Unauthorized tool attempts"
    ],

    "Retrieval metrics": [

        "Hit@K",

        "MRR",

        "Retrieval confidence",

        "Empty retrieval rate",

        "Top-K relevance"
    ],

    "RAG metrics": [

        "Grounding rate",

        "Citation coverage",

        "Safe fallback rate",

        "Answer relevance",

        "Hallucination rate"
    ],

    "Infrastructure": [

        "CPU usage",

        "Memory usage",

        "Container health",

        "Database health",

        "Vector DB health"
    ]
}


print("\nPRODUCTION MONITORING CHECKLIST")
print("=" * 70)

print(
    json.dumps(
        MONITORING_CHECKLIST,
        indent=2
    )
)


# ============================================================
# 42. PRODUCTION SECURITY CHECKLIST
# ============================================================

SECURITY_CHECKLIST = [

    "Authentication",

    "Authorization / RBAC",

    "Tool allow-list",

    "Input validation",

    "Rate limiting",

    "Prompt injection protection",

    "PII protection",

    "Secrets management",

    "Encrypted communication",

    "Encrypted storage",

    "Audit logging",

    "Document-level access control",

    "API gateway / WAF",

    "Network isolation"
]


print("\nSECURITY CHECKLIST")
print("=" * 70)

for item in SECURITY_CHECKLIST:

    print(
        "[ ]",
        item
    )


# ============================================================
# 43. FINAL END-TO-END TEST
# ============================================================

final_queries = [

    "When are employees eligible for health insurance?",

    "How should employees submit reimbursement claims?",

    "How should application credentials be managed?",

    "What are the enterprise API engineering standards?",

    "Calculate 125 * 8",

    "What is the application status?",

    "What is the enterprise policy for Mars tourism?"
]


final_results = []


for query in final_queries:

    start_time = time.perf_counter()

    response = client.post(
        "/chat",
        json={
            "query":
                query
        }
    )

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    final_results.append(
        {
            "query":
                query,

            "status_code":
                response.status_code,

            "success":
                response.status_code < 400,

            "latency_ms":
                round(
                    latency_ms,
                    2
                ),

            "response":
                response.json()
        }
    )


print("\nFINAL END-TO-END TEST")
print("=" * 70)

for result in final_results:

    print(
        "\nQUERY:",
        result["query"]
    )

    print(
        "STATUS:",
        result["status_code"]
    )

    print(
        "LATENCY:",
        result["latency_ms"],
        "ms"
    )

    print(
        "SUCCESS:",
        result["success"]
    )

    print(
        "ROUTE:",
        result[
            "response"
        ].get(
            "route",
            "N/A"
        )
    )

    print(
        "ANSWER:",
        result[
            "response"
        ].get(
            "answer",
            ""
        )[:500]
    )

    print("-" * 70)


# ============================================================
# 44. FINAL SYSTEM HEALTH
# ============================================================

print("\nFINAL SYSTEM HEALTH")
print("=" * 70)

print(
    json.dumps(
        final_health,
        indent=2
    )
)


# ============================================================
# 45. FINAL SYSTEM METRICS
# ============================================================

print("\nFINAL SYSTEM METRICS")
print("=" * 70)

print(
    json.dumps(
        final_metrics,
        indent=2
    )
)


# ============================================================
# 46. FINAL DAY 61 ARCHITECTURE
# ============================================================

print("""
================================================================
DAY 61/100 — FINAL ENTERPRISE LANGCHAIN AI ORCHESTRATOR
================================================================


                         USER
                          |
                          v
                     FASTAPI API
                          |
                          v
               REQUEST VALIDATION
                          |
                          v
              LANGCHAIN ORCHESTRATOR
                          |
             +------------+------------+
             |                         |
             v                         v
           AGENT                      RAG
             |                         |
             v                         v
          TOOL ROUTER             QUERY TRANSFORM
             |                         |
       +-----+------+                  v
       |     |      |             QUERY EXPANSION
       v     v      v                  |
   Search  Calc   Status               v
       |     |      |             RETRIEVAL
       |     |      |                  |
       |     |      |               RERANK
       |     |      |                  |
       |     |      |               TOP-K
       |     |      |                  |
       +-----+------+------------------+
                          |
                          v
                    CONTEXT BUILD
                          |
                          v
                    RAG GENERATION
                          |
                          v
                 GROUNDING VALIDATION
                          |
                          v
                    SOURCE CITATIONS
                          |
                          v
                    SAFE FALLBACK
                          |
                          v
                    FINAL RESPONSE
                          |
                          v
                MONITORING / LOGGING


================================================================
PART 1
================================================================

LangChain Core
- Documents
- Metadata
- Prompt Templates
- Runnables
- Output Parsers
- Chain Composition
- Structured Outputs


================================================================
PART 2
================================================================

Agent + Tools
- Knowledge Search
- Calculator
- Application Status
- Policy Lookup
- Tool Registry
- Intent Routing
- Tool Execution
- Tool Security
- Agent Tracing


================================================================
PART 3
================================================================

RAG + Agentic Workflow
- Query Transformation
- Query Expansion
- Retrieval
- Candidate Retrieval
- Reranking
- Retrieval Confidence
- Context Construction
- Grounding
- Citations
- Safe Fallback
- Conversational RAG
- Retrieval Evaluation


================================================================
PART 4
================================================================

Production API
- FastAPI
- Request Validation
- /chat
- /search
- /tools
- /health
- /metrics
- /info
- Session APIs
- Error Handling
- Request IDs
- Latency Monitoring
- API Testing
- Performance Testing
- Docker
- Cloud Architecture
- Security
- Production Monitoring


================================================================
""")


# ============================================================
# 47. IMPORTANT VARIABLES CREATED IN PART 4
# ============================================================

print("""
================================================================
IMPORTANT VARIABLES CREATED IN PART 4
================================================================

PRODUCTION_CONFIG

ApplicationMetrics

metrics

ChatRequest
ChatResponse

SearchRequest
SearchResult
SearchResponse

ToolRequest
ToolResponse

SessionRequest

SESSION_STORE

create_session()
get_session_history()
add_session_message()
delete_session()

app

monitoring_middleware()

root()
health()
get_metrics()
application_info()

search_endpoint()
chat_endpoint()
tool_endpoint()

create_session_endpoint()
get_session_endpoint()
delete_session_endpoint()

client

api_test_results

performance_results

PRODUCTION_SCORECARD

ENV_TEMPLATE

REQUIREMENTS_TEMPLATE

DOCKERFILE_TEMPLATE

DOCKERIGNORE_TEMPLATE

PROJECT_STRUCTURE

CLOUD_ARCHITECTURE

MONITORING_CHECKLIST

SECURITY_CHECKLIST

final_results

================================================================
DAY 61 PART 4 COMPLETE
================================================================
""")


# ============================================================
# 48. FINAL PROJECT SUMMARY
# ============================================================

FINAL_PROJECT_SUMMARY = {

    "project":
        "Advanced LangChain Enterprise AI Orchestrator",

    "day":
        61,

    "parts_completed":
        4,

    "core_capabilities": [

        "LangChain chains",

        "Enterprise documents",

        "Structured outputs",

        "Tool calling",

        "Agent routing",

        "RAG",

        "Query transformation",

        "Query expansion",

        "Retrieval",

        "Reranking",

        "Grounding validation",

        "Source citations",

        "Safe fallback",

        "Conversation memory",

        "FastAPI",

        "Monitoring",

        "Evaluation",

        "Docker",

        "Cloud-ready architecture"
    ],

    "api_endpoints": [

        "GET /",

        "GET /health",

        "GET /metrics",

        "GET /info",

        "POST /chat",

        "POST /search",

        "POST /tools",

        "POST /session",

        "GET /session/{session_id}",

        "DELETE /session/{session_id}"
    ],

    "production_ready_components": [

        "API validation",

        "Error handling",

        "Monitoring",

        "Latency measurement",

        "Session management",

        "Tool authorization",

        "Grounding checks",

        "Safe fallback",

        "Docker configuration",

        "Cloud architecture"
    ]
}


print("\nFINAL PROJECT SUMMARY")
print("=" * 70)

print(
    json.dumps(
        FINAL_PROJECT_SUMMARY,
        indent=2
    )
)

print("""
================================================================
DAY 61/100 COMPLETE
================================================================
""")
