# ============================================================
# DAY 54/100
# PROJECT: AGENTIC VECTOR CHATBOT
# PART 1: KNOWLEDGE BASE + VECTOR STORE
#
# CPU FRIENDLY
# LOW STORAGE
# NO EXTERNAL DATASET
# ============================================================


# ============================================================
# SECTION 1 : IMPORTS
# ============================================================

from langchain_core.documents import Document

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

print("Libraries Imported Successfully")
# ============================================================
# SECTION 2 : CREATE KNOWLEDGE BASE
# ============================================================

documents = [

    Document(
        page_content="""
        Artificial Intelligence is the field of creating
        computer systems capable of performing tasks that
        normally require human intelligence. AI is used in
        automation, language processing, computer vision,
        robotics, and recommendation systems.
        """,
        metadata={
            "topic": "AI",
            "category": "fundamentals"
        }
    ),

    Document(
        page_content="""
        Machine Learning is a subset of Artificial Intelligence.
        Machine Learning algorithms learn patterns from data
        and use those patterns to make predictions or decisions.
        Supervised learning, unsupervised learning, and
        reinforcement learning are common types of Machine Learning.
        """,
        metadata={
            "topic": "Machine Learning",
            "category": "fundamentals"
        }
    ),

    Document(
        page_content="""
        Deep Learning uses artificial neural networks with
        multiple layers to learn complex patterns from data.
        It is widely used in computer vision, natural language
        processing, speech recognition, and generative AI.
        """,
        metadata={
            "topic": "Deep Learning",
            "category": "deep learning"
        }
    ),

    Document(
        page_content="""
        Retrieval-Augmented Generation, or RAG, retrieves
        relevant information from an external knowledge base
        and provides that information to a language model.
        This helps generate responses grounded in retrieved
        information.
        """,
        metadata={
            "topic": "RAG",
            "category": "generative AI"
        }
    ),

    Document(
        page_content="""
        A vector database stores numerical representations
        of information called vectors. Vector databases are
        useful for similarity search, semantic search,
        recommendation systems, and retrieval-augmented
        generation applications.
        """,
        metadata={
            "topic": "Vector Database",
            "category": "generative AI"
        }
    ),

    Document(
        page_content="""
        An AI agent is a system that can reason about a task,
        select appropriate tools, execute actions, observe
        results, and continue processing until it can produce
        a useful result.
        """,
        metadata={
            "topic": "AI Agent",
            "category": "agentic AI"
        }
    ),

    Document(
        page_content="""
        LangChain is a framework for building applications
        powered by language models. It provides components
        for documents, prompts, retrieval, vector stores,
        tools, chains, and agents.
        """,
        metadata={
            "topic": "LangChain",
            "category": "framework"
        }
    ),

    Document(
        page_content="""
        Large Language Models are neural network models
        trained on large collections of text. They can perform
        tasks such as question answering, summarization,
        text generation, reasoning, and code generation.
        """,
        metadata={
            "topic": "LLM",
            "category": "generative AI"
        }
    )

]


print()
print("=" * 60)
print("KNOWLEDGE BASE CREATED")
print("=" * 60)

print(
    "Documents:",
    len(documents)
)
# ============================================================
# SECTION 3 : TEXT CLEANING
# ============================================================

def clean_text(text):

    return " ".join(
        text.split()
    )


cleaned_documents = []


for document in documents:

    cleaned_document = Document(

        page_content=clean_text(
            document.page_content
        ),

        metadata=document.metadata

    )

    cleaned_documents.append(
        cleaned_document
    )


print()
print("Text cleaning completed.")
# ============================================================
# SECTION 4 : CREATE LIGHTWEIGHT EMBEDDINGS
# ============================================================

texts = [

    document.page_content

    for document in cleaned_documents

]


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)


document_vectors = vectorizer.fit_transform(
    texts
)


document_vectors = document_vectors.toarray()


print()
print("=" * 60)
print("EMBEDDINGS CREATED")
print("=" * 60)

print(
    "Documents:",
    document_vectors.shape[0]
)

print(
    "Vector dimensions:",
    document_vectors.shape[1]
)
# ============================================================
# SECTION 5 : VECTOR STORE
# ============================================================

class AgenticVectorStore:

    def __init__(
        self,
        documents,
        vectors,
        vectorizer
    ):

        self.documents = documents

        self.vectors = vectors

        self.vectorizer = vectorizer


    def search(
        self,
        query,
        k=3
    ):

        # Convert query to vector

        query_vector = self.vectorizer.transform(
            [query]
        ).toarray()


        # Calculate similarity

        scores = cosine_similarity(

            query_vector,

            self.vectors

        )[0]


        # Get top results

        top_indexes = np.argsort(
            scores
        )[::-1][:k]


        results = []


        for index in top_indexes:

            results.append({

                "document":
                self.documents[index],

                "score":
                float(scores[index])

            })


        return results


# Create vector store

vector_store = AgenticVectorStore(

    documents=cleaned_documents,

    vectors=document_vectors,

    vectorizer=vectorizer

)


print()
print("=" * 60)
print("VECTOR STORE CREATED")
print("=" * 60)
# ============================================================
# SECTION 6 : RETRIEVAL FUNCTION
# ============================================================

def retrieve_knowledge(
    query,
    k=2
):

    results = vector_store.search(
        query,
        k=k
    )


    return results

# ============================================================
# SECTION 7 : TEST VECTOR RETRIEVAL
# ============================================================

test_queries = [

    "What is artificial intelligence?",

    "Explain RAG",

    "What is an AI agent?",

    "What is a vector database?",

    "What are large language models?"

]


print()
print("=" * 60)
print("VECTOR RETRIEVAL TEST")
print("=" * 60)


for query in test_queries:

    results = retrieve_knowledge(
        query,
        k=2
    )


    print()
    print("-" * 60)

    print(
        "QUERY:",
        query
    )


    for result in results:

        document = result["document"]

        score = result["score"]


        print()

        print(
            "Topic:",
            document.metadata["topic"]
        )

        print(
            "Score:",
            round(score, 4)
        )

        print(
            "Content:",
            document.page_content
        )
  # ============================================================
# SECTION 8 : RETRIEVAL WITH THRESHOLD
# ============================================================

SIMILARITY_THRESHOLD = 0.15


def retrieve_relevant_knowledge(
    query,
    k=2
):

    results = vector_store.search(
        query,
        k=k
    )


    relevant_results = []


    for result in results:

        if result["score"] >= SIMILARITY_THRESHOLD:

            relevant_results.append(
                result
            )


    return relevant_results


print()
print(
    "Similarity threshold:",
    SIMILARITY_THRESHOLD
)
# ============================================================
# SECTION 9 : THRESHOLD TEST
# ============================================================

queries = [

    "What is RAG?",

    "Explain AI agents",

    "Tell me about vector databases",

    "What is the weather today?"

]


print()
print("=" * 60)
print("THRESHOLD RETRIEVAL TEST")
print("=" * 60)


for query in queries:

    results = retrieve_relevant_knowledge(
        query
    )


    print()
    print("QUERY:")
    print(query)


    if len(results) == 0:

        print(
            "No relevant knowledge found."
        )


    else:

        for result in results:

            print(
                "→",
                result["document"].metadata["topic"],
                "| Score:",
                round(
                    result["score"],
                    4
                )
            )

# ============================================================
# SECTION 10 : PART 1 VALIDATION
# ============================================================

print()
print("=" * 60)
print("DAY 54 — PART 1 COMPLETED")
print("=" * 60)

print("""
✔ Created AI knowledge base
✔ Created LangChain Documents
✔ Added metadata
✔ Cleaned documents
✔ Created lightweight embeddings
✔ Created vector store
✔ Implemented similarity search
✔ Added retrieval threshold
✔ Tested relevant queries
✔ No external dataset
✔ No large model
✔ No GPU
✔ Very low storage usage
✔ CPU friendly

READY FOR PART 2
""")
# ============================================================
# DAY 54/100
# PART 2 : AGENT TOOLS
# ============================================================


# ============================================================
# SECTION 1 : KNOWLEDGE SEARCH TOOL
# ============================================================

def knowledge_search_tool(query):

    results = retrieve_relevant_knowledge(
        query,
        k=2
    )


    if len(results) == 0:

        return {
            "tool": "knowledge_search",
            "found": False,
            "results": []
        }


    formatted_results = []


    for result in results:

        document = result["document"]

        formatted_results.append({

            "topic":
            document.metadata["topic"],

            "category":
            document.metadata["category"],

            "content":
            document.page_content,

            "score":
            round(
                result["score"],
                4
            )

        })


    return {

        "tool": "knowledge_search",

        "found": True,

        "results": formatted_results

    }


print(
    "Knowledge Search Tool Created"
)
# ============================================================
# SECTION 2 : CALCULATOR TOOL
# ============================================================

def calculator_tool(expression):

    try:

        # Allow only basic mathematical characters

        allowed_characters = (
            "0123456789"
            "+-*/().% "
        )


        if not all(
            character in allowed_characters
            for character in expression
        ):

            return {
                "tool": "calculator",
                "success": False,
                "result": "Invalid expression"
            }


        result = eval(
            expression,
            {
                "__builtins__": None
            },
            {}
        )


        return {

            "tool": "calculator",

            "success": True,

            "result": result

        }


    except Exception:

        return {

            "tool": "calculator",

            "success": False,

            "result": "Could not calculate expression"

        }


print(
    "Calculator Tool Created"
)
# ============================================================
# SECTION 3 : GENERAL TOOL
# ============================================================

def general_tool(query):

    return {

        "tool": "general",

        "response": (
            "I can help with questions related to "
            "AI, Machine Learning, Deep Learning, "
            "RAG, Vector Databases, LangChain, "
            "AI Agents, and basic calculations."
        )

    }


print(
    "General Tool Created"
)
# ============================================================
# SECTION 4 : TOOL REGISTRY
# ============================================================

TOOLS = {

    "knowledge_search":
    knowledge_search_tool,

    "calculator":
    calculator_tool,

    "general":
    general_tool

}


print()
print("=" * 60)
print("AVAILABLE TOOLS")
print("=" * 60)


for tool_name in TOOLS:

    print(
        "-",
        tool_name
    )
    # ============================================================
# SECTION 5 : TEST KNOWLEDGE TOOL
# ============================================================

result = knowledge_search_tool(
    "What is RAG?"
)


print()
print("=" * 60)
print("KNOWLEDGE TOOL TEST")
print("=" * 60)

# ============================================================
# SECTION 6 : TEST CALCULATOR TOOL
# ============================================================

result = calculator_tool(
    "50 * 20"
)


print()
print("=" * 60)
print("CALCULATOR TOOL TEST")
print("=" * 60)


print(result)
# ============================================================
# SECTION 7 : TEST GENERAL TOOL
# ============================================================

result = general_tool(
    "Tell me something"
)


print()
print("=" * 60)
print("GENERAL TOOL TEST")
print("=" * 60)


print(result)
print(result)
# ============================================================
# SECTION 8 : TOOL EXECUTOR
# ============================================================

def execute_tool(
    tool_name,
    query
):

    if tool_name not in TOOLS:

        return {

            "success": False,

            "error":
            "Tool does not exist"

        }


    tool = TOOLS[
        tool_name
    ]


    result = tool(
        query
    )


    return {

        "success": True,

        "tool":
        tool_name,

        "result":
        result

    }


print(
    "Tool Executor Created"
)
# ============================================================
# SECTION 9 : TOOL EXECUTOR TEST
# ============================================================

result = execute_tool(
    "knowledge_search",
    "What is an AI agent?"
)


print()
print("=" * 60)
print("TOOL EXECUTOR TEST")
print("=" * 60)


print(result)
# ============================================================
# SECTION 10 : TOOL DESCRIPTIONS
# ============================================================

TOOL_DESCRIPTIONS = {

    "knowledge_search":
        "Use this tool for questions about "
        "AI, Machine Learning, Deep Learning, "
        "RAG, Vector Databases, LangChain, "
        "AI Agents, and LLMs.",


    "calculator":
        "Use this tool for mathematical "
        "calculations and arithmetic.",


    "general":
        "Use this tool when the question "
        "does not match the knowledge base "
        "or calculator."

}


print()
print("=" * 60)
print("TOOL DESCRIPTIONS")
print("=" * 60)


for name, description in TOOL_DESCRIPTIONS.items():

    print()
    print(
        name.upper()
    )

    print(
        description
    )
# ============================================================
# SECTION 11 : FINAL TOOL TEST
# ============================================================

print()
print("=" * 60)
print("FINAL TOOL TEST")
print("=" * 60)


# Knowledge

knowledge_result = execute_tool(
    "knowledge_search",
    "What is deep learning?"
)


print()
print("KNOWLEDGE TOOL:")

print(
    knowledge_result
)


# Calculator

calculator_result = execute_tool(
    "calculator",
    "100 / 5"
)


print()
print("CALCULATOR TOOL:")

print(
    calculator_result
)


# General

general_result = execute_tool(
    "general",
    "Tell me something"
)


print()
print("GENERAL TOOL:")

print(
    general_result
)
# ============================================================
# SECTION 12 : PART 2 SUMMARY
# ============================================================

print()
print("=" * 60)
print("DAY 54 — PART 2 COMPLETED")
print("=" * 60)

print("""
✔ Knowledge Search Tool created
✔ Vector database connected as a tool
✔ Calculator Tool created
✔ General Tool created
✔ Tool Registry created
✔ Tool Executor created
✔ Tool descriptions created
✔ All tools tested
✔ No external model
✔ No external dataset
✔ No GPU
✔ Minimal storage
✔ CPU friendly

NEXT:
PART 3 — AGENT DECISION MAKING + TOOL ROUTING
""")
# ============================================================
# DAY 54/100
# PART 3 : AGENT DECISION MAKING + TOOL ROUTING
#
# CONTINUES FROM PART-1 AND PART-2
# ============================================================


# ============================================================
# SECTION 1 : KNOWLEDGE KEYWORDS
# ============================================================

KNOWLEDGE_KEYWORDS = [

    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "rag",
    "retrieval",
    "vector",
    "vector database",
    "langchain",
    "agent",
    "ai agent",
    "llm",
    "large language model",
    "embedding",
    "embeddings",
    "generative ai"

]


print(
    "Knowledge keywords loaded:",
    len(KNOWLEDGE_KEYWORDS)
)
# ============================================================
# SECTION 2 : CALCULATOR DETECTION
# ============================================================

MATH_KEYWORDS = [

    "calculate",
    "what is",
    "solve",
    "multiply",
    "divide",
    "plus",
    "minus",
    "add",
    "subtract"

]


def looks_like_math(query):

    query_lower = query.lower().strip()


    # Check for arithmetic operators

    math_symbols = [
        "+",
        "-",
        "*",
        "/",
        "%"
    ]


    for symbol in math_symbols:

        if symbol in query_lower:

            return True


    # Check for calculator-style words

    for keyword in MATH_KEYWORDS:

        if keyword in query_lower:

            # Make sure the query contains a number

            if any(
                character.isdigit()
                for character in query_lower
            ):

                return True


    return False

# ============================================================
# SECTION 3 : KNOWLEDGE QUERY DETECTION
# ============================================================

def looks_like_knowledge_query(query):

    query_lower = query.lower().strip()


    for keyword in KNOWLEDGE_KEYWORDS:

        if keyword in query_lower:

            return True


    return False

# ============================================================
# SECTION 4 : AGENT ROUTER
# ============================================================

def agent_router(query):

    query_lower = query.lower().strip()


    # --------------------------------------------------------
    # STEP 1 : CHECK CALCULATOR
    # --------------------------------------------------------

    if looks_like_math(query):

        return "calculator"


    # --------------------------------------------------------
    # STEP 2 : CHECK KNOWLEDGE BASE
    # --------------------------------------------------------

    if looks_like_knowledge_query(query):

        return "knowledge_search"


    # --------------------------------------------------------
    # STEP 3 : FALLBACK
    # --------------------------------------------------------

    return "general"
# ============================================================
# SECTION 5 : TEST AGENT ROUTER
# ============================================================

test_queries = [

    "What is RAG?",

    "Explain vector databases",

    "What is machine learning?",

    "25 * 8",

    "Calculate 100 / 4",

    "Tell me something interesting",

    "Hello"

]


print()
print("=" * 60)
print("AGENT ROUTER TEST")
print("=" * 60)


for query in test_queries:

    selected_tool = agent_router(
        query
    )


    print()
    print(
        "Query:",
        query
    )

    print(
        "Selected Tool:",
        selected_tool
    )
# ============================================================
# SECTION 6 : AGENT
# ============================================================

def agent(query):

    # --------------------------------------------------------
    # STEP 1 : DECIDE TOOL
    # --------------------------------------------------------

    selected_tool = agent_router(
        query
    )


    # --------------------------------------------------------
    # STEP 2 : EXECUTE TOOL
    # --------------------------------------------------------

    tool_result = execute_tool(
        selected_tool,
        query
    )


    # --------------------------------------------------------
    # STEP 3 : RETURN AGENT RESULT
    # --------------------------------------------------------

    return {

        "query": query,

        "selected_tool":
        selected_tool,

        "tool_result":
        tool_result

    }


print(
    "Agent Created Successfully"
)
# ============================================================
# SECTION 7 : COMPLETE AGENT TEST
# ============================================================

queries = [

    "What is RAG?",

    "What is a vector database?",

    "Explain AI agents",

    "25 * 8",

    "100 / 4",

    "Hello"

]


print()
print("=" * 60)
print("COMPLETE AGENT TEST")
print("=" * 60)


for query in queries:

    result = agent(
        query
    )


    print()
    print("-" * 60)

    print(
        "USER:",
        query
    )


    print(
        "AGENT SELECTED:",
        result["selected_tool"]
    )


    print(
        "TOOL RESULT:"
    )


    print(
        result["tool_result"]
    )

# ============================================================
# SECTION 8 : RESPONSE FORMATTER
# ============================================================

def format_agent_response(agent_result):
    tool = agent_result.get("selected_tool") or agent_result.get("tool")
    
    # Extract the nested output from execute_tool
    tool_result = agent_result.get("tool_result", {})
    if "result" in tool_result:
        tool_result = tool_result["result"]

    # --------------------------------------------------------
    # KNOWLEDGE SEARCH RESPONSE FORMATTING
    # --------------------------------------------------------
    if tool == "knowledge_search":
        if not tool_result.get("found", False):
            return "I could not find relevant knowledge in the database."

        formatted_text = "Here is what I found:\n"
        for item in tool_result.get("results", []):
            formatted_text += f"\n- {item['topic']}: {item['content']}"
        return formatted_text

    # --------------------------------------------------------
    # CALCULATOR RESPONSE FORMATTING
    # --------------------------------------------------------
    elif tool == "calculator":
        if not tool_result.get("success", False):
            return f"Calculation error: {tool_result.get('result', 'Invalid expression')}"
        return f"Result: {tool_result.get('result')}"

    # --------------------------------------------------------
    # GENERAL RESPONSE FORMATTING
    # --------------------------------------------------------
    elif tool == "general":
        return tool_result.get("response", "How can I help you today?")

    return "No response generated."
    

# ============================================================
# SECTION 9 : COMPLETE AGENT PIPELINE
# ============================================================

def run_agent(
    query
):

    # Agent reasoning / routing

    agent_result = agent(
        query
    )


    # Generate user-friendly response

    response = format_agent_response(
        agent_result
    )


    return {

        "query": query,

        "tool":
        agent_result["selected_tool"],

        "response":
        response

    }

# ============================================================
# SECTION 10 : FINAL PIPELINE TEST
# ============================================================

test_queries = [

    "What is artificial intelligence?",

    "Explain RAG",

    "What is a vector database?",

    "What is an AI agent?",

    "50 * 20",

    "Calculate 144 / 12",

    "Hello"

]


print()
print("=" * 60)
print("FINAL AGENT PIPELINE")
print("=" * 60)


for query in test_queries:

    result = run_agent(
        query
    )


    print()
    print("-" * 60)


    print(
        "USER:",
        result["query"]
    )


    print(
        "SELECTED TOOL:",
        result["tool"]
    )


    print()
    print(
        "AGENT RESPONSE:"
    )


    print(
        result["response"]
    )
# ============================================================
# SECTION 11 : AGENT TRACE
# ============================================================

def agent_with_trace(
    query
):

    print()
    print("=" * 60)
    print("AGENT TRACE")
    print("=" * 60)


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print()
    print(
        "1. User Query:"
    )

    print(
        query
    )


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    selected_tool = agent_router(
        query
    )


    print()
    print(
        "2. Tool Selected:"
    )

    print(
        selected_tool
    )


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    tool_result = execute_tool(
        selected_tool,
        query
    )


    print()
    print(
        "3. Tool Executed"
    )


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    response = format_agent_response({

        "selected_tool":
        selected_tool,

        "tool_result":
        tool_result

    })


    print()
    print(
        "4. Final Response:"
    )

    print(
        response
    )


    return response

# ============================================================
# SECTION 12 : PART 3 SUMMARY
# ============================================================

print()
print("=" * 60)
print("DAY 54 — PART 3 COMPLETED")
print("=" * 60)

print("""
✔ Agent router created
✔ Query classification implemented
✔ Knowledge tool routing implemented
✔ Calculator routing implemented
✔ General fallback implemented
✔ Tool execution connected
✔ Agent response formatter created
✔ Agent trace implemented
✔ Complete agent pipeline tested
✔ No large model downloaded
✔ No external dataset
✔ No GPU
✔ Minimal storage
✔ CPU friendly

NEXT:
PART 4 — COMPLETE AGENTIC VECTOR CHATBOT
""")

# ============================================================
# DAY 54/100
# PART 4 : COMPLETE AGENTIC VECTOR CHATBOT
# ============================================================


# ============================================================
# SECTION 1 : CONVERSATION MEMORY
# ============================================================

conversation_history = []


def add_to_memory(
    user_query,
    agent_response
):

    conversation_history.append({

        "user":
        user_query,

        "assistant":
        agent_response

    })


def get_memory():

    return conversation_history

# ============================================================
# SECTION 2 : COMPLETE AGENT
# ============================================================

def complete_agent(
    query
):

    # --------------------------------------------------------
    # STEP 1 : SELECT TOOL
    # --------------------------------------------------------

    selected_tool = agent_router(
        query
    )


    # --------------------------------------------------------
    # STEP 2 : EXECUTE TOOL
    # --------------------------------------------------------

    tool_result = execute_tool(
        selected_tool,
        query
    )


    # --------------------------------------------------------
    # STEP 3 : FORMAT RESPONSE
    # --------------------------------------------------------

    response = format_agent_response({

        "selected_tool":
        selected_tool,

        "tool_result":
        tool_result

    })


    # --------------------------------------------------------
    # STEP 4 : SAVE CONVERSATION
    # --------------------------------------------------------

    add_to_memory(
        query,
        response
    )


    # --------------------------------------------------------
    # STEP 5 : RETURN RESULT
    # --------------------------------------------------------

    return {

        "query":
        query,

        "tool":
        selected_tool,

        "response":
        response

    }

# ============================================================
# SECTION 3 : CONVERSATION CONTEXT
# ============================================================

def display_conversation_history():

    if len(conversation_history) == 0:

        print(
            "No conversation history yet."
        )

        return


    print()
    print("=" * 60)
    print("CONVERSATION HISTORY")
    print("=" * 60)


    for index, message in enumerate(
        conversation_history
    ):

        print()

        print(
            f"Conversation {index + 1}"
        )

        print(
            "User:",
            message["user"]
        )

        print(
            "Assistant:",
            message["assistant"]
        )

  # ============================================================
# SECTION 4 : INTERACTIVE CHATBOT
# ============================================================

def run_agentic_chatbot():

    print()
    print("=" * 60)
    print("       DAY 54 — AGENTIC VECTOR CHATBOT")
    print("=" * 60)

    print()
    print(
        "Ask me questions about the AI knowledge base."
    )

    print()
    print("Available commands:")

    print(
        "  history  → Show conversation history"
    )

    print(
        "  stats    → Show vector database statistics"
    )

    print(
        "  clear    → Clear conversation history"
    )

    print(
        "  exit     → Exit chatbot"
    )

    print()


    while True:

        query = input(
            "You: "
        )


        query = query.strip()


        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if query.lower() == "exit":

            print()

            print(
                "Agentic chatbot stopped."
            )

            break


        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if query == "":

            print(
                "Please enter a question."
            )

            continue


        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        if query.lower() == "history":

            display_conversation_history()

            continue


        # ----------------------------------------------------
        # CLEAR MEMORY
        # ----------------------------------------------------

        if query.lower() == "clear":

            conversation_history.clear()

            print(
                "Conversation history cleared."
            )

            continue


        # ----------------------------------------------------
        # VECTOR DATABASE STATISTICS
        # ----------------------------------------------------

        if query.lower() == "stats":

            print()
            print("=" * 50)

            print(
                "VECTOR DATABASE STATISTICS"
            )

            print("=" * 50)

            print(
                "Documents:",
                len(vector_store.documents)
            )

            print(
                "Vectors:",
                len(vector_store.vectors)
            )

            print(
                "Vector dimensions:",
                vector_store.vectors.shape[1]
            )

            continue


        # ----------------------------------------------------
        # RUN AGENT
        # ----------------------------------------------------

        result = complete_agent(
            query
        )


        # ----------------------------------------------------
        # DISPLAY RESPONSE
        # ----------------------------------------------------

        print()

        print(
            "Agent:"
        )

        print(
            result["response"]
        )


        # ----------------------------------------------------
        # SHOW TOOL
        # ----------------------------------------------------

        print()

        print(
            "Tool Used:",
            result["tool"]
        )

        print()

  # ============================================================
# SECTION 5 : COMPLETE SYSTEM TEST
# ============================================================

test_queries = [

    "What is artificial intelligence?",

    "Explain RAG",

    "What is a vector database?",

    "What is an AI agent?",

    "What is LangChain?",

    "25 * 8",

    "100 / 4",

    "Hello"

]


print()
print("=" * 60)
print("COMPLETE AGENTIC SYSTEM TEST")
print("=" * 60)


for query in test_queries:

    result = complete_agent(
        query
    )


    print()
    print("-" * 60)

    print(
        "USER:"
    )

    print(
        query
    )

    print()

    print(
        "TOOL SELECTED:"
    )

    print(
        result["tool"]
    )

    print()

    print(
        "AGENT RESPONSE:"
    )

    print(
        result["response"]
    )

# ============================================================
# SECTION 6 : AGENT DECISION TRACE
# ============================================================

def show_agent_trace(
    query
):

    print()
    print("=" * 60)
    print("AGENT DECISION TRACE")
    print("=" * 60)


    # STEP 1

    print()
    print(
        "STEP 1 — USER QUERY"
    )

    print(
        query
    )


    # STEP 2

    selected_tool = agent_router(
        query
    )


    print()
    print(
        "STEP 2 — AGENT DECISION"
    )

    print(
        "Selected:",
        selected_tool
    )


    # STEP 3

    tool_result = execute_tool(
        selected_tool,
        query
    )


    print()
    print(
        "STEP 3 — TOOL EXECUTION"
    )

    print(
        tool_result
    )


    # STEP 4

    response = format_agent_response({

        "selected_tool":
        selected_tool,

        "tool_result":
        tool_result

    })


    print()
    print(
        "STEP 4 — FINAL RESPONSE"
    )

    print(
        response
    )


    return response

# ============================================================
# SECTION 7 : AGENT BEHAVIOR TEST
# ============================================================

agent_tests = {

    "Knowledge Question":
        "What is deep learning?",

    "Vector Question":
        "Explain vector databases",

    "Agent Question":
        "What is an AI agent?",

    "Calculation":
        "125 * 8",

    "Division":
        "144 / 12",

    "General":
        "Hello"

}


print()
print("=" * 60)
print("AGENT BEHAVIOR TEST")
print("=" * 60)


for test_name, query in agent_tests.items():

    result = complete_agent(
        query
    )


    print()
    print(
        test_name
    )

    print(
        "Query:",
        query
    )

    print(
        "Tool:",
        result["tool"]
    )

    print(
        "Response:",
        result["response"]
    )

# ============================================================
# SECTION 8 : FINAL PROJECT STATISTICS
# ============================================================

print()
print("=" * 60)
print("FINAL PROJECT STATISTICS")
print("=" * 60)


print(
    "Knowledge Documents:",
    len(vector_store.documents)
)


print(
    "Stored Vectors:",
    len(vector_store.vectors)
)


print(
    "Vector Dimensions:",
    vector_store.vectors.shape[1]
)


print(
    "Available Tools:",
    len(TOOLS)
)


print(
    "Conversation Messages:",
    len(conversation_history)
)


print()
print("Tools:")

for tool_name in TOOLS:

    print(
        "→",
        tool_name
    )

# ============================================================
# SECTION 9 : FINAL ARCHITECTURE
# ============================================================

print("""
============================================================
        DAY 54 — AGENTIC VECTOR CHATBOT
============================================================

                       USER
                         |
                         v
                  USER QUERY
                         |
                         v
                  AGENT ROUTER
                         |
            +------------+------------+
            |            |            |
            v            v            v
       KNOWLEDGE     CALCULATOR    GENERAL
         SEARCH         TOOL         TOOL
            |            |            |
            +------------+------------+
                         |
                         v
                    TOOL RESULT
                         |
                         v
                RESPONSE FORMATTER
                         |
                         v
                  FINAL RESPONSE
                         |
                         v
                       USER


COMPONENTS
------------------------------------------------------------

Knowledge Base
      ↓
Document Processing
      ↓
TF-IDF Embeddings
      ↓
Vector Store
      ↓
Retriever
      ↓
Tools
      ↓
Agent Router
      ↓
Tool Execution
      ↓
Response
      ↓
Conversation Memory

============================================================
""")

# ============================================================
# SECTION 10 : PROJECT COMPLETION
# ============================================================

print()
print("=" * 60)
print("DAY 54/100 — PROJECT COMPLETED")
print("=" * 60)


print("""
PROJECT:
Agentic Vector Chatbot

PART 1:
✔ Knowledge Base
✔ LangChain Documents
✔ Lightweight Embeddings
✔ Vector Store
✔ Similarity Search

PART 2:
✔ Knowledge Search Tool
✔ Calculator Tool
✔ General Tool
✔ Tool Registry
✔ Tool Executor

PART 3:
✔ Agent Router
✔ Query Classification
✔ Tool Selection
✔ Tool Execution
✔ Response Formatter
✔ Agent Trace

PART 4:
✔ Complete Agent
✔ Conversation Memory
✔ Interactive Chatbot
✔ History
✔ Clear Memory
✔ Vector Database Statistics
✔ Agent Decision Trace
✔ Complete System Testing

HARDWARE:
✔ No large LLM
✔ No external dataset
✔ No GPU
✔ Very low storage
✔ CPU friendly

DAY 54/100 COMPLETE ✔
""")
# Start the chatbot
run_agentic_chatbot()
You: What is RAG?

You: What is an AI agent?

You: 25 * 8

You: What is a vector database?

You: stats

You: history

You: clear

You: exit
