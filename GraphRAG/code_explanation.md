# ============================================================
# DAY 55/100
# PROJECT: GRAPHRAG
# PART 1: KNOWLEDGE BASE + ENTITY/RELATIONSHIP EXTRACTION
#
# HARDWARE OPTIMIZED:
# - SLOW CPU
# - VERY LIMITED STORAGE
# - NO GPU
# - NO EXTERNAL DATASET
# - NO LARGE MODEL
# ============================================================

from langchain_core.documents import Document
import re

print("Libraries Imported Successfully")
print("Running on CPU")
print("GraphRAG Part 1 Started")


# ============================================================
# CREATE PREDEFINED KNOWLEDGE BASE
# ============================================================

documents = [

    Document(
        page_content="""
        Artificial Intelligence is a field of computer science
        focused on creating systems that can perform tasks
        requiring human-like intelligence. Machine Learning is
        a major subset of Artificial Intelligence.
        """,
        metadata={
            "topic": "Artificial Intelligence"
        }
    ),

    Document(
        page_content="""
        Machine Learning is a subset of Artificial Intelligence.
        Machine Learning algorithms learn patterns from data and
        use those patterns to make predictions or decisions.
        Deep Learning is a specialized area of Machine Learning.
        """,
        metadata={
            "topic": "Machine Learning"
        }
    ),

    Document(
        page_content="""
        Deep Learning is a branch of Machine Learning that uses
        neural networks with multiple layers. Deep Learning is
        widely used in computer vision, natural language
        processing, speech recognition, and generative AI.
        """,
        metadata={
            "topic": "Deep Learning"
        }
    ),

    Document(
        page_content="""
        Large Language Models are deep learning models trained
        on large collections of text. Large Language Models can
        perform text generation, question answering,
        summarization, reasoning, and code generation.
        """,
        metadata={
            "topic": "Large Language Models"
        }
    ),

    Document(
        page_content="""
        Retrieval-Augmented Generation combines information
        retrieval with Large Language Models. RAG retrieves
        relevant information from an external knowledge base
        and provides that information to a language model.
        """,
        metadata={
            "topic": "RAG"
        }
    ),

    Document(
        page_content="""
        Vector databases store numerical representations called
        vectors. Vector databases are commonly used for semantic
        search and Retrieval-Augmented Generation applications.
        """,
        metadata={
            "topic": "Vector Database"
        }
    ),

    Document(
        page_content="""
        LangChain is a framework used to build applications
        powered by language models. LangChain provides components
        for prompts, documents, retrieval, tools, agents, and
        vector stores.
        """,
        metadata={
            "topic": "LangChain"
        }
    ),

    Document(
        page_content="""
        AI Agents are systems that can select tools, perform
        actions, observe results, and continue processing a task.
        LangChain can be used to build agentic applications.
        """,
        metadata={
            "topic": "AI Agents"
        }
    ),

    Document(
        page_content="""
        GraphRAG uses a knowledge graph together with retrieval
        techniques. A knowledge graph represents entities as
        nodes and relationships between entities as edges.
        GraphRAG can retrieve connected information from the
        graph to improve contextual understanding.
        """,
        metadata={
            "topic": "GraphRAG"
        }
    )

]


print()
print("Knowledge Base Created")
print("Number of Documents:", len(documents))


# ============================================================
# DEFINE ENTITIES
# ============================================================

ENTITY_LIST = [

    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Large Language Models",
    "RAG",
    "Vector Database",
    "LangChain",
    "AI Agents",
    "GraphRAG",
    "Computer Vision",
    "Natural Language Processing",
    "Speech Recognition",
    "Generative AI",
    "Knowledge Graph"

]


print()
print("Entities Created")
print("Number of Entities:", len(ENTITY_LIST))


# ============================================================
# ENTITY EXTRACTION FUNCTION
# ============================================================

def extract_entities(text):

    found_entities = []

    text_lower = text.lower()

    for entity in ENTITY_LIST:

        if entity.lower() in text_lower:

            found_entities.append(entity)

    return found_entities


# ============================================================
# EXTRACT ENTITIES FROM DOCUMENTS
# ============================================================

document_entity_map = {}

for index, document in enumerate(documents):

    entities = extract_entities(
        document.page_content
    )

    document_entity_map[index] = entities


print()
print("Entity Extraction Completed")


for document_index, entities in document_entity_map.items():

    print()
    print(
        "Document:",
        document_index + 1
    )

    print(
        "Topic:",
        documents[
            document_index
        ].metadata["topic"]
    )

    print(
        "Entities:",
        entities
    )


# ============================================================
# DEFINE KNOWLEDGE GRAPH RELATIONSHIPS
# ============================================================

RELATIONSHIPS = [

    (
        "Artificial Intelligence",
        "HAS_SUBSET",
        "Machine Learning"
    ),

    (
        "Machine Learning",
        "HAS_SPECIALIZATION",
        "Deep Learning"
    ),

    (
        "Deep Learning",
        "USED_FOR",
        "Computer Vision"
    ),

    (
        "Deep Learning",
        "USED_FOR",
        "Natural Language Processing"
    ),

    (
        "Deep Learning",
        "USED_FOR",
        "Speech Recognition"
    ),

    (
        "Deep Learning",
        "USED_FOR",
        "Generative AI"
    ),

    (
        "Deep Learning",
        "SUPPORTS",
        "Large Language Models"
    ),

    (
        "RAG",
        "USES",
        "Large Language Models"
    ),

    (
        "RAG",
        "RETRIEVES_FROM",
        "Vector Database"
    ),

    (
        "RAG",
        "RELATED_TO",
        "Knowledge Graph"
    ),

    (
        "LangChain",
        "BUILDS",
        "RAG"
    ),

    (
        "LangChain",
        "BUILDS",
        "AI Agents"
    ),

    (
        "GraphRAG",
        "USES",
        "Knowledge Graph"
    ),

    (
        "GraphRAG",
        "USES",
        "RAG"
    ),

    (
        "GraphRAG",
        "RETRIEVES",
        "Knowledge Graph"
    ),

    (
        "AI Agents",
        "CAN_USE",
        "LangChain"
    )

]


print()
print("Relationships Created")
print(
    "Number of Relationships:",
    len(RELATIONSHIPS)
)


for source, relation, target in RELATIONSHIPS:

    print(
        source,
        "--[",
        relation,
        "]-->",
        target
    )


# ============================================================
# CREATE ENTITY RECORDS
# ============================================================

entity_records = []

for entity in ENTITY_LIST:

    entity_records.append({

        "name": entity,

        "type": "concept"

    })


print()
print("Entity Records Created")
print(
    "Total Entity Records:",
    len(entity_records)
)


# ============================================================
# CREATE RELATIONSHIP RECORDS
# ============================================================

relationship_records = []

for source, relation, target in RELATIONSHIPS:

    relationship_records.append({

        "source": source,

        "relationship": relation,

        "target": target

    })


print()
print("Relationship Records Created")
print(
    "Total Relationship Records:",
    len(relationship_records)
)


# ============================================================
# BUILD LIGHTWEIGHT KNOWLEDGE GRAPH
# ============================================================

graph = {}

for entity in ENTITY_LIST:

    graph[entity] = []


for source, relation, target in RELATIONSHIPS:

    graph[source].append({

        "relationship": relation,

        "target": target

    })


print()
print("Lightweight Knowledge Graph Created")

print(
    "Number of Graph Nodes:",
    len(graph)
)

print(
    "Number of Graph Edges:",
    len(RELATIONSHIPS)
)


# ============================================================
# DISPLAY KNOWLEDGE GRAPH
# ============================================================

print()
print("=" * 60)
print("KNOWLEDGE GRAPH")
print("=" * 60)

for entity, connections in graph.items():

    print()

    print(entity)

    if len(connections) == 0:

        print(
            "  No outgoing relationships"
        )

    else:

        for connection in connections:

            print(
                "  --[",
                connection["relationship"],
                "]-->",
                connection["target"]
            )


# ============================================================
# FIND NEIGHBORS OF AN ENTITY
# ============================================================

def get_neighbors(entity):

    if entity not in graph:

        return []

    return graph[entity]


# ============================================================
# TEST NEIGHBOR SEARCH
# ============================================================

test_entity = "Machine Learning"

neighbors = get_neighbors(
    test_entity
)


print()
print("=" * 60)
print("NEIGHBOR SEARCH TEST")
print("=" * 60)

print(
    "Entity:",
    test_entity
)


for neighbor in neighbors:

    print(
        "--[",
        neighbor["relationship"],
        "]-->",
        neighbor["target"]
    )


# ============================================================
# FIND ENTITIES IN USER QUERY
# ============================================================

def find_query_entities(query):

    found_entities = []

    query_lower = query.lower()

    for entity in ENTITY_LIST:

        if entity.lower() in query_lower:

            found_entities.append(entity)

    return found_entities


# ============================================================
# TEST QUERY ENTITY DETECTION
# ============================================================

test_query = (
    "How is Deep Learning related to Machine Learning?"
)

query_entities = find_query_entities(
    test_query
)


print()
print("=" * 60)
print("QUERY ENTITY DETECTION")
print("=" * 60)

print(
    "Query:",
    test_query
)

print(
    "Detected Entities:",
    query_entities
)


# ============================================================
# CREATE DOCUMENT + ENTITY + RELATIONSHIP SUMMARY
# ============================================================

print()
print("=" * 60)
print("GRAPHRAG KNOWLEDGE BASE SUMMARY")
print("=" * 60)

print()
print(
    "Documents:",
    len(documents)
)

print(
    "Entities:",
    len(ENTITY_LIST)
)

print(
    "Relationships:",
    len(RELATIONSHIPS)
)

print(
    "Graph Nodes:",
    len(graph)
)

print(
    "Graph Edges:",
    len(RELATIONSHIPS)
)


# ============================================================
# PART 1 VALIDATION
# ============================================================

print()
print("=" * 60)
print("DAY 55 — PART 1 COMPLETED")
print("=" * 60)

print()
print("✔ Predefined knowledge base created")
print("✔ LangChain Documents created")
print("✔ Entities defined")
print("✔ Entities extracted from documents")
print("✔ Relationships defined")
print("✔ Entity records created")
print("✔ Relationship records created")
print("✔ Document-to-entity mapping created")
print("✔ Lightweight knowledge graph created")
print("✔ Neighbor search implemented")
print("✔ Query entity detection implemented")

print()
print("HARDWARE OPTIMIZATION")
print("✔ No external dataset")
print("✔ No large NLP model")
print("✔ No embedding model")
print("✔ No GPU")
print("✔ No Neo4j")
print("✔ No persistent database")
print("✔ Minimal storage")
print("✔ CPU friendly")

print()
print("DAY 55 PART 1 COMPLETE ✔")
print()
print("NEXT → PART 2: GRAPH TRAVERSAL + GRAPH RETRIEVAL")
# ============================================================
# DAY 55/100
# PROJECT: GRAPHRAG
# PART 2: GRAPH TRAVERSAL + GRAPH RETRIEVAL
#
# CONTINUES FROM PART 1
#
# HARDWARE:
# - SLOW CPU
# - VERY LIMITED STORAGE
# - NO GPU
# - NO EXTERNAL DATASET
# - NO LARGE MODEL
# ============================================================


print("=" * 60)
print("DAY 55/100 — GRAPHRAG PART 2")
print("GRAPH TRAVERSAL + GRAPH RETRIEVAL")
print("=" * 60)


# ============================================================
# BASIC GRAPH INFORMATION
# ============================================================

print()
print("Graph Nodes:", len(graph))
print("Graph Edges:", len(RELATIONSHIPS))


# ============================================================
# CREATE REVERSE GRAPH
#
# The graph from Part 1 contains outgoing relationships.
# For GraphRAG, we also want to know which entities point
# TO a particular entity.
# ============================================================

reverse_graph = {}

for entity in ENTITY_LIST:

    reverse_graph[entity] = []


for source, relation, target in RELATIONSHIPS:

    reverse_graph[target].append({

        "source": source,

        "relationship": relation

    })


# ============================================================
# FUNCTION TO GET OUTGOING NEIGHBORS
# ============================================================

def get_outgoing_neighbors(entity):

    if entity not in graph:

        return []

    return graph[entity]


# ============================================================
# FUNCTION TO GET INCOMING NEIGHBORS
# ============================================================

def get_incoming_neighbors(entity):

    if entity not in reverse_graph:

        return []

    return reverse_graph[entity]


# ============================================================
# FUNCTION TO GET ALL DIRECT CONNECTIONS
#
# This combines both:
# - outgoing relationships
# - incoming relationships
# ============================================================

def get_direct_connections(entity):

    connections = []


    # --------------------------------------------------------
    # OUTGOING CONNECTIONS
    # --------------------------------------------------------

    for connection in get_outgoing_neighbors(entity):

        connections.append({

            "entity": connection["target"],

            "relationship": connection["relationship"],

            "direction": "outgoing"

        })


    # --------------------------------------------------------
    # INCOMING CONNECTIONS
    # --------------------------------------------------------

    for connection in get_incoming_neighbors(entity):

        connections.append({

            "entity": connection["source"],

            "relationship": connection["relationship"],

            "direction": "incoming"

        })


    return connections


# ============================================================
# TEST DIRECT CONNECTIONS
# ============================================================

test_entity = "Deep Learning"

connections = get_direct_connections(
    test_entity
)


print()
print("=" * 60)
print("DIRECT CONNECTION TEST")
print("=" * 60)

print()
print("Entity:", test_entity)

for connection in connections:

    print(
        connection["direction"],
        "--[",
        connection["relationship"],
        "]-->",
        connection["entity"]
    )


# ============================================================
# GRAPH TRAVERSAL
#
# We will traverse the graph up to a specified number of
# hops.
#
# Example:
#
# Deep Learning
#      ↓
# Computer Vision
#
# Deep Learning
#      ↓
# Machine Learning
#      ↓
# Artificial Intelligence
# ============================================================

def traverse_graph(
    start_entity,
    max_hops=2
):

    if start_entity not in ENTITY_LIST:

        return []


    visited = set()

    results = []

    current_level = [
        start_entity
    ]

    visited.add(
        start_entity
    )


    for hop in range(
        1,
        max_hops + 1
    ):

        next_level = []


        for current_entity in current_level:

            connections = get_direct_connections(
                current_entity
            )


            for connection in connections:

                connected_entity = connection[
                    "entity"
                ]


                if connected_entity not in visited:

                    visited.add(
                        connected_entity
                    )


                    next_level.append(
                        connected_entity
                    )


                    results.append({

                        "entity":
                        connected_entity,

                        "relationship":
                        connection[
                            "relationship"
                        ],

                        "direction":
                        connection[
                            "direction"
                        ],

                        "hop":
                        hop

                    })


        current_level = next_level


        if len(current_level) == 0:

            break


    return results


# ============================================================
# TEST GRAPH TRAVERSAL
# ============================================================

start_entity = "Deep Learning"

traversal_results = traverse_graph(
    start_entity,
    max_hops=2
)


print()
print("=" * 60)
print("GRAPH TRAVERSAL TEST")
print("=" * 60)

print()
print(
    "Starting Entity:",
    start_entity
)

print(
    "Maximum Hops:",
    2
)


for result in traversal_results:

    print()

    print(
        "Hop:",
        result["hop"]
    )

    print(
        "Entity:",
        result["entity"]
    )

    print(
        "Relationship:",
        result["relationship"]
    )

    print(
        "Direction:",
        result["direction"]
    )


# ============================================================
# FIND RELEVANT DOCUMENTS FOR AN ENTITY
# ============================================================

def get_documents_for_entity(
    entity
):

    matching_documents = []


    for document_index, entities in document_entity_map.items():

        if entity in entities:

            matching_documents.append(
                document_index
            )


    return matching_documents


# ============================================================
# TEST ENTITY → DOCUMENT RETRIEVAL
# ============================================================

test_entity = "RAG"

matching_documents = get_documents_for_entity(
    test_entity
)


print()
print("=" * 60)
print("ENTITY → DOCUMENT RETRIEVAL")
print("=" * 60)

print()
print(
    "Entity:",
    test_entity
)

print(
    "Matching Document IDs:",
    [
        index + 1
        for index in matching_documents
    ]
)


# ============================================================
# CREATE GRAPH CONTEXT
#
# This converts graph relationships into readable context.
# ============================================================

def create_graph_context(
    entity,
    max_hops=2
):

    traversal_results = traverse_graph(
        entity,
        max_hops=max_hops
    )


    context_lines = []


    context_lines.append(
        f"Main Entity: {entity}"
    )


    for result in traversal_results:

        if result["direction"] == "outgoing":

            line = (
                f"{entity} "
                f"--[{result['relationship']}]--> "
                f"{result['entity']}"
            )

        else:

            line = (
                f"{result['entity']} "
                f"--[{result['relationship']}]--> "
                f"{entity}"
            )


        context_lines.append(
            line
        )


    return "\n".join(
        context_lines
    )


# ============================================================
# TEST GRAPH CONTEXT
# ============================================================

graph_context = create_graph_context(
    "Deep Learning",
    max_hops=2
)


print()
print("=" * 60)
print("GRAPH CONTEXT")
print("=" * 60)

print()
print(graph_context)


# ============================================================
# FIND QUERY ENTITIES
#
# This function was created in Part 1.
# We use it here to identify entities mentioned by the user.
# ============================================================

def get_query_entities(
    query
):

    return find_query_entities(
        query
    )


# ============================================================
# GRAPH RETRIEVAL FUNCTION
#
# This is the main GraphRAG retrieval component.
#
# USER QUERY
#      ↓
# ENTITY DETECTION
#      ↓
# GRAPH TRAVERSAL
#      ↓
# RELATED ENTITIES
#      ↓
# DOCUMENT RETRIEVAL
#      ↓
# GRAPH CONTEXT
# ============================================================

def graph_retrieve(
    query,
    max_hops=2
):

    query_entities = get_query_entities(
        query
    )


    # --------------------------------------------------------
    # NO ENTITY FOUND
    # --------------------------------------------------------

    if len(query_entities) == 0:

        return {

            "query": query,

            "entities": [],

            "related_entities": [],

            "documents": [],

            "context": "",

            "found": False

        }


    related_entities = []

    document_indices = []

    context_parts = []


    # --------------------------------------------------------
    # PROCESS EACH QUERY ENTITY
    # --------------------------------------------------------

    for entity in query_entities:

        # --------------------------------------------
        # CREATE GRAPH CONTEXT
        # --------------------------------------------

        entity_context = create_graph_context(
            entity,
            max_hops=max_hops
        )


        context_parts.append(
            entity_context
        )


        # --------------------------------------------
        # GRAPH TRAVERSAL
        # --------------------------------------------

        traversal_results = traverse_graph(
            entity,
            max_hops=max_hops
        )


        for result in traversal_results:

            related_entity = result[
                "entity"
            ]


            if related_entity not in related_entities:

                related_entities.append(
                    related_entity
                )


        # --------------------------------------------
        # GET DOCUMENTS FOR MAIN ENTITY
        # --------------------------------------------

        entity_documents = get_documents_for_entity(
            entity
        )


        for document_index in entity_documents:

            if document_index not in document_indices:

                document_indices.append(
                    document_index
                )


        # --------------------------------------------
        # GET DOCUMENTS FOR RELATED ENTITIES
        # --------------------------------------------

        for related_entity in related_entities:

            related_documents = get_documents_for_entity(
                related_entity
            )


            for document_index in related_documents:

                if document_index not in document_indices:

                    document_indices.append(
                        document_index
                    )


    # --------------------------------------------------------
    # COMBINE CONTEXT
    # --------------------------------------------------------

    combined_context = "\n\n".join(
        context_parts
    )


    return {

        "query": query,

        "entities": query_entities,

        "related_entities": related_entities,

        "documents": document_indices,

        "context": combined_context,

        "found": True

    }


# ============================================================
# TEST GRAPH RETRIEVAL
# ============================================================

test_query = (
    "How is Deep Learning related to Machine Learning?"
)


retrieval_result = graph_retrieve(
    test_query,
    max_hops=2
)


print()
print("=" * 60)
print("GRAPHRAG RETRIEVAL TEST")
print("=" * 60)

print()
print(
    "Query:"
)

print(
    retrieval_result["query"]
)

print()
print(
    "Detected Entities:"
)

print(
    retrieval_result["entities"]
)

print()
print(
    "Related Entities:"
)

print(
    retrieval_result["related_entities"]
)

print()
print(
    "Retrieved Documents:"
)

print(
    [
        index + 1
        for index in retrieval_result["documents"]
    ]
)

print()
print(
    "Graph Context:"
)

print(
    retrieval_result["context"]
)


# ============================================================
# RETRIEVE ACTUAL DOCUMENT CONTENT
# ============================================================

def get_document_context(
    document_indices
):

    contexts = []


    for document_index in document_indices:

        document = documents[
            document_index
        ]


        topic = document.metadata[
            "topic"
        ]


        content = document.page_content.strip()


        contexts.append(

            f"Topic: {topic}\n"
            f"Content: {content}"

        )


    return "\n\n".join(
        contexts
    )


# ============================================================
# COMPLETE GRAPHRAG RETRIEVAL
#
# Combines:
# - Graph structure
# - Graph traversal
# - Related documents
# - Document context
# ============================================================

def complete_graph_retrieval(
    query,
    max_hops=2
):

    result = graph_retrieve(
        query,
        max_hops=max_hops
    )


    if not result["found"]:

        return {

            "query": query,

            "entities": [],

            "related_entities": [],

            "documents": [],

            "graph_context": "",

            "document_context": "",

            "found": False

        }


    document_context = get_document_context(
        result["documents"]
    )


    return {

        "query":
        query,

        "entities":
        result["entities"],

        "related_entities":
        result["related_entities"],

        "documents":
        result["documents"],

        "graph_context":
        result["context"],

        "document_context":
        document_context,

        "found":
        True

    }


# ============================================================
# TEST COMPLETE GRAPHRAG RETRIEVAL
# ============================================================

test_queries = [

    "What is RAG?",

    "What is Deep Learning?",

    "How is Deep Learning related to Machine Learning?",

    "What does RAG use?",

    "How is LangChain related to AI Agents?",

    "What is GraphRAG?"

]


print()
print("=" * 60)
print("COMPLETE GRAPHRAG RETRIEVAL TEST")
print("=" * 60)


for query in test_queries:

    result = complete_graph_retrieval(
        query,
        max_hops=2
    )


    print()
    print("-" * 60)

    print(
        "QUERY:"
    )

    print(
        query
    )


    print()

    print(
        "ENTITIES:"
    )

    print(
        result["entities"]
    )


    print()

    print(
        "RELATED ENTITIES:"
    )

    print(
        result["related_entities"]
    )


    print()

    print(
        "DOCUMENTS:"
    )

    print(
        [
            index + 1
            for index in result["documents"]
        ]
    )


    print()

    print(
        "GRAPH CONTEXT:"
    )

    print(
        result["graph_context"]
    )


# ============================================================
# SIMPLE GRAPH RETRIEVAL SCORE
#
# We give closer entities a higher score.
# This is lightweight and does not require embeddings.
# ============================================================

def calculate_graph_score(
    hop
):

    if hop == 1:

        return 1.0

    if hop == 2:

        return 0.5

    if hop == 3:

        return 0.25

    return 0.1


# ============================================================
# RANK GRAPH RESULTS
# ============================================================

def ranked_graph_retrieval(
    query,
    max_hops=2
):

    query_entities = find_query_entities(
        query
    )


    ranked_results = []


    for entity in query_entities:

        traversal_results = traverse_graph(
            entity,
            max_hops=max_hops
        )


        # Main entity gets highest relevance

        ranked_results.append({

            "entity": entity,

            "hop": 0,

            "score": 1.0

        })


        for result in traversal_results:

            ranked_results.append({

                "entity":
                result["entity"],

                "hop":
                result["hop"],

                "score":
                calculate_graph_score(
                    result["hop"]
                )

            })


    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    unique_results = {}


    for result in ranked_results:

        entity = result[
            "entity"
        ]


        if (
            entity not in unique_results
            or
            result["score"]
            >
            unique_results[entity]["score"]
        ):

            unique_results[entity] = result


    # --------------------------------------------------------
    # SORT BY SCORE
    # --------------------------------------------------------

    final_results = sorted(

        unique_results.values(),

        key=lambda item:
        item["score"],

        reverse=True

    )


    return final_results


# ============================================================
# TEST RANKING
# ============================================================

ranking_query = (
    "How is Deep Learning related to Machine Learning?"
)


ranked_results = ranked_graph_retrieval(
    ranking_query,
    max_hops=2
)


print()
print("=" * 60)
print("RANKED GRAPH RETRIEVAL")
print("=" * 60)

print()
print(
    "Query:",
    ranking_query
)


for result in ranked_results:

    print()

    print(
        "Entity:",
        result["entity"]
    )

    print(
        "Hop:",
        result["hop"]
    )

    print(
        "Score:",
        result["score"]
    )


# ============================================================
# FINAL GRAPH RETRIEVAL FUNCTION
# ============================================================

def final_graph_retriever(
    query,
    max_hops=2
):

    results = ranked_graph_retrieval(
        query,
        max_hops=max_hops
    )


    if len(results) == 0:

        return {

            "query": query,

            "results": [],

            "context": "",

            "found": False

        }


    context_parts = []


    for result in results:

        entity = result[
            "entity"
        ]


        # --------------------------------------------
        # ENTITY INFORMATION
        # --------------------------------------------

        entity_context = create_graph_context(
            entity,
            max_hops=1
        )


        context_parts.append(
            entity_context
        )


        # --------------------------------------------
        # DOCUMENT INFORMATION
        # --------------------------------------------

        document_indices = get_documents_for_entity(
            entity
        )


        if len(document_indices) > 0:

            document_context = get_document_context(
                document_indices
            )


            context_parts.append(
                document_context
            )


    final_context = "\n\n".join(
        context_parts
    )


    return {

        "query":
        query,

        "results":
        results,

        "context":
        final_context,

        "found":
        True

    }


# ============================================================
# FINAL TEST
# ============================================================

final_query = (
    "Explain the relationship between "
    "RAG and Vector Database"
)


final_result = final_graph_retriever(
    final_query,
    max_hops=2
)


print()
print("=" * 60)
print("FINAL GRAPHRAG RETRIEVER TEST")
print("=" * 60)

print()
print(
    "USER QUERY:"
)

print(
    final_query
)

print()
print(
    "RETRIEVED ENTITIES:"
)

for result in final_result["results"]:

    print(
        result["entity"],
        "→ score:",
        result["score"]
    )

print()
print(
    "RETRIEVED CONTEXT:"
)

print(
    final_result["context"]
)


# ============================================================
# PART 2 VALIDATION
# ============================================================

print()
print("=" * 60)
print("DAY 55 — PART 2 COMPLETED")
print("=" * 60)

print()
print("✔ Reverse graph created")
print("✔ Incoming relationships implemented")
print("✔ Outgoing relationships implemented")
print("✔ Direct graph connections implemented")
print("✔ Multi-hop graph traversal implemented")
print("✔ Entity-to-document retrieval implemented")
print("✔ Graph context generation implemented")
print("✔ Query entity detection connected")
print("✔ GraphRAG retrieval implemented")
print("✔ Graph results ranked")
print("✔ Lightweight relevance scoring implemented")
print("✔ Complete graph retriever implemented")

print()
print("HARDWARE OPTIMIZATION")
print("✔ No large model")
print("✔ No embedding model")
print("✔ No GPU")
print("✔ No Neo4j")
print("✔ No external dataset")
print("✔ Minimal storage")
print("✔ CPU friendly")

print()
print("DAY 55 PART 2 COMPLETE ✔")
print()
print("NEXT → PART 3: GRAPH CONTEXT + ANSWER GENERATION")
# ============================================================
# DAY 55/100
# PROJECT: GRAPHRAG
# PART 3: GRAPH CONTEXT + ANSWER GENERATION
#
# CONTINUES FROM:
# PART 1 → Knowledge Graph
# PART 2 → Graph Traversal + Graph Retrieval
#
# HARDWARE:
# - SLOW CPU
# - VERY LIMITED STORAGE
# - NO GPU
# - NO LARGE LLM
# - NO EXTERNAL DATASET
# ============================================================


print("=" * 60)
print("DAY 55/100 — GRAPHRAG PART 3")
print("GRAPH CONTEXT + ANSWER GENERATION")
print("=" * 60)


# ============================================================
# CHECK PART 1 AND PART 2
# ============================================================

required_variables = [

    "documents",
    "ENTITY_LIST",
    "RELATIONSHIPS",
    "graph",
    "reverse_graph",
    "document_entity_map"
]


missing_variables = []


for variable in required_variables:

    if variable not in globals():

        missing_variables.append(
            variable
        )


if len(missing_variables) > 0:

    print()

    print(
        "Missing variables:"
    )

    print(
        missing_variables
    )

    print()

    print(
        "Please run Part 1 and Part 2 first."
    )

else:

    print()
    print(
        "Part 1 and Part 2 data detected successfully."
    )


# ============================================================
# GRAPH CONTEXT BUILDER
# ============================================================

def build_entity_context(
    entity,
    max_hops=1
):

    if entity not in ENTITY_LIST:

        return ""


    context_lines = []


    context_lines.append(
        "ENTITY: " + entity
    )


    # --------------------------------------------------------
    # DIRECT OUTGOING RELATIONSHIPS
    # --------------------------------------------------------

    outgoing = get_outgoing_neighbors(
        entity
    )


    for connection in outgoing:

        relationship = connection[
            "relationship"
        ]

        target = connection[
            "target"
        ]


        context_lines.append(

            entity
            + " --["
            + relationship
            + "]--> "
            + target

        )


    # --------------------------------------------------------
    # DIRECT INCOMING RELATIONSHIPS
    # --------------------------------------------------------

    incoming = get_incoming_neighbors(
        entity
    )


    for connection in incoming:

        relationship = connection[
            "relationship"
        ]

        source = connection[
            "source"
        ]


        context_lines.append(

            source
            + " --["
            + relationship
            + "]--> "
            + entity

        )


    # --------------------------------------------------------
    # MULTI-HOP RELATIONSHIPS
    # --------------------------------------------------------

    if max_hops > 1:

        traversal = traverse_graph(
            entity,
            max_hops=max_hops
        )


        for result in traversal:

            related_entity = result[
                "entity"
            ]

            relationship = result[
                "relationship"
            ]

            hop = result[
                "hop"
            ]


            if hop > 1:

                context_lines.append(

                    "HOP "
                    + str(hop)
                    + ": "
                    + related_entity
                    + " via "
                    + relationship

                )


    return "\n".join(
        context_lines
    )


# ============================================================
# DOCUMENT CONTEXT BUILDER
# ============================================================

def build_document_context(
    document_indices
):

    context_lines = []


    for index in document_indices:

        if index < 0:

            continue


        if index >= len(documents):

            continue


        document = documents[
            index
        ]


        topic = document.metadata.get(
            "topic",
            "Unknown"
        )


        content = document.page_content.strip()


        context_lines.append(

            "TOPIC: "
            + topic
            + "\n"
            + content

        )


    return "\n\n".join(
        context_lines
    )


# ============================================================
# CREATE COMBINED GRAPHRAG CONTEXT
# ============================================================

def create_graphrag_context(
    query,
    max_hops=2
):

    retrieval = complete_graph_retrieval(
        query,
        max_hops=max_hops
    )


    if not retrieval["found"]:

        return {

            "query": query,

            "entities": [],

            "related_entities": [],

            "graph_context": "",

            "document_context": "",

            "combined_context": "",

            "found": False

        }


    graph_context_parts = []


    for entity in retrieval[
        "entities"
    ]:

        entity_context = build_entity_context(
            entity,
            max_hops=max_hops
        )


        if entity_context != "":

            graph_context_parts.append(
                entity_context
            )


    graph_context = "\n\n".join(
        graph_context_parts
    )


    document_context = build_document_context(
        retrieval["documents"]
    )


    combined_context = (

        "GRAPH CONTEXT\n"
        "-------------\n"
        + graph_context
        + "\n\n"
        + "DOCUMENT CONTEXT\n"
        "----------------\n"
        + document_context

    )


    return {

        "query":
        query,

        "entities":
        retrieval["entities"],

        "related_entities":
        retrieval["related_entities"],

        "graph_context":
        graph_context,

        "document_context":
        document_context,

        "combined_context":
        combined_context,

        "found":
        True

    }


# ============================================================
# TEST GRAPHRAG CONTEXT
# ============================================================

test_query = (
    "How is Deep Learning related to Machine Learning?"
)


context_result = create_graphrag_context(
    test_query,
    max_hops=2
)


print()
print("=" * 60)
print("GRAPHRAG CONTEXT TEST")
print("=" * 60)

print()

print(
    "Query:"
)

print(
    test_query
)

print()

print(
    "Detected Entities:"
)

print(
    context_result["entities"]
)

print()

print(
    "Related Entities:"
)

print(
    context_result["related_entities"]
)

print()

print(
    "Combined Context:"
)

print(
    context_result["combined_context"]
)


# ============================================================
# SIMPLE ANSWER GENERATOR
#
# Instead of downloading a large language model, we create a
# lightweight answer generator using the retrieved graph
# relationships and document content.
# ============================================================

def generate_lightweight_answer(
    query,
    context_result
):

    if not context_result["found"]:

        return (
            "I could not find a relevant entity "
            "in the knowledge graph for this question."
        )


    entities = context_result[
        "entities"
    ]


    related_entities = context_result[
        "related_entities"
    ]


    graph_context = context_result[
        "graph_context"
    ]


    document_context = context_result[
        "document_context"
    ]


    answer_parts = []


    # --------------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------------

    if len(entities) > 0:

        answer_parts.append(

            "The question is related to: "
            + ", ".join(entities)
            + "."

        )


    # --------------------------------------------------------
    # GRAPH RELATIONSHIP INFORMATION
    # --------------------------------------------------------

    if graph_context != "":

        answer_parts.append(

            "The knowledge graph shows the following "
            "relationships:\n"
            + graph_context

        )


    # --------------------------------------------------------
    # RELATED ENTITIES
    # --------------------------------------------------------

    if len(related_entities) > 0:

        answer_parts.append(

            "Related concepts include: "
            + ", ".join(related_entities)
            + "."

        )


    # --------------------------------------------------------
    # DOCUMENT INFORMATION
    # --------------------------------------------------------

    if document_context != "":

        answer_parts.append(

            "Relevant knowledge from the documents:\n"
            + document_context

        )


    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    return "\n\n".join(
        answer_parts
    )


# ============================================================
# COMPLETE GRAPHRAG ANSWER PIPELINE
#
# QUERY
#   ↓
# ENTITY DETECTION
#   ↓
# GRAPH TRAVERSAL
#   ↓
# RELATED ENTITIES
#   ↓
# DOCUMENT RETRIEVAL
#   ↓
# GRAPH CONTEXT
#   ↓
# ANSWER
# ============================================================

def answer_with_graphrag(
    query,
    max_hops=2
):

    context_result = create_graphrag_context(
        query,
        max_hops=max_hops
    )


    answer = generate_lightweight_answer(
        query,
        context_result
    )


    return {

        "query":
        query,

        "entities":
        context_result["entities"],

        "related_entities":
        context_result["related_entities"],

        "graph_context":
        context_result["graph_context"],

        "document_context":
        context_result["document_context"],

        "answer":
        answer,

        "found":
        context_result["found"]

    }


# ============================================================
# TEST COMPLETE GRAPHRAG ANSWER PIPELINE
# ============================================================

test_queries = [

    "What is RAG?",

    "What is Deep Learning?",

    "How is Deep Learning related to Machine Learning?",

    "What does RAG use?",

    "How is LangChain related to AI Agents?",

    "What is GraphRAG?"

]


print()
print("=" * 60)
print("COMPLETE GRAPHRAG ANSWER TEST")
print("=" * 60)


for query in test_queries:

    result = answer_with_graphrag(
        query,
        max_hops=2
    )


    print()
    print("-" * 60)

    print(
        "USER QUERY:"
    )

    print(
        query
    )

    print()

    print(
        "ENTITIES:"
    )

    print(
        result["entities"]
    )

    print()

    print(
        "RELATED ENTITIES:"
    )

    print(
        result["related_entities"]
    )

    print()

    print(
        "ANSWER:"
    )

    print(
        result["answer"]
    )


# ============================================================
# CREATE A CLEAN ANSWER
#
# This function removes unnecessary graph formatting and
# produces a cleaner chatbot-style response.
# ============================================================

def create_clean_answer(
    query,
    max_hops=2
):

    result = answer_with_graphrag(
        query,
        max_hops=max_hops
    )


    if not result["found"]:

        return (
            "I could not find enough information "
            "in the knowledge graph to answer this question."
        )


    entities = result[
        "entities"
    ]


    graph_context = result[
        "graph_context"
    ]


    document_context = result[
        "document_context"
    ]


    response = []


    if len(entities) > 0:

        response.append(

            "The query is related to "
            + ", ".join(entities)
            + "."

        )


    if graph_context != "":

        response.append(
            "Graph relationships:\n"
            + graph_context
        )


    if document_context != "":

        response.append(
            "Supporting information:\n"
            + document_context
        )


    return "\n\n".join(
        response
    )


# ============================================================
# TEST CLEAN ANSWER
# ============================================================

clean_query = (
    "What is the relationship between RAG "
    "and Vector Database?"
)


clean_response = create_clean_answer(
    clean_query,
    max_hops=2
)


print()
print("=" * 60)
print("CLEAN GRAPHRAG ANSWER")
print("=" * 60)

print()

print(
    "Question:"
)

print(
    clean_query
)

print()

print(
    "Answer:"
)

print(
    clean_response
)


# ============================================================
# GRAPHRAG PIPELINE TRACE
# ============================================================

def show_graphrag_trace(
    query
):

    print()
    print("=" * 60)
    print("GRAPHRAG PIPELINE TRACE")
    print("=" * 60)


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print()

    print(
        "STEP 1 — USER QUERY"
    )

    print(
        query
    )


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    entities = find_query_entities(
        query
    )


    print()

    print(
        "STEP 2 — ENTITY DETECTION"
    )

    print(
        entities
    )


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    all_related_entities = []


    for entity in entities:

        traversal = traverse_graph(
            entity,
            max_hops=2
        )


        for result in traversal:

            related_entity = result[
                "entity"
            ]


            if related_entity not in all_related_entities:

                all_related_entities.append(
                    related_entity
                )


    print()

    print(
        "STEP 3 — GRAPH TRAVERSAL"
    )

    print(
        all_related_entities
    )


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    result = create_graphrag_context(
        query,
        max_hops=2
    )


    print()

    print(
        "STEP 4 — CONTEXT RETRIEVAL"
    )

    print(
        result["combined_context"]
    )


    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    answer = generate_lightweight_answer(
        query,
        result
    )


    print()

    print(
        "STEP 5 — ANSWER GENERATION"
    )

    print(
        answer
    )


    return answer


# ============================================================
# TRACE TEST
# ============================================================

show_graphrag_trace(
    "How does RAG use a Vector Database?"
)


# ============================================================
# MULTI-HOP TEST
# ============================================================

multi_hop_query = (
    "How is Artificial Intelligence related "
    "to Deep Learning?"
)


multi_hop_result = answer_with_graphrag(
    multi_hop_query,
    max_hops=2
)


print()
print("=" * 60)
print("MULTI-HOP GRAPHRAG TEST")
print("=" * 60)

print()

print(
    "Query:"
)

print(
    multi_hop_query
)

print()

print(
    "Entities:"
)

print(
    multi_hop_result["entities"]
)

print()

print(
    "Related Entities:"
)

print(
    multi_hop_result["related_entities"]
)

print()

print(
    "Answer:"
)

print(
    multi_hop_result["answer"]
)


# ============================================================
# UNKNOWN QUERY TEST
# ============================================================

unknown_query = (
    "What is quantum computing?"
)


unknown_result = answer_with_graphrag(
    unknown_query,
    max_hops=2
)


print()
print("=" * 60)
print("UNKNOWN QUERY TEST")
print("=" * 60)

print()

print(
    "Query:"
)

print(
    unknown_query
)

print()

print(
    "Answer:"
)

print(
    unknown_result["answer"]
)


# ============================================================
# PART 3 VALIDATION
# ============================================================

print()
print("=" * 60)
print("DAY 55 — PART 3 COMPLETED")
print("=" * 60)

print()
print("✔ Graph context builder created")
print("✔ Entity context created")
print("✔ Document context created")
print("✔ Graph + document context combined")
print("✔ Lightweight answer generation implemented")
print("✔ Complete GraphRAG answer pipeline implemented")
print("✔ Clean answer generation implemented")
print("✔ GraphRAG pipeline trace implemented")
print("✔ Multi-hop retrieval tested")
print("✔ Unknown query handling tested")

print()
print("GRAPHRAG PIPELINE")

print("""
User Query
    ↓
Entity Detection
    ↓
Graph Traversal
    ↓
Related Entities
    ↓
Document Retrieval
    ↓
Graph Context
    ↓
Document Context
    ↓
Combined Context
    ↓
Answer Generation
    ↓
Final Answer
""")

print("HARDWARE OPTIMIZATION")
print("✔ No large LLM")
print("✔ No embedding model")
print("✔ No GPU")
print("✔ No Neo4j")
print("✔ No external dataset")
print("✔ Minimal storage")
print("✔ CPU friendly")

print()
print("DAY 55 PART 3 COMPLETE ✔")
print()
print("NEXT → PART 4: COMPLETE GRAPHRAG CHATBOT")
# ============================================================
# DAY 55/100
# PROJECT: GRAPHRAG
# PART 4: COMPLETE GRAPHRAG CHATBOT
#
# CONTINUES FROM PART 1 + PART 2 + PART 3
#
# HARDWARE:
# - SLOW CPU
# - VERY LIMITED STORAGE
# - NO GPU
# - NO LARGE MODEL
# - NO EXTERNAL DATASET
# ============================================================


print("=" * 60)
print("DAY 55/100 — GRAPHRAG")
print("PART 4 — COMPLETE GRAPHRAG CHATBOT")
print("=" * 60)


# ============================================================
# VERIFY PREVIOUS PARTS
# ============================================================

required_variables = [

    "documents",
    "ENTITY_LIST",
    "RELATIONSHIPS",
    "graph",
    "reverse_graph",
    "document_entity_map"

]


missing_variables = []


for variable in required_variables:

    if variable not in globals():

        missing_variables.append(
            variable
        )


if len(missing_variables) > 0:

    print()
    print("The following variables are missing:")
    print(missing_variables)

    print()
    print(
        "Please run Part 1, Part 2 and Part 3 "
        "before running Part 4."
    )

else:

    print()
    print("Previous parts detected successfully ✔")


# ============================================================
# COMPLETE GRAPHRAG PIPELINE
# ============================================================

def run_graphrag(
    query,
    max_hops=2
):

    # --------------------------------------------------------
    # STEP 1: ENTITY DETECTION
    # --------------------------------------------------------

    query_entities = find_query_entities(
        query
    )


    # --------------------------------------------------------
    # HANDLE UNKNOWN QUERY
    # --------------------------------------------------------

    if len(query_entities) == 0:

        return {

            "query": query,

            "entities": [],

            "related_entities": [],

            "documents": [],

            "graph_context": "",

            "document_context": "",

            "answer": (
                "I could not find a relevant entity "
                "in the knowledge graph for this question."
            ),

            "found": False

        }


    # --------------------------------------------------------
    # STEP 2: GRAPH TRAVERSAL
    # --------------------------------------------------------

    related_entities = []

    graph_context_parts = []

    document_indices = []


    for entity in query_entities:

        # ----------------------------------------------------
        # BUILD GRAPH CONTEXT
        # ----------------------------------------------------

        entity_context = build_entity_context(
            entity,
            max_hops=max_hops
        )


        if entity_context != "":

            graph_context_parts.append(
                entity_context
            )


        # ----------------------------------------------------
        # TRAVERSE GRAPH
        # ----------------------------------------------------

        traversal_results = traverse_graph(
            entity,
            max_hops=max_hops
        )


        for result in traversal_results:

            related_entity = result[
                "entity"
            ]


            if related_entity not in related_entities:

                related_entities.append(
                    related_entity
                )


    # --------------------------------------------------------
    # STEP 3: DOCUMENT RETRIEVAL
    # --------------------------------------------------------

    all_entities = []

    all_entities.extend(
        query_entities
    )

    all_entities.extend(
        related_entities
    )


    for entity in all_entities:

        entity_documents = get_documents_for_entity(
            entity
        )


        for document_index in entity_documents:

            if document_index not in document_indices:

                document_indices.append(
                    document_index
                )


    # --------------------------------------------------------
    # STEP 4: BUILD GRAPH CONTEXT
    # --------------------------------------------------------

    graph_context = "\n\n".join(
        graph_context_parts
    )


    # --------------------------------------------------------
    # STEP 5: BUILD DOCUMENT CONTEXT
    # --------------------------------------------------------

    document_context = build_document_context(
        document_indices
    )


    # --------------------------------------------------------
    # STEP 6: CREATE ANSWER
    # --------------------------------------------------------

    context_result = {

        "query":
        query,

        "entities":
        query_entities,

        "related_entities":
        related_entities,

        "graph_context":
        graph_context,

        "document_context":
        document_context,

        "found":
        True

    }


    answer = generate_lightweight_answer(
        query,
        context_result
    )


    # --------------------------------------------------------
    # RETURN COMPLETE RESULT
    # --------------------------------------------------------

    return {

        "query":
        query,

        "entities":
        query_entities,

        "related_entities":
        related_entities,

        "documents":
        document_indices,

        "graph_context":
        graph_context,

        "document_context":
        document_context,

        "answer":
        answer,

        "found":
        True

    }


# ============================================================
# CREATE A CLEAN CHATBOT RESPONSE
# ============================================================

def chatbot_response(
    query
):

    result = run_graphrag(
        query,
        max_hops=2
    )


    if not result["found"]:

        return result["answer"]


    entities = result[
        "entities"
    ]


    related_entities = result[
        "related_entities"
    ]


    document_indices = result[
        "documents"
    ]


    graph_context = result[
        "graph_context"
    ]


    document_context = result[
        "document_context"
    ]


    response = []


    # --------------------------------------------------------
    # MAIN ANSWER
    # --------------------------------------------------------

    response.append(
        result["answer"]
    )


    # --------------------------------------------------------
    # GRAPH INFORMATION
    # --------------------------------------------------------

    if len(entities) > 0:

        response.append(
            "Detected concepts: "
            + ", ".join(entities)
        )


    # --------------------------------------------------------
    # RELATED CONCEPTS
    # --------------------------------------------------------

    if len(related_entities) > 0:

        response.append(
            "Related concepts: "
            + ", ".join(related_entities)
        )


    # --------------------------------------------------------
    # SOURCE DOCUMENTS
    # --------------------------------------------------------

    if len(document_indices) > 0:

        document_numbers = [

            str(index + 1)

            for index in document_indices

        ]


        response.append(
            "Retrieved document IDs: "
            + ", ".join(document_numbers)
        )


    return "\n\n".join(
        response
    )


# ============================================================
# TEST COMPLETE GRAPHRAG CHATBOT
# ============================================================

test_queries = [

    "What is Artificial Intelligence?",

    "What is Machine Learning?",

    "How is Deep Learning related to Machine Learning?",

    "What is RAG?",

    "What does RAG use?",

    "How does LangChain relate to AI Agents?",

    "What is GraphRAG?"

]


print()
print("=" * 60)
print("GRAPHRAG CHATBOT TEST")
print("=" * 60)


for query in test_queries:

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
        "ASSISTANT:"
    )

    print(
        chatbot_response(query)
    )


# ============================================================
# SHOW ONE COMPLETE PIPELINE
# ============================================================

demo_query = (
    "How is RAG related to Vector Database?"
)


demo_result = run_graphrag(
    demo_query,
    max_hops=2
)


print()
print("=" * 60)
print("COMPLETE PIPELINE DEMONSTRATION")
print("=" * 60)


print()
print("1. USER QUERY")
print(
    demo_result["query"]
)


print()
print("2. ENTITY DETECTION")
print(
    demo_result["entities"]
)


print()
print("3. RELATED ENTITIES")
print(
    demo_result["related_entities"]
)


print()
print("4. GRAPH CONTEXT")
print(
    demo_result["graph_context"]
)


print()
print("5. RETRIEVED DOCUMENTS")

print(
    [
        index + 1
        for index in demo_result["documents"]
    ]
)


print()
print("6. DOCUMENT CONTEXT")
print(
    demo_result["document_context"]
)


print()
print("7. FINAL ANSWER")
print(
    demo_result["answer"]
)


# ============================================================
# INTERACTIVE GRAPHRAG CHATBOT
# ============================================================

def start_graphrag_chatbot():

    print()
    print("=" * 60)
    print("GRAPHRAG CHATBOT")
    print("=" * 60)

    print()
    print(
        "Ask questions about the AI knowledge base."
    )

    print(
        "Type 'exit' to stop."
    )

    print()


    while True:

        query = input(
            "You: "
        )


        if query.lower().strip() == "exit":

            print()
            print(
                "GraphRAG chatbot stopped."
            )

            break


        if query.strip() == "":

            print(
                "Please enter a question."
            )

            continue


        answer = chatbot_response(
            query
        )


        print()

        print(
            "GraphRAG:"
        )

        print(
            answer
        )

        print()


# ============================================================
# OPTIONAL: START CHATBOT
#
# Uncomment the next line to start the interactive chatbot.
# ============================================================

# start_graphrag_chatbot()


# ============================================================
# GRAPHRAG VS TRADITIONAL RAG CONCEPT
# ============================================================

print()
print("=" * 60)
print("GRAPHRAG ARCHITECTURE")
print("=" * 60)

print("""
USER QUERY
    |
    v
ENTITY DETECTION
    |
    v
KNOWLEDGE GRAPH
    |
    +----------------------+
    |                      |
    v                      v
GRAPH TRAVERSAL       RELATIONSHIPS
    |                      |
    +----------+-----------+
               |
               v
       RELATED ENTITIES
               |
               v
       DOCUMENT RETRIEVAL
               |
               v
       GRAPH + DOCUMENT
            CONTEXT
               |
               v
        ANSWER GENERATION
               |
               v
          FINAL ANSWER
""")


# ============================================================
# SHOW GRAPH STATISTICS
# ============================================================

print()
print("=" * 60)
print("GRAPHRAG PROJECT STATISTICS")
print("=" * 60)

print()

print(
    "Number of Documents:",
    len(documents)
)

print(
    "Number of Entities:",
    len(ENTITY_LIST)
)

print(
    "Number of Relationships:",
    len(RELATIONSHIPS)
)

print(
    "Number of Graph Nodes:",
    len(graph)
)

print(
    "Number of Graph Edges:",
    len(RELATIONSHIPS)
)


# ============================================================
# FINAL PROJECT VALIDATION
# ============================================================

print()
print("=" * 60)
print("DAY 55/100 — GRAPHRAG PROJECT COMPLETE")
print("=" * 60)

print()

print("PART 1")
print("✔ Knowledge base created")
print("✔ Entities extracted")
print("✔ Relationships created")
print("✔ Knowledge graph created")

print()

print("PART 2")
print("✔ Reverse graph created")
print("✔ Graph traversal implemented")
print("✔ Multi-hop retrieval implemented")
print("✔ Entity-to-document retrieval implemented")

print()

print("PART 3")
print("✔ Graph context created")
print("✔ Document context created")
print("✔ Combined context created")
print("✔ Answer generation implemented")

print()

print("PART 4")
print("✔ Complete GraphRAG pipeline")
print("✔ GraphRAG chatbot")
print("✔ Query processing")
print("✔ Context retrieval")
print("✔ Final answer generation")
print("✔ Interactive chatbot")
print("✔ Unknown query handling")

print()

print("HARDWARE OPTIMIZATION")
print("✔ No external dataset")
print("✔ No large language model")
print("✔ No embedding model")
print("✔ No GPU")
print("✔ No Neo4j")
print("✔ No persistent database")
print("✔ Very low storage usage")
print("✔ Slow CPU friendly")

print()
print("DAY 55/100 COMPLETE ✔")
print()
print("PROJECT: GRAPHRAG ✔")
print("STATUS: END-TO-END COMPLETE ✔")
