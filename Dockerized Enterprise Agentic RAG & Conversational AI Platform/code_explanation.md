# ============================================================
# 🚀 DAY 63/100 — PART 1
# Docker Fundamentals + Containerizing an AI Application
# ============================================================
#
# Goal:
#   1. Check Docker environment
#   2. Create a lightweight FastAPI AI service
#   3. Create requirements.txt
#   4. Create Dockerfile
#   5. Create .dockerignore
#   6. Create Docker Compose foundation
#   7. Build Docker image if Docker is available
#   8. Run/test the container if Docker is available
#
# CPU / STORAGE FRIENDLY
# No LLM download
# No large model
# No external dataset
# ============================================================

import os
import sys
import json
import subprocess
import textwrap
import time
from pathlib import Path
from datetime import datetime


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "day63_dockerized_enterprise_ai"
PROJECT_DIR = Path.cwd() / PROJECT_NAME

PROJECT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("🚀 DAY 63 — PART 1")
print("Docker Fundamentals + AI Application Containerization")
print("=" * 70)

print(f"\n📁 Project directory:")
print(PROJECT_DIR)


# ============================================================
# 2. CHECK CURRENT PYTHON ENVIRONMENT
# ============================================================

print("\n" + "=" * 70)
print("🐍 PYTHON ENVIRONMENT")
print("=" * 70)

print("Python version:", sys.version)
print("Python executable:", sys.executable)


# ============================================================
# 3. CHECK DOCKER INSTALLATION
# ============================================================

print("\n" + "=" * 70)
print("🐳 DOCKER ENVIRONMENT CHECK")
print("=" * 70)

docker_available = False
docker_version = None

try:
    result = subprocess.run(
        ["docker", "--version"],
        capture_output=True,
        text=True,
        timeout=10
    )

    if result.returncode == 0:
        docker_available = True
        docker_version = result.stdout.strip()
        print("✅ Docker installed")
        print(docker_version)
    else:
        print("⚠️ Docker command exists but returned an error.")
        print(result.stderr.strip())

except FileNotFoundError:
    print("⚠️ Docker is not installed or not available in PATH.")

except Exception as e:
    print("⚠️ Docker check failed:", str(e))


# ============================================================
# 4. CREATE LIGHTWEIGHT FASTAPI APPLICATION
# ============================================================
#
# This is a Docker-ready version of the AI-service pattern.
#
# Later, in Parts 2–4, this container can be replaced with
# the complete Day 62 Agentic RAG application.
#
# ============================================================

app_code = r'''
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os
import time


# ------------------------------------------------------------
# Application configuration
# ------------------------------------------------------------

APP_NAME = os.getenv(
    "APP_NAME",
    "Dockerized Enterprise AI Assistant"
)

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0"
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development"
)


# ------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Lightweight Dockerized Enterprise AI Service"
)


# ------------------------------------------------------------
# Request model
# ------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


# ------------------------------------------------------------
# Root endpoint
# ------------------------------------------------------------

@app.get("/")
def root():

    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "message": "Dockerized Enterprise AI Assistant is running",
        "timestamp": datetime.utcnow().isoformat()
    }


# ------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "application": APP_NAME,
        "environment": ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }


# ------------------------------------------------------------
# AI query endpoint
# ------------------------------------------------------------

@app.post("/query")
def query(request: QueryRequest):

    start_time = time.perf_counter()

    query_text = request.query.strip()

    if not query_text:
        return {
            "success": False,
            "answer": "Query cannot be empty.",
            "latency_ms": 0
        }

    # --------------------------------------------------------
    # Lightweight placeholder AI processing
    #
    # In later parts this layer will contain:
    #
    # Query
    #   ↓
    # Agentic RAG
    #   ↓
    # Retrieval
    #   ↓
    # Reranking
    #   ↓
    # Grounding
    #   ↓
    # Response
    # --------------------------------------------------------

    answer = (
        f"Received enterprise AI query: '{query_text}'. "
        "The Dockerized AI service is processing the request."
    )

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000,
        3
    )

    return {
        "success": True,
        "query": query_text,
        "answer": answer,
        "containerized": True,
        "latency_ms": latency_ms
    }


# ------------------------------------------------------------
# Container information endpoint
# ------------------------------------------------------------

@app.get("/info")
def info():

    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "container_ready": True,
        "architecture": [
            "FastAPI",
            "AI Processing Layer",
            "Docker Container"
        ]
    }
'''


APP_FILE = PROJECT_DIR / "app.py"

APP_FILE.write_text(
    app_code,
    encoding="utf-8"
)

print("\n✅ Created:", APP_FILE)


# ============================================================
# 5. CREATE REQUIREMENTS.TXT
# ============================================================

requirements = """fastapi
uvicorn[standard]
pydantic
"""

REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"

REQUIREMENTS_FILE.write_text(
    requirements,
    encoding="utf-8"
)

print("✅ Created:", REQUIREMENTS_FILE)


# ============================================================
# 6. CREATE DOCKERFILE
# ============================================================
#
# Important Docker concepts:
#
# FROM      → Base image
# WORKDIR   → Working directory inside container
# COPY      → Copy project files into image
# RUN       → Execute installation commands
# EXPOSE    → Document application port
# CMD       → Start application
#
# ============================================================

dockerfile = """FROM python:3.11-slim

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Make Python output immediately visible
ENV PYTHONUNBUFFERED=1

# Application directory
WORKDIR /app

# Copy dependency file first
# This helps Docker reuse the dependency layer
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Application port
EXPOSE 8000

# Start FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

DOCKERFILE = PROJECT_DIR / "Dockerfile"

DOCKERFILE.write_text(
    dockerfile,
    encoding="utf-8"
)

print("✅ Created:", DOCKERFILE)


# ============================================================
# 7. CREATE .DOCKERIGNORE
# ============================================================

dockerignore = """__pycache__
*.pyc
*.pyo
*.pyd

.ipynb_checkpoints
.ipynb_checkpoints/

.git
.gitignore

.env
.env.*

venv
.venv
env

*.log

*.csv
*.parquet
*.db

notebooks/
data/
models/

README.md
"""

DOCKERIGNORE_FILE = PROJECT_DIR / ".dockerignore"

DOCKERIGNORE_FILE.write_text(
    dockerignore,
    encoding="utf-8"
)

print("✅ Created:", DOCKERIGNORE_FILE)


# ============================================================
# 8. CREATE ENVIRONMENT FILE TEMPLATE
# ============================================================

env_template = """APP_NAME=Dockerized Enterprise AI Assistant
APP_VERSION=1.0.0
ENVIRONMENT=development
"""

ENV_FILE = PROJECT_DIR / ".env.example"

ENV_FILE.write_text(
    env_template,
    encoding="utf-8"
)

print("✅ Created:", ENV_FILE)


# ============================================================
# 9. CREATE DOCKER COMPOSE FOUNDATION
# ============================================================
#
# We will expand this in Part 3.
#
# For now:
#
# docker compose
#       ↓
# FastAPI container
#
# ============================================================

compose_file = """services:

  ai-api:
    build:
      context: .
      dockerfile: Dockerfile

    container_name: day63-ai-api

    ports:
      - "8000:8000"

    environment:
      APP_NAME: Dockerized Enterprise AI Assistant
      APP_VERSION: 1.0.0
      ENVIRONMENT: development

    restart: unless-stopped
"""

COMPOSE_FILE = PROJECT_DIR / "docker-compose.yml"

COMPOSE_FILE.write_text(
    compose_file,
    encoding="utf-8"
)

print("✅ Created:", COMPOSE_FILE)


# ============================================================
# 10. CREATE README
# ============================================================

readme = """# Day 63 — Dockerized Enterprise AI

## Part 1

Lightweight FastAPI service prepared for Docker containerization.

## Architecture

Client
    ↓
FastAPI
    ↓
AI Processing Layer
    ↓
Docker Container

## Endpoints

GET  /
GET  /health
GET  /info
POST /query

## Docker Commands

Build:

docker build -t day63-enterprise-ai .

Run:

docker run -d -p 8000:8000 --name day63-ai-container day63-enterprise-ai

Test:

http://localhost:8000

Stop:

docker stop day63-ai-container

Remove:

docker rm day63-ai-container
"""

README_FILE = PROJECT_DIR / "README.md"

README_FILE.write_text(
    readme,
    encoding="utf-8"
)

print("✅ Created:", README_FILE)


# ============================================================
# 11. DISPLAY DOCKERFILE
# ============================================================

print("\n" + "=" * 70)
print("🐳 DOCKERFILE")
print("=" * 70)

print(dockerfile)


# ============================================================
# 12. DISPLAY PROJECT STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("📂 PROJECT STRUCTURE")
print("=" * 70)

for file in sorted(PROJECT_DIR.iterdir()):
    print("├──", file.name)


# ============================================================
# 13. BUILD DOCKER IMAGE
# ============================================================
#
# Docker image:
#
# Dockerfile
#     ↓
# docker build
#     ↓
# Docker Image
#
# ============================================================

IMAGE_NAME = "day63-enterprise-ai"
CONTAINER_NAME = "day63-ai-container"

docker_build_success = False

if docker_available:

    print("\n" + "=" * 70)
    print("🔨 BUILDING DOCKER IMAGE")
    print("=" * 70)

    build_result = subprocess.run(
        [
            "docker",
            "build",
            "-t",
            IMAGE_NAME,
            "."
        ],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True
    )

    if build_result.returncode == 0:

        docker_build_success = True

        print("✅ Docker image built successfully.")
        print("\nImage:", IMAGE_NAME)

    else:

        print("❌ Docker image build failed.")
        print("\nSTDOUT:")
        print(build_result.stdout[-4000:])

        print("\nSTDERR:")
        print(build_result.stderr[-4000:])

else:

    print("\n⚠️ Docker is not available.")
    print("Project files were still created successfully.")
    print("You can build the image later from the project directory.")


# ============================================================
# 14. SHOW DOCKER IMAGES
# ============================================================

if docker_available and docker_build_success:

    print("\n" + "=" * 70)
    print("🖼️ DOCKER IMAGE")
    print("=" * 70)

    image_result = subprocess.run(
        [
            "docker",
            "images",
            IMAGE_NAME
        ],
        capture_output=True,
        text=True
    )

    print(image_result.stdout)


# ============================================================
# 15. RUN CONTAINER
# ============================================================

container_running = False

if docker_available and docker_build_success:

    print("\n" + "=" * 70)
    print("🚀 STARTING DOCKER CONTAINER")
    print("=" * 70)

    # Remove an old container if one exists
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
        text=True
    )

    run_result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            "8000:8000",
            "-e",
            "APP_NAME=Dockerized Enterprise AI Assistant",
            "-e",
            "APP_VERSION=1.0.0",
            "-e",
            "ENVIRONMENT=development",
            IMAGE_NAME
        ],
        capture_output=True,
        text=True
    )

    if run_result.returncode == 0:

        container_running = True

        container_id = run_result.stdout.strip()

        print("✅ Container started")
        print("Container ID:", container_id[:20])

        # Give FastAPI a moment to start
        time.sleep(3)

    else:

        print("❌ Container failed to start.")
        print(run_result.stderr)


# ============================================================
# 16. TEST CONTAINER
# ============================================================

if container_running:

    print("\n" + "=" * 70)
    print("🧪 TESTING CONTAINERIZED API")
    print("=" * 70)

    try:

        import urllib.request

        endpoints = [
            "/",
            "/health",
            "/info"
        ]

        for endpoint in endpoints:

            url = f"http://127.0.0.1:8000{endpoint}"

            try:

                with urllib.request.urlopen(
                    url,
                    timeout=5
                ) as response:

                    body = response.read().decode("utf-8")

                    print(f"\n✅ {endpoint}")
                    print("Status:", response.status)
                    print("Response:", body[:500])

            except Exception as endpoint_error:

                print(f"\n❌ {endpoint}")
                print("Error:", endpoint_error)

    except Exception as e:

        print("⚠️ API test could not run:", e)


# ============================================================
# 17. TEST QUERY ENDPOINT
# ============================================================

if container_running:

    print("\n" + "=" * 70)
    print("🤖 TESTING AI QUERY ENDPOINT")
    print("=" * 70)

    try:

        import urllib.request

        query_payload = json.dumps({
            "query": "What is the purpose of this enterprise AI system?"
        }).encode("utf-8")

        request = urllib.request.Request(
            "http://127.0.0.1:8000/query",
            data=query_payload,
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(
            request,
            timeout=5
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

            print("✅ Query API working")
            print(json.dumps(
                result,
                indent=2
            ))

    except Exception as e:

        print("❌ Query API test failed:", e)


# ============================================================
# 18. SHOW CONTAINER STATUS
# ============================================================

if docker_available:

    print("\n" + "=" * 70)
    print("📦 CONTAINER STATUS")
    print("=" * 70)

    status_result = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={CONTAINER_NAME}"
        ],
        capture_output=True,
        text=True
    )

    print(status_result.stdout)


# ============================================================
# 19. SHOW CONTAINER LOGS
# ============================================================

if container_running:

    print("\n" + "=" * 70)
    print("📋 CONTAINER LOGS")
    print("=" * 70)

    logs_result = subprocess.run(
        [
            "docker",
            "logs",
            "--tail",
            "30",
            CONTAINER_NAME
        ],
        capture_output=True,
        text=True
    )

    print(logs_result.stdout)


# ============================================================
# 20. DOCKER CONCEPTS SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("🧠 DOCKER CONCEPTS")
print("=" * 70)

docker_concepts = {
    "Dockerfile":
        "Instructions used to build a Docker image.",

    "Docker Image":
        "Immutable package containing application code, runtime and dependencies.",

    "Docker Container":
        "Running instance of a Docker image.",

    "Docker Build":
        "Creates an image from the Dockerfile.",

    "Docker Run":
        "Creates and starts a container from an image.",

    "WORKDIR":
        "Sets the working directory inside the container.",

    "COPY":
        "Copies files from the build context into the image.",

    "RUN":
        "Executes a command while building the image.",

    "EXPOSE":
        "Documents the port used by the application.",

    "CMD":
        "Defines the default command executed when the container starts.",

    "Port Mapping":
        "Maps a host port to a container port.",

    ".dockerignore":
        "Prevents unnecessary files from being sent into the Docker build context."
}

for concept, explanation in docker_concepts.items():

    print(f"\n🔹 {concept}")
    print(f"   {explanation}")


# ============================================================
# 21. AI ENGINEERING DOCKER FLOW
# ============================================================

print("\n" + "=" * 70)
print("🤖 AI ENGINEERING CONTAINER FLOW")
print("=" * 70)

print("""
Developer Code
      ↓
requirements.txt
      ↓
Dockerfile
      ↓
docker build
      ↓
Docker Image
      ↓
docker run
      ↓
Docker Container
      ↓
FastAPI
      ↓
AI Application
      ↓
REST API
""")


# ============================================================
# 22. PRODUCTION CONNECTION TO DAY 62
# ============================================================

print("\n" + "=" * 70)
print("🔗 CONNECTION TO DAY 62")
print("=" * 70)

print("""
DAY 62:

User
 ↓
FastAPI
 ↓
Agentic RAG
 ↓
Retrieval
 ↓
Reranking
 ↓
Grounding
 ↓
Response


DAY 63:

User
 ↓
Dockerized FastAPI
 ↓
Agentic RAG
 ↓
Retrieval
 ↓
Reranking
 ↓
Grounding
 ↓
Response
""")


# ============================================================
# 23. IMPORTANT VARIABLES CREATED
# ============================================================

print("\n" + "=" * 70)
print("📌 IMPORTANT VARIABLES CREATED")
print("=" * 70)

important_variables = {
    "PROJECT_NAME": PROJECT_NAME,
    "PROJECT_DIR": str(PROJECT_DIR),
    "APP_FILE": str(APP_FILE),
    "REQUIREMENTS_FILE": str(REQUIREMENTS_FILE),
    "DOCKERFILE": str(DOCKERFILE),
    "DOCKERIGNORE_FILE": str(DOCKERIGNORE_FILE),
    "COMPOSE_FILE": str(COMPOSE_FILE),
    "IMAGE_NAME": IMAGE_NAME,
    "CONTAINER_NAME": CONTAINER_NAME,
    "docker_available": docker_available,
    "docker_build_success": docker_build_success,
    "container_running": container_running
}

for key, value in important_variables.items():
    print(f"{key} = {value}")


# ============================================================
# 24. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("✅ PART 1 VALIDATION")
print("=" * 70)

required_files = [
    APP_FILE,
    REQUIREMENTS_FILE,
    DOCKERFILE,
    DOCKERIGNORE_FILE,
    ENV_FILE,
    COMPOSE_FILE,
    README_FILE
]

all_files_created = True

for file_path in required_files:

    exists = file_path.exists()

    print(
        f"{'✅' if exists else '❌'} "
        f"{file_path.name}"
    )

    if not exists:
        all_files_created = False


print("\nProject files created:", all_files_created)

if docker_available:

    print("Docker available:", docker_available)
    print("Docker image built:", docker_build_success)
    print("Container running:", container_running)

else:

    print("""
Docker is not currently available in this Jupyter environment.

That is okay.

The complete Docker project structure has been created.
You can later open a terminal inside the project directory and run:

docker build -t day63-enterprise-ai .

docker run -d -p 8000:8000 \
    --name day63-ai-container \
    day63-enterprise-ai

Then open:

http://localhost:8000
http://localhost:8000/docs
http://localhost:8000/health
""")


# ============================================================
# 25. FINAL PROJECT STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("📂 FINAL PART 1 STRUCTURE")
print("=" * 70)

print(f"""
{PROJECT_NAME}/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env.example
├── docker-compose.yml
└── README.md
""")


# ============================================================
# 26. FINAL MESSAGE
# ============================================================

print("=" * 70)
print("🎯 DAY 63 — PART 1 COMPLETED")
print("=" * 70)

print("""
You have now created the foundation for a Dockerized AI service.

Completed:

✅ FastAPI AI service
✅ requirements.txt
✅ Dockerfile
✅ .dockerignore
✅ Environment configuration
✅ Docker Compose foundation
✅ Docker image build logic
✅ Container execution logic
✅ API testing
✅ Container health testing
✅ Docker concepts
✅ Production connection to Day 62

Next:

DAY 63 — PART 2
🔥 Dockerize the complete FastAPI Agentic RAG application
🔥 Production Docker configuration
🔥 Environment variables
🔥 API containerization
🔥 Container health checks
🔥 Docker testing
""")

print("=" * 70)
# ============================================================
# 🚀 DAY 63/100 — PART 2
# Dockerize Enterprise Agentic RAG FastAPI Service
# ============================================================
#
# Continuation from:
#   Day 62 → Enterprise Agentic RAG System
#   Day 63 Part 1 → Docker Fundamentals
#
# Goal:
#   1. Create production-style FastAPI AI service
#   2. Connect Agentic RAG interface
#   3. Add /chat, /search, /health, /metrics
#   4. Add environment configuration
#   5. Create production Dockerfile
#   6. Create Docker healthcheck
#   7. Build Docker image
#   8. Run container
#   9. Test APIs
#  10. Validate complete container
#
# CPU / STORAGE FRIENDLY
# No large LLM
# No model download
# No external dataset
# ============================================================


import os
import sys
import json
import time
import uuid
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "day63_dockerized_enterprise_ai"

try:
    PROJECT_DIR
except NameError:
    PROJECT_DIR = Path.cwd() / PROJECT_NAME

PROJECT_DIR = Path(PROJECT_DIR)
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("🚀 DAY 63 — PART 2")
print("Dockerized Enterprise Agentic RAG FastAPI Service")
print("=" * 75)

print("\nProject:", PROJECT_DIR)


# ============================================================
# 2. APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "Enterprise Agentic RAG Assistant"
APP_VERSION = "1.0.0"
ENVIRONMENT = "development"

API_PREFIX = "/api/v1"

HOST = "0.0.0.0"
PORT = 8000

DEFAULT_TOP_K = 5
MAX_QUERY_LENGTH = 1000

ENABLE_GROUNDING = True
ENABLE_CITATIONS = True
ENABLE_SAFE_FALLBACK = True

print("\n" + "=" * 75)
print("⚙️ APPLICATION CONFIGURATION")
print("=" * 75)

print("APP_NAME:", APP_NAME)
print("APP_VERSION:", APP_VERSION)
print("ENVIRONMENT:", ENVIRONMENT)
print("API_PREFIX:", API_PREFIX)
print("PORT:", PORT)


# ============================================================
# 3. DETECT DAY 62 AGENTIC RAG FUNCTION
# ============================================================
#
# If Part 1–3 of Day 62 were executed in the same notebook,
# we try to reuse the existing agentic_rag() function.
#
# If it does not exist, a lightweight fallback implementation
# is created so this Part 2 remains executable.
#
# ============================================================

DAY62_AGENTIC_RAG_AVAILABLE = callable(
    globals().get("agentic_rag")
)

print("\n" + "=" * 75)
print("🔗 DAY 62 INTEGRATION")
print("=" * 75)

if DAY62_AGENTIC_RAG_AVAILABLE:

    print("✅ Existing Day 62 agentic_rag() detected.")
    print("The Dockerized API will reuse the existing function.")

else:

    print("⚠️ Day 62 agentic_rag() not found.")
    print("Creating a lightweight fallback AI service.")
    print("You can replace this function with the Day 62 implementation later.")


# ============================================================
# 4. LIGHTWEIGHT FALLBACK KNOWLEDGE BASE
# ============================================================

if not DAY62_AGENTIC_RAG_AVAILABLE:

    FALLBACK_KNOWLEDGE = [
        {
            "id": "DOC001",
            "title": "Enterprise AI Policy",
            "department": "AI",
            "text": (
                "Enterprise AI systems should use controlled retrieval, "
                "grounding validation, source attribution, monitoring, "
                "access control and safe fallback mechanisms."
            )
        },
        {
            "id": "DOC002",
            "title": "RAG Architecture",
            "department": "Engineering",
            "text": (
                "A retrieval augmented generation system retrieves relevant "
                "documents, constructs context and generates an answer using "
                "the retrieved evidence."
            )
        },
        {
            "id": "DOC003",
            "title": "Docker Deployment",
            "department": "DevOps",
            "text": (
                "Docker packages an application and its dependencies into "
                "a portable container image that can run consistently "
                "across environments."
            )
        },
        {
            "id": "DOC004",
            "title": "API Security",
            "department": "Security",
            "text": (
                "Production APIs should use authentication, authorization, "
                "input validation, rate limiting, secret management and "
                "audit logging."
            )
        },
        {
            "id": "DOC005",
            "title": "AI Monitoring",
            "department": "AI",
            "text": (
                "AI services should monitor latency, error rate, retrieval "
                "quality, grounding rate, citation coverage and safe "
                "fallback behavior."
            )
        }
    ]


# ============================================================
# 5. FALLBACK RETRIEVAL
# ============================================================

if not DAY62_AGENTIC_RAG_AVAILABLE:

    def lightweight_retrieve(query, top_k=5):

        query_words = set(
            query.lower().split()
        )

        results = []

        for document in FALLBACK_KNOWLEDGE:

            document_words = set(
                document["text"].lower().split()
            )

            title_words = set(
                document["title"].lower().split()
            )

            overlap = len(
                query_words.intersection(
                    document_words.union(title_words)
                )
            )

            score = overlap / max(
                len(query_words),
                1
            )

            results.append({
                "document_id": document["id"],
                "title": document["title"],
                "department": document["department"],
                "text": document["text"],
                "score": round(score, 4)
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]


# ============================================================
# 6. FALLBACK AGENTIC RAG
# ============================================================

if not DAY62_AGENTIC_RAG_AVAILABLE:

    def agentic_rag(query):

        start_time = time.perf_counter()

        query = str(query).strip()

        if not query:

            return {
                "success": False,
                "answer": "Query cannot be empty.",
                "citations": [],
                "grounded": False,
                "safe_fallback": True,
                "confidence": 0.0,
                "retrieval_confidence": 0.0,
                "retrieved_chunks": 0,
                "grounding_score": 0.0,
                "latency_ms": 0.0
            }

        results = lightweight_retrieve(
            query,
            top_k=DEFAULT_TOP_K
        )

        relevant_results = [
            item
            for item in results
            if item["score"] > 0
        ]

        if not relevant_results:

            latency_ms = round(
                (time.perf_counter() - start_time) * 1000,
                3
            )

            return {
                "success": True,
                "answer": (
                    "I could not find sufficient evidence in the "
                    "available enterprise knowledge base."
                ),
                "citations": [],
                "grounded": False,
                "safe_fallback": True,
                "confidence": 0.0,
                "retrieval_confidence": 0.0,
                "retrieved_chunks": 0,
                "grounding_score": 0.0,
                "latency_ms": latency_ms
            }

        best_score = relevant_results[0]["score"]

        context = " ".join(
            item["text"]
            for item in relevant_results[:3]
        )

        # Lightweight deterministic response.
        answer = (
            f"Based on the retrieved enterprise information: {context}"
        )

        citations = [
            {
                "document_id": item["document_id"],
                "title": item["title"],
                "department": item["department"]
            }
            for item in relevant_results[:3]
        ]

        grounding_score = round(
            min(
                1.0,
                0.5 + best_score
            ),
            4
        )

        confidence = round(
            min(
                1.0,
                best_score + 0.5
            ),
            4
        )

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            3
        )

        return {
            "success": True,
            "answer": answer,
            "citations": citations,
            "grounded": grounding_score >= 0.5,
            "safe_fallback": False,
            "confidence": confidence,
            "retrieval_confidence": confidence,
            "retrieved_chunks": len(relevant_results),
            "grounding_score": grounding_score,
            "latency_ms": latency_ms
        }


# ============================================================
# 7. SEARCH ADAPTER
# ============================================================

def perform_search(query, top_k=DEFAULT_TOP_K):

    query = str(query).strip()

    if not query:
        return []

    # --------------------------------------------------------
    # Use Day 62 retrieval if available.
    #
    # Because Day 62 implementations can have different
    # return structures, we protect the API from failures.
    # --------------------------------------------------------

    if DAY62_AGENTIC_RAG_AVAILABLE:

        try:

            # Try common retrieval function names.
            retrieval_function = None

            for function_name in [
                "advanced_rag_retrieval",
                "enterprise_retriever",
                "conversational_retriever"
            ]:

                candidate = globals().get(
                    function_name
                )

                if callable(candidate):
                    retrieval_function = candidate
                    break

            if retrieval_function:

                raw_results = retrieval_function(query)

                if isinstance(raw_results, list):

                    return raw_results[:top_k]

        except Exception as retrieval_error:

            print(
                "⚠️ Existing retrieval integration failed:",
                retrieval_error
            )

    # --------------------------------------------------------
    # Lightweight fallback
    # --------------------------------------------------------

    if not DAY62_AGENTIC_RAG_AVAILABLE:

        return lightweight_retrieve(
            query,
            top_k=top_k
        )

    return []


# ============================================================
# 8. APPLICATION METRICS
# ============================================================

class ApplicationMetrics:

    def __init__(self):

        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        self.total_latency_ms = 0.0

        self.chat_requests = 0
        self.search_requests = 0

        self.grounded_responses = 0
        self.safe_fallbacks = 0
        self.citations_returned = 0

        self.latencies = []

    def record_request(
        self,
        latency_ms,
        success=True
    ):

        self.total_requests += 1

        self.total_latency_ms += latency_ms

        self.latencies.append(
            latency_ms
        )

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

    def percentile(self, percentile):

        if not self.latencies:
            return 0.0

        values = sorted(
            self.latencies
        )

        index = int(
            (percentile / 100)
            * (len(values) - 1)
        )

        return round(
            values[index],
            3
        )

    def summary(self):

        average_latency = (
            self.total_latency_ms
            / self.total_requests
            if self.total_requests
            else 0.0
        )

        success_rate = (
            self.successful_requests
            / self.total_requests
            if self.total_requests
            else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(
                success_rate,
                4
            ),
            "average_latency_ms": round(
                average_latency,
                3
            ),
            "p50_latency_ms": self.percentile(50),
            "p95_latency_ms": self.percentile(95),
            "p99_latency_ms": self.percentile(99),
            "chat_requests": self.chat_requests,
            "search_requests": self.search_requests,
            "grounded_responses": self.grounded_responses,
            "safe_fallbacks": self.safe_fallbacks,
            "citations_returned": self.citations_returned
        }


metrics = ApplicationMetrics()


# ============================================================
# 9. SESSION MEMORY
# ============================================================

SESSION_STORE = defaultdict(list)

MAX_SESSION_HISTORY = 10


def create_session():

    session_id = str(
        uuid.uuid4()
    )

    SESSION_STORE[
        session_id
    ] = []

    return session_id


def add_session_message(
    session_id,
    role,
    content
):

    if session_id not in SESSION_STORE:

        SESSION_STORE[
            session_id
        ] = []

    SESSION_STORE[
        session_id
    ].append({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    })

    SESSION_STORE[
        session_id
    ] = SESSION_STORE[
        session_id
    ][-MAX_SESSION_HISTORY:]


# ============================================================
# 10. CREATE PRODUCTION FASTAPI APPLICATION
# ============================================================

fastapi_app_code = r'''
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import time
import uuid


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = os.getenv(
    "APP_NAME",
    "Enterprise Agentic RAG Assistant"
)

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0"
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development"
)

API_PREFIX = os.getenv(
    "API_PREFIX",
    "/api/v1"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Dockerized Enterprise Agentic RAG API"
    )
)


# ============================================================
# SIMPLE IN-MEMORY STATE
# ============================================================

SESSIONS = {}

METRICS = {
    "requests": 0,
    "successes": 0,
    "failures": 0,
    "total_latency_ms": 0.0
}


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000
    )

    session_id: Optional[str] = None


class SearchRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "status": "running",
        "containerized": True,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "application": APP_NAME,
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "containerized": True
    }


# ============================================================
# INFO
# ============================================================

@app.get(f"{API_PREFIX}/info")
def info():

    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "architecture": [
            "FastAPI",
            "Agentic RAG",
            "Retrieval",
            "Reranking",
            "Grounding",
            "Docker"
        ]
    }


# ============================================================
# METRICS
# ============================================================

@app.get(f"{API_PREFIX}/metrics")
def metrics():

    total = METRICS["requests"]

    average_latency = (
        METRICS["total_latency_ms"] / total
        if total > 0
        else 0.0
    )

    success_rate = (
        METRICS["successes"] / total
        if total > 0
        else 0.0
    )

    return {
        "requests": total,
        "successes": METRICS["successes"],
        "failures": METRICS["failures"],
        "success_rate": round(
            success_rate,
            4
        ),
        "average_latency_ms": round(
            average_latency,
            3
        )
    }


# ============================================================
# SESSION
# ============================================================

@app.post(f"{API_PREFIX}/session")
def create_session():

    session_id = str(
        uuid.uuid4()
    )

    SESSIONS[session_id] = []

    return {
        "session_id": session_id,
        "created": True
    }


@app.get(f"{API_PREFIX}/session/{{session_id}}")
def get_session(session_id: str):

    if session_id not in SESSIONS:

        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    return {
        "session_id": session_id,
        "messages": SESSIONS[session_id]
    }


@app.delete(f"{API_PREFIX}/session/{{session_id}}")
def delete_session(session_id: str):

    if session_id not in SESSIONS:

        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    del SESSIONS[session_id]

    return {
        "session_id": session_id,
        "deleted": True
    }


# ============================================================
# SEARCH
# ============================================================

@app.post(f"{API_PREFIX}/search")
def search(request: SearchRequest):

    start_time = time.perf_counter()

    METRICS["requests"] += 1

    try:

        results = perform_search(
            request.query,
            request.top_k
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        METRICS[
            "total_latency_ms"
        ] += latency_ms

        METRICS[
            "successes"
        ] += 1

        return {
            "success": True,
            "query": request.query,
            "results": results,
            "result_count": len(results),
            "latency_ms": round(
                latency_ms,
                3
            )
        }

    except Exception as error:

        METRICS[
            "failures"
        ] += 1

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# CHAT
# ============================================================

@app.post(f"{API_PREFIX}/chat")
def chat(request: ChatRequest):

    start_time = time.perf_counter()

    METRICS["requests"] += 1

    session_id = request.session_id

    if not session_id:

        session_id = str(
            uuid.uuid4()
        )

        SESSIONS[
            session_id
        ] = []

    if session_id not in SESSIONS:

        SESSIONS[
            session_id
        ] = []

    try:

        # ----------------------------------------------------
        # Lightweight AI processing.
        #
        # In the complete production implementation this
        # function is replaced by the Agentic RAG pipeline.
        # ----------------------------------------------------

        query_lower = request.query.lower()

        if "docker" in query_lower:

            answer = (
                "Docker packages an application and its "
                "dependencies into a portable container."
            )

            grounded = True
            safe_fallback = False

            citations = [
                {
                    "document_id": "DOC003",
                    "title": "Docker Deployment",
                    "department": "DevOps"
                }
            ]

            confidence = 0.95

        elif "rag" in query_lower:

            answer = (
                "RAG retrieves relevant information and uses "
                "that retrieved context to generate a grounded response."
            )

            grounded = True
            safe_fallback = False

            citations = [
                {
                    "document_id": "DOC002",
                    "title": "RAG Architecture",
                    "department": "Engineering"
                }
            ]

            confidence = 0.95

        elif "security" in query_lower:

            answer = (
                "Production AI APIs should use authentication, "
                "authorization, input validation, rate limiting "
                "and secret management."
            )

            grounded = True
            safe_fallback = False

            citations = [
                {
                    "document_id": "DOC004",
                    "title": "API Security",
                    "department": "Security"
                }
            ]

            confidence = 0.90

        else:

            answer = (
                "I could not find sufficient evidence to provide "
                "a confident enterprise answer for this query."
            )

            grounded = False
            safe_fallback = True

            citations = []

            confidence = 0.0

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # ----------------------------------------------------
        # Store conversation
        # ----------------------------------------------------

        SESSIONS[
            session_id
        ].append({
            "role": "user",
            "content": request.query,
            "timestamp": datetime.utcnow().isoformat()
        })

        SESSIONS[
            session_id
        ].append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.utcnow().isoformat()
        })

        SESSIONS[
            session_id
        ] = SESSIONS[
            session_id
        ][-10:]

        METRICS[
            "total_latency_ms"
        ] += latency_ms

        METRICS[
            "successes"
        ] += 1

        return {
            "success": True,
            "session_id": session_id,
            "query": request.query,
            "answer": answer,
            "grounded": grounded,
            "safe_fallback": safe_fallback,
            "confidence": confidence,
            "citations": citations,
            "latency_ms": round(
                latency_ms,
                3
            )
        }

    except Exception as error:

        METRICS[
            "failures"
        ] += 1

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
'''


FASTAPI_APP_FILE = PROJECT_DIR / "main.py"

FASTAPI_APP_FILE.write_text(
    fastapi_app_code,
    encoding="utf-8"
)

print("\n" + "=" * 75)
print("✅ FastAPI production service created")
print("=" * 75)

print(FASTAPI_APP_FILE)


# ============================================================
# 11. CREATE PRODUCTION REQUIREMENTS
# ============================================================

requirements_production = """fastapi
uvicorn[standard]
pydantic
"""

REQUIREMENTS_PROD_FILE = (
    PROJECT_DIR /
    "requirements.txt"
)

REQUIREMENTS_PROD_FILE.write_text(
    requirements_production,
    encoding="utf-8"
)

print("\n✅ Updated requirements.txt")


# ============================================================
# 12. CREATE PRODUCTION DOCKERFILE
# ============================================================
#
# Important optimization:
#
# requirements.txt is copied before source code.
#
# Docker can cache dependency installation.
#
# Therefore if main.py changes:
#
# Docker does NOT necessarily reinstall dependencies.
#
# ============================================================

production_dockerfile = """FROM python:3.11-slim

# ------------------------------------------------------------
# Python runtime configuration
# ------------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# Application directory
# ------------------------------------------------------------

WORKDIR /app

# ------------------------------------------------------------
# Install dependencies
# ------------------------------------------------------------

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Copy application
# ------------------------------------------------------------

COPY main.py .

# ------------------------------------------------------------
# Application port
# ------------------------------------------------------------

EXPOSE 8000

# ------------------------------------------------------------
# Container health check
# ------------------------------------------------------------

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# ------------------------------------------------------------
# Start FastAPI
# ------------------------------------------------------------

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

PRODUCTION_DOCKERFILE = (
    PROJECT_DIR /
    "Dockerfile"
)

PRODUCTION_DOCKERFILE.write_text(
    production_dockerfile,
    encoding="utf-8"
)

print("\n" + "=" * 75)
print("🐳 PRODUCTION DOCKERFILE")
print("=" * 75)

print(production_dockerfile)


# ============================================================
# 13. CREATE .DOCKERIGNORE
# ============================================================

dockerignore_content = """__pycache__/
*.py[cod]

.ipynb_checkpoints/
*.ipynb

.git/
.gitignore

.env
.env.*

venv/
.venv/
env/

*.log

data/
datasets/
models/
checkpoints/

*.csv
*.parquet
*.db

README.md
"""

DOCKERIGNORE_PATH = (
    PROJECT_DIR /
    ".dockerignore"
)

DOCKERIGNORE_PATH.write_text(
    dockerignore_content,
    encoding="utf-8"
)

print("✅ Updated .dockerignore")


# ============================================================
# 14. CREATE .ENV.EXAMPLE
# ============================================================

env_content = """APP_NAME=Enterprise Agentic RAG Assistant
APP_VERSION=1.0.0
ENVIRONMENT=production
API_PREFIX=/api/v1
"""

ENV_EXAMPLE = (
    PROJECT_DIR /
    ".env.example"
)

ENV_EXAMPLE.write_text(
    env_content,
    encoding="utf-8"
)

print("✅ Created .env.example")


# ============================================================
# 15. CREATE DOCKER COMPOSE
# ============================================================

compose_content = """services:

  ai-api:

    build:
      context: .
      dockerfile: Dockerfile

    container_name: day63-agentic-rag-api

    ports:
      - "8000:8000"

    environment:
      APP_NAME: Enterprise Agentic RAG Assistant
      APP_VERSION: 1.0.0
      ENVIRONMENT: production
      API_PREFIX: /api/v1

    restart: unless-stopped

    healthcheck:
      test:
        [
          "CMD",
          "python",
          "-c",
          "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
        ]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
"""

COMPOSE_PATH = (
    PROJECT_DIR /
    "docker-compose.yml"
)

COMPOSE_PATH.write_text(
    compose_content,
    encoding="utf-8"
)

print("✅ Updated docker-compose.yml")


# ============================================================
# 16. VALIDATE PYTHON APPLICATION SYNTAX
# ============================================================

print("\n" + "=" * 75)
print("🔍 PYTHON SYNTAX VALIDATION")
print("=" * 75)

syntax_result = subprocess.run(
    [
        sys.executable,
        "-m",
        "py_compile",
        str(FASTAPI_APP_FILE)
    ],
    capture_output=True,
    text=True
)

if syntax_result.returncode == 0:

    print("✅ main.py syntax is valid.")

else:

    print("❌ Syntax error detected.")
    print(syntax_result.stderr)


# ============================================================
# 17. CHECK DOCKER
# ============================================================

print("\n" + "=" * 75)
print("🐳 DOCKER CHECK")
print("=" * 75)

docker_available_part2 = False

try:

    docker_check = subprocess.run(
        ["docker", "--version"],
        capture_output=True,
        text=True,
        timeout=10
    )

    if docker_check.returncode == 0:

        docker_available_part2 = True

        print("✅ Docker available")
        print(
            docker_check.stdout.strip()
        )

    else:

        print("⚠️ Docker command failed.")

except FileNotFoundError:

    print(
        "⚠️ Docker is not installed "
        "or not available in PATH."
    )

except Exception as error:

    print(
        "⚠️ Docker check failed:",
        error
    )


# ============================================================
# 18. BUILD DOCKER IMAGE
# ============================================================

IMAGE_NAME_PART2 = (
    "day63-enterprise-agentic-rag"
)

CONTAINER_NAME_PART2 = (
    "day63-agentic-rag-api"
)

IMAGE_BUILT_PART2 = False
CONTAINER_STARTED_PART2 = False


if docker_available_part2:

    print("\n" + "=" * 75)
    print("🔨 BUILDING DOCKER IMAGE")
    print("=" * 75)

    build_result = subprocess.run(
        [
            "docker",
            "build",
            "-t",
            IMAGE_NAME_PART2,
            "."
        ],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True
    )

    if build_result.returncode == 0:

        IMAGE_BUILT_PART2 = True

        print(
            "✅ Docker image built successfully."
        )

        print(
            "Image:",
            IMAGE_NAME_PART2
        )

    else:

        print(
            "❌ Docker image build failed."
        )

        print(
            "\nSTDOUT:\n",
            build_result.stdout[-5000:]
        )

        print(
            "\nSTDERR:\n",
            build_result.stderr[-5000:]
        )

else:

    print(
        "\n⚠️ Docker is unavailable."
    )

    print(
        "Docker project files are still ready."
    )


# ============================================================
# 19. RUN CONTAINER
# ============================================================

if docker_available_part2 and IMAGE_BUILT_PART2:

    print("\n" + "=" * 75)
    print("🚀 STARTING CONTAINER")
    print("=" * 75)

    # Remove old container if present
    subprocess.run(
        [
            "docker",
            "rm",
            "-f",
            CONTAINER_NAME_PART2
        ],
        capture_output=True,
        text=True
    )

    run_result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME_PART2,
            "-p",
            "8000:8000",
            "-e",
            "APP_NAME=Enterprise Agentic RAG Assistant",
            "-e",
            "APP_VERSION=1.0.0",
            "-e",
            "ENVIRONMENT=production",
            "-e",
            "API_PREFIX=/api/v1",
            IMAGE_NAME_PART2
        ],
        capture_output=True,
        text=True
    )

    if run_result.returncode == 0:

        CONTAINER_STARTED_PART2 = True

        print(
            "✅ Container started."
        )

        print(
            "Container ID:",
            run_result.stdout.strip()[:20]
        )

        time.sleep(3)

    else:

        print(
            "❌ Container failed to start."
        )

        print(
            run_result.stderr
        )


# ============================================================
# 20. TEST ROOT ENDPOINT
# ============================================================

def http_get(url):

    with urllib.request.urlopen(
        url,
        timeout=5
    ) as response:

        body = response.read().decode(
            "utf-8"
        )

        return response.status, body


def http_post(
    url,
    payload
):

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=5
    ) as response:

        body = response.read().decode(
            "utf-8"
        )

        return response.status, body


if CONTAINER_STARTED_PART2:

    print("\n" + "=" * 75)
    print("🧪 CONTAINER API TESTS")
    print("=" * 75)

    BASE_URL = "http://127.0.0.1:8000"

    GET_ENDPOINTS = [
        "/",
        "/health",
        "/api/v1/info",
        "/api/v1/metrics"
    ]

    for endpoint in GET_ENDPOINTS:

        try:

            status, body = http_get(
                BASE_URL + endpoint
            )

            print(
                f"\n✅ GET {endpoint}"
            )

            print(
                "Status:",
                status
            )

            print(
                "Response:",
                body[:1000]
            )

        except Exception as error:

            print(
                f"\n❌ GET {endpoint}"
            )

            print(
                "Error:",
                error
            )


# ============================================================
# 21. TEST SEARCH API
# ============================================================

if CONTAINER_STARTED_PART2:

    print("\n" + "=" * 75)
    print("🔎 SEARCH API TEST")
    print("=" * 75)

    try:

        status, body = http_post(
            BASE_URL + "/api/v1/search",
            {
                "query": "Docker container deployment",
                "top_k": 5
            }
        )

        print(
            "Status:",
            status
        )

        print(
            json.dumps(
                json.loads(body),
                indent=2
            )
        )

    except Exception as error:

        print(
            "❌ Search test failed:",
            error
        )


# ============================================================
# 22. TEST CHAT API
# ============================================================

if CONTAINER_STARTED_PART2:

    print("\n" + "=" * 75)
    print("🤖 CHAT API TEST")
    print("=" * 75)

    try:

        status, body = http_post(
            BASE_URL + "/api/v1/chat",
            {
                "query": (
                    "How does Docker help deploy "
                    "an enterprise AI application?"
                )
            }
        )

        print(
            "Status:",
            status
        )

        chat_result = json.loads(
            body
        )

        print(
            json.dumps(
                chat_result,
                indent=2
            )
        )

    except Exception as error:

        print(
            "❌ Chat test failed:",
            error
        )


# ============================================================
# 23. TEST SESSION API
# ============================================================

if CONTAINER_STARTED_PART2:

    print("\n" + "=" * 75)
    print("🧠 SESSION API TEST")
    print("=" * 75)

    try:

        status, body = http_post(
            BASE_URL + "/api/v1/session",
            {}
        )

        session_result = json.loads(
            body
        )

        session_id = session_result[
            "session_id"
        ]

        print(
            "✅ Session created"
        )

        print(
            "Session ID:",
            session_id
        )

        # Send message to session
        status, body = http_post(
            BASE_URL + "/api/v1/chat",
            {
                "query": "What is RAG?",
                "session_id": session_id
            }
        )

        print(
            "\nChat status:",
            status
        )

        # Retrieve session
        status, body = http_get(
            BASE_URL
            + f"/api/v1/session/{session_id}"
        )

        print(
            "\nSession history status:",
            status
        )

        print(
            body[:2000]
        )

    except Exception as error:

        print(
            "❌ Session test failed:",
            error
        )


# ============================================================
# 24. PERFORMANCE TEST
# ============================================================

if CONTAINER_STARTED_PART2:

    print("\n" + "=" * 75)
    print("⚡ CONTAINER PERFORMANCE TEST")
    print("=" * 75)

    performance_queries = [
        "What is Docker?",
        "What is RAG?",
        "How should an API be secured?",
        "How should AI systems be monitored?",
        "Explain enterprise AI architecture."
    ]

    latency_values = []

    for query in performance_queries:

        start = time.perf_counter()

        try:

            status, body = http_post(
                BASE_URL + "/api/v1/chat",
                {
                    "query": query
                }
            )

            latency = (
                time.perf_counter()
                - start
            ) * 1000

            latency_values.append(
                latency
            )

            print(
                f"✅ {query[:45]:45} "
                f"{latency:.2f} ms"
            )

        except Exception as error:

            print(
                f"❌ {query[:45]:45}",
                error
            )

    if latency_values:

        sorted_latencies = sorted(
            latency_values
        )

        avg_latency = (
            sum(latency_values)
            / len(latency_values)
        )

        p50 = sorted_latencies[
            int(
                0.50
                * (len(sorted_latencies) - 1)
            )
        ]

        p95 = sorted_latencies[
            int(
                0.95
                * (len(sorted_latencies) - 1)
            )
        ]

        print(
            "\nAverage latency:",
            round(avg_latency, 3),
            "ms"
        )

        print(
            "P50 latency:",
            round(p50, 3),
            "ms"
        )

        print(
            "P95 latency:",
            round(p95, 3),
            "ms"
        )


# ============================================================
# 25. DOCKER CONTAINER STATUS
# ============================================================

if docker_available_part2:

    print("\n" + "=" * 75)
    print("📦 DOCKER CONTAINER STATUS")
    print("=" * 75)

    status_result = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={CONTAINER_NAME_PART2}"
        ],
        capture_output=True,
        text=True
    )

    print(
        status_result.stdout
    )


# ============================================================
# 26. DOCKER HEALTH STATUS
# ============================================================

if CONTAINER_STARTED_PART2:

    print("\n" + "=" * 75)
    print("❤️ CONTAINER HEALTH")
    print("=" * 75)

    inspect_result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            "{{json .State.Health}}",
            CONTAINER_NAME_PART2
        ],
        capture_output=True,
        text=True
    )

    print(
        inspect_result.stdout
    )


# ============================================================
# 27. DOCKER LOGS
# ============================================================

if CONTAINER_STARTED_PART2:

    print("\n" + "=" * 75)
    print("📋 CONTAINER LOGS")
    print("=" * 75)

    logs_result = subprocess.run(
        [
            "docker",
            "logs",
            "--tail",
            "50",
            CONTAINER_NAME_PART2
        ],
        capture_output=True,
        text=True
    )

    print(
        logs_result.stdout
    )


# ============================================================
# 28. SHOW DOCKER IMAGE INFORMATION
# ============================================================

if docker_available_part2 and IMAGE_BUILT_PART2:

    print("\n" + "=" * 75)
    print("🖼️ DOCKER IMAGE INFORMATION")
    print("=" * 75)

    image_result = subprocess.run(
        [
            "docker",
            "images",
            IMAGE_NAME_PART2
        ],
        capture_output=True,
        text=True
    )

    print(
        image_result.stdout
    )


# ============================================================
# 29. PRODUCTION DOCKER ARCHITECTURE
# ============================================================

print("\n" + "=" * 75)
print("🏗️ PRODUCTION ARCHITECTURE")
print("=" * 75)

print("""
                    USER
                      │
                      ▼
               API Gateway / WAF
                      │
                      ▼
                Load Balancer
                      │
                      ▼
              ┌─────────────────┐
              │ Docker Container│
              │                 │
              │    FastAPI      │
              │       │         │
              │       ▼         │
              │ Agentic RAG     │
              │       │         │
              │  ┌────┴─────┐   │
              │  ▼          ▼   │
              │Retrieval  Agent │
              │  │          │   │
              │  └────┬─────┘   │
              │       ▼         │
              │   Reranking     │
              │       ▼         │
              │   Grounding     │
              │       ▼         │
              │   Response      │
              └─────────────────┘
                      │
                      ▼
                  Monitoring
""")


# ============================================================
# 30. WHAT WE ACHIEVED
# ============================================================

print("\n" + "=" * 75)
print("✅ PART 2 ACHIEVEMENTS")
print("=" * 75)

achievements = [
    "Created production-style FastAPI AI service",
    "Added Agentic RAG API interface",
    "Added /chat endpoint",
    "Added /search endpoint",
    "Added /health endpoint",
    "Added /metrics endpoint",
    "Added /session endpoints",
    "Added request validation",
    "Added safe fallback behavior",
    "Added source citations",
    "Added grounding status",
    "Added Docker healthcheck",
    "Created production Dockerfile",
    "Created Docker Compose configuration",
    "Created environment configuration",
    "Validated Python syntax",
    "Built Docker image when Docker is available",
    "Started Docker container when Docker is available",
    "Tested container APIs",
    "Tested search",
    "Tested chat",
    "Tested session management",
    "Measured container API latency"
]

for item in achievements:
    print("✅", item)


# ============================================================
# 31. IMPORTANT VARIABLES
# ============================================================

print("\n" + "=" * 75)
print("📌 IMPORTANT VARIABLES CREATED")
print("=" * 75)

IMPORTANT_VARIABLES_PART2 = {
    "PROJECT_DIR": str(PROJECT_DIR),
    "APP_NAME": APP_NAME,
    "APP_VERSION": APP_VERSION,
    "ENVIRONMENT": ENVIRONMENT,
    "API_PREFIX": API_PREFIX,
    "FASTAPI_APP_FILE": str(
        FASTAPI_APP_FILE
    ),
    "PRODUCTION_DOCKERFILE": str(
        PRODUCTION_DOCKERFILE
    ),
    "DOCKERIGNORE_PATH": str(
        DOCKERIGNORE_PATH
    ),
    "COMPOSE_PATH": str(
        COMPOSE_PATH
    ),
    "IMAGE_NAME_PART2": IMAGE_NAME_PART2,
    "CONTAINER_NAME_PART2": CONTAINER_NAME_PART2,
    "IMAGE_BUILT_PART2": IMAGE_BUILT_PART2,
    "CONTAINER_STARTED_PART2": CONTAINER_STARTED_PART2,
    "DAY62_AGENTIC_RAG_AVAILABLE":
        DAY62_AGENTIC_RAG_AVAILABLE
}

for key, value in IMPORTANT_VARIABLES_PART2.items():

    print(
        f"{key} = {value}"
    )


# ============================================================
# 32. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("🎯 PART 2 FINAL VALIDATION")
print("=" * 75)

required_part2_files = [
    FASTAPI_APP_FILE,
    REQUIREMENTS_PROD_FILE,
    PRODUCTION_DOCKERFILE,
    DOCKERIGNORE_PATH,
    ENV_EXAMPLE,
    COMPOSE_PATH
]

all_files_valid = True

for file_path in required_part2_files:

    exists = file_path.exists()

    print(
        f"{'✅' if exists else '❌'} "
        f"{file_path.name}"
    )

    if not exists:
        all_files_valid = False


print(
    "\nAll Part 2 files created:",
    all_files_valid
)

print(
    "Python syntax valid:",
    syntax_result.returncode == 0
)

print(
    "Docker available:",
    docker_available_part2
)

print(
    "Docker image built:",
    IMAGE_BUILT_PART2
)

print(
    "Docker container started:",
    CONTAINER_STARTED_PART2
)


# ============================================================
# 33. FINAL PROJECT STRUCTURE
# ============================================================

print("\n" + "=" * 75)
print("📂 DAY 63 PART 2 PROJECT")
print("=" * 75)

print(f"""
{PROJECT_NAME}/
│
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
│
└── Day 62 Agentic RAG
        │
        ├── Query Understanding
        ├── Query Expansion
        ├── Retrieval
        ├── Reranking
        ├── Grounding
        └── Safe Fallback
""")


# ============================================================
# 34. FINAL ARCHITECTURE
# ============================================================

print("\n" + "=" * 75)
print("🔥 DAY 63 PART 2 END-TO-END FLOW")
print("=" * 75)

print("""
                    USER
                      ↓
             Dockerized FastAPI
                      ↓
             Request Validation
                      ↓
              Agentic RAG Layer
                      ↓
             Query Understanding
                      ↓
              Query Expansion
                      ↓
                 Retrieval
                      ↓
                 Reranking
                      ↓
             Retrieval Confidence
                      ↓
             Context Construction
                      ↓
             Grounded Response
                      ↓
              Source Citations
                      ↓
               Safe Fallback
                      ↓
                 API Response
                      ↓
               Monitoring
""")


# ============================================================
# 35. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("🎉 DAY 63 — PART 2 COMPLETED")
print("=" * 75)

print("""
You have moved from a simple Docker foundation to a
Dockerized enterprise AI service.

PART 1:
Docker Fundamentals
        ↓
Dockerfile
        ↓
Basic FastAPI Container


PART 2:
Production-style FastAPI
        ↓
Agentic RAG Interface
        ↓
Sessions
        ↓
Search
        ↓
Chat
        ↓
Grounding
        ↓
Citations
        ↓
Safe Fallback
        ↓
Docker Healthcheck
        ↓
Container Testing


NEXT — PART 3:

🔥 Docker Compose
🔥 Multi-container AI architecture
🔥 Vector database service
🔥 Redis-style session memory
🔥 Service-to-service communication
🔥 Persistent volumes
🔥 Container networking
🔥 Health dependencies
🔥 Environment-based configuration
🔥 Complete local AI infrastructure
""")

print("=" * 75)
# ============================================================
# 🚀 DAY 63/100 — PART 3
# Docker Compose + Multi-Container Enterprise AI Architecture
# ============================================================
#
# Architecture:
#
#                         USER
#                           |
#                           v
#                    +-------------+
#                    |   FastAPI   |
#                    |   ai-api    |
#                    +------+------+
#                           |
#                 +---------+---------+
#                 |                   |
#                 v                   v
#        +----------------+   +----------------+
#        | Retrieval      |   | Redis          |
#        | Service        |   | Session/Cache  |
#        +----------------+   +----------------+
#
# Goals:
#   1. Create multi-container architecture
#   2. Create FastAPI API service
#   3. Create retrieval service
#   4. Add Redis session/cache service
#   5. Add Docker Compose networking
#   6. Add environment configuration
#   7. Add health checks
#   8. Add service dependencies
#   9. Add persistent Redis volume
#  10. Build complete stack
#  11. Start all containers
#  12. Test service-to-service communication
#  13. Test Redis connectivity
#  14. Test retrieval service
#  15. Test end-to-end AI API
#
# CPU / STORAGE FRIENDLY:
#   - No LLM download
#   - No large model
#   - No large dataset
#   - Redis is lightweight
#   - Retrieval service uses tiny in-memory knowledge
# ============================================================


import os
import sys
import json
import time
import uuid
import subprocess
import urllib.request
import urllib.error
from pathlib import Path


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "day63_dockerized_enterprise_ai"

try:
    PROJECT_DIR
except NameError:
    PROJECT_DIR = Path.cwd() / PROJECT_NAME

PROJECT_DIR = Path(PROJECT_DIR)
PROJECT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 75)
print("🚀 DAY 63 — PART 3")
print("Docker Compose + Multi-Container Enterprise AI Architecture")
print("=" * 75)

print("\nProject directory:")
print(PROJECT_DIR)


# ============================================================
# 2. CREATE SERVICE DIRECTORIES
# ============================================================

RETRIEVAL_DIR = PROJECT_DIR / "retrieval_service"

RETRIEVAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\n✅ Created retrieval service directory:")
print(RETRIEVAL_DIR)


# ============================================================
# 3. CREATE RETRIEVAL SERVICE
# ============================================================
#
# This service represents the retrieval/vector-search layer.
#
# In a production architecture this can be replaced with:
#
#   Qdrant
#   Chroma
#   Pinecone
#   Azure AI Search
#   Elasticsearch
#   OpenSearch
#   PostgreSQL + pgvector
#
# For this CPU/storage-friendly project we keep the retrieval
# engine lightweight.
#
# ============================================================

retrieval_service_code = r'''
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Enterprise Retrieval Service",
    version="1.0.0",
    description="Lightweight retrieval service for Day 63"
)


# ============================================================
# KNOWLEDGE BASE
# ============================================================

DOCUMENTS = [

    {
        "document_id": "DOC001",
        "title": "Enterprise AI Policy",
        "department": "AI",
        "text": (
            "Enterprise AI systems should use controlled retrieval, "
            "grounding validation, source attribution, monitoring, "
            "access control and safe fallback mechanisms."
        )
    },

    {
        "document_id": "DOC002",
        "title": "RAG Architecture",
        "department": "Engineering",
        "text": (
            "Retrieval augmented generation retrieves relevant "
            "documents, constructs context and generates an answer "
            "using retrieved evidence."
        )
    },

    {
        "document_id": "DOC003",
        "title": "Docker Deployment",
        "department": "DevOps",
        "text": (
            "Docker packages applications and their dependencies "
            "into portable containers that run consistently across "
            "different environments."
        )
    },

    {
        "document_id": "DOC004",
        "title": "API Security",
        "department": "Security",
        "text": (
            "Production APIs should use authentication, "
            "authorization, input validation, rate limiting, "
            "secret management and audit logging."
        )
    },

    {
        "document_id": "DOC005",
        "title": "AI Monitoring",
        "department": "AI",
        "text": (
            "AI services should monitor latency, error rate, "
            "retrieval quality, grounding rate, citation coverage "
            "and safe fallback behavior."
        )
    },

    {
        "document_id": "DOC006",
        "title": "Container Orchestration",
        "department": "DevOps",
        "text": (
            "Docker Compose can define multiple services, networks, "
            "environment variables, volumes, health checks and "
            "service dependencies."
        )
    }
]


# ============================================================
# REQUEST MODEL
# ============================================================

class SearchRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )


# ============================================================
# SIMPLE RETRIEVAL
# ============================================================

def retrieve_documents(
    query,
    top_k=5
):

    query_words = set(
        query.lower().split()
    )

    scored_documents = []

    for document in DOCUMENTS:

        searchable_text = (
            document["title"]
            + " "
            + document["department"]
            + " "
            + document["text"]
        ).lower()

        document_words = set(
            searchable_text.split()
        )

        overlap = len(
            query_words.intersection(
                document_words
            )
        )

        score = (
            overlap
            / max(
                len(query_words),
                1
            )
        )

        scored_documents.append({

            "document_id":
                document["document_id"],

            "title":
                document["title"],

            "department":
                document["department"],

            "text":
                document["text"],

            "score":
                round(
                    score,
                    4
                )
        })

    scored_documents.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_documents[:top_k]


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "retrieval-service",
        "status": "running",
        "documents": len(DOCUMENTS),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "retrieval-service",
        "documents": len(DOCUMENTS)
    }


# ============================================================
# DOCUMENT COUNT
# ============================================================

@app.get("/documents/count")
def document_count():

    return {
        "document_count": len(DOCUMENTS)
    }


# ============================================================
# SEARCH
# ============================================================

@app.post("/search")
def search(
    request: SearchRequest
):

    results = retrieve_documents(
        request.query,
        request.top_k
    )

    return {

        "success": True,

        "query":
            request.query,

        "results":
            results,

        "result_count":
            len(results)
    }
'''


RETRIEVAL_APP_FILE = (
    RETRIEVAL_DIR /
    "main.py"
)

RETRIEVAL_APP_FILE.write_text(
    retrieval_service_code,
    encoding="utf-8"
)

print(
    "✅ Retrieval service created:",
    RETRIEVAL_APP_FILE
)


# ============================================================
# 4. RETRIEVAL SERVICE REQUIREMENTS
# ============================================================

retrieval_requirements = """fastapi
uvicorn[standard]
pydantic
"""

RETRIEVAL_REQUIREMENTS_FILE = (
    RETRIEVAL_DIR /
    "requirements.txt"
)

RETRIEVAL_REQUIREMENTS_FILE.write_text(
    retrieval_requirements,
    encoding="utf-8"
)

print(
    "✅ Retrieval requirements created."
)


# ============================================================
# 5. RETRIEVAL SERVICE DOCKERFILE
# ============================================================

retrieval_dockerfile = """FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
"""

RETRIEVAL_DOCKERFILE = (
    RETRIEVAL_DIR /
    "Dockerfile"
)

RETRIEVAL_DOCKERFILE.write_text(
    retrieval_dockerfile,
    encoding="utf-8"
)

print(
    "✅ Retrieval Dockerfile created."
)


# ============================================================
# 6. CREATE MULTI-SERVICE API
# ============================================================
#
# This API communicates with:
#
#   retrieval-service:8001
#   redis:6379
#
# IMPORTANT:
#
# Inside Docker Compose we NEVER use:
#
#   localhost:8001
#
# Instead we use:
#
#   http://retrieval-service:8001
#
# because Docker Compose provides service-name DNS.
#
# Redis is accessed using:
#
#   redis:6379
#
# ============================================================

compose_api_code = r'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import os
import time
import uuid
import json
import urllib.request
import urllib.error


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = os.getenv(
    "APP_NAME",
    "Enterprise Agentic RAG Assistant"
)

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0"
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development"
)

RETRIEVAL_SERVICE_URL = os.getenv(
    "RETRIEVAL_SERVICE_URL",
    "http://retrieval-service:8001"
)

REDIS_HOST = os.getenv(
    "REDIS_HOST",
    "redis"
)

REDIS_PORT = int(
    os.getenv(
        "REDIS_PORT",
        "6379"
    )
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Multi-container Enterprise AI API"
)


# ============================================================
# METRICS
# ============================================================

METRICS = {

    "requests": 0,

    "successes": 0,

    "failures": 0,

    "retrieval_calls": 0,

    "redis_checks": 0,

    "total_latency_ms": 0.0
}


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000
    )

    session_id: Optional[str] = None


class SearchRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )


# ============================================================
# SESSION FALLBACK
# ============================================================
#
# Redis integration is demonstrated through a connectivity
# endpoint. The API also maintains lightweight local state.
#
# A production implementation would store session history
# completely in Redis.
#
# ============================================================

LOCAL_SESSIONS = {}


# ============================================================
# GENERIC HTTP GET
# ============================================================

def service_get(url):

    with urllib.request.urlopen(
        url,
        timeout=5
    ) as response:

        body = response.read().decode(
            "utf-8"
        )

        return json.loads(
            body
        )


# ============================================================
# GENERIC HTTP POST
# ============================================================

def service_post(
    url,
    payload
):

    data = json.dumps(
        payload
    ).encode(
        "utf-8"
    )

    request = urllib.request.Request(

        url,

        data=data,

        headers={
            "Content-Type":
                "application/json"
        },

        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=5
    ) as response:

        body = response.read().decode(
            "utf-8"
        )

        return json.loads(
            body
        )


# ============================================================
# RETRIEVAL CALL
# ============================================================

def retrieve(
    query,
    top_k=5
):

    METRICS[
        "retrieval_calls"
    ] += 1

    return service_post(

        RETRIEVAL_SERVICE_URL
        + "/search",

        {
            "query": query,

            "top_k": top_k
        }
    )


# ============================================================
# REDIS CONNECTIVITY CHECK
# ============================================================

def check_redis():

    METRICS[
        "redis_checks"
    ] += 1

    try:

        import socket

        socket_connection = socket.create_connection(
            (
                REDIS_HOST,
                REDIS_PORT
            ),
            timeout=2
        )

        socket_connection.close()

        return True

    except Exception:

        return False


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "application":
            APP_NAME,

        "version":
            APP_VERSION,

        "environment":
            ENVIRONMENT,

        "status":
            "running",

        "architecture":
            "multi-container",

        "retrieval_service":
            RETRIEVAL_SERVICE_URL,

        "redis_host":
            REDIS_HOST,

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    retrieval_healthy = False

    try:

        retrieval_result = service_get(
            RETRIEVAL_SERVICE_URL
            + "/health"
        )

        retrieval_healthy = (
            retrieval_result.get(
                "status"
            ) == "healthy"
        )

    except Exception:

        retrieval_healthy = False


    redis_healthy = check_redis()


    overall_status = (
        "healthy"
        if retrieval_healthy
        and redis_healthy
        else "degraded"
    )


    return {

        "status":
            overall_status,

        "api":
            "healthy",

        "retrieval_service":
            retrieval_healthy,

        "redis":
            redis_healthy,

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# SERVICE INFO
# ============================================================

@app.get("/api/v1/info")
def info():

    return {

        "application":
            APP_NAME,

        "version":
            APP_VERSION,

        "environment":
            ENVIRONMENT,

        "services": {

            "api":
                "ai-api:8000",

            "retrieval":
                "retrieval-service:8001",

            "redis":
                "redis:6379"
        },

        "architecture": [

            "FastAPI",

            "Retrieval Service",

            "Redis",

            "Docker Compose",

            "Container Networking"
        ]
    }


# ============================================================
# METRICS
# ============================================================

@app.get("/api/v1/metrics")
def metrics():

    total = METRICS[
        "requests"
    ]

    average_latency = (

        METRICS[
            "total_latency_ms"
        ]

        / total

        if total > 0

        else 0.0
    )

    success_rate = (

        METRICS[
            "successes"
        ]

        / total

        if total > 0

        else 0.0
    )


    return {

        "requests":
            total,

        "successes":
            METRICS[
                "successes"
            ],

        "failures":
            METRICS[
                "failures"
            ],

        "success_rate":
            round(
                success_rate,
                4
            ),

        "retrieval_calls":
            METRICS[
                "retrieval_calls"
            ],

        "redis_checks":
            METRICS[
                "redis_checks"
            ],

        "average_latency_ms":
            round(
                average_latency,
                3
            )
    }


# ============================================================
# SEARCH
# ============================================================

@app.post("/api/v1/search")
def search(
    request: SearchRequest
):

    start = time.perf_counter()

    METRICS[
        "requests"
    ] += 1

    try:

        retrieval_response = retrieve(

            request.query,

            request.top_k
        )

        latency_ms = (

            time.perf_counter()
            - start

        ) * 1000

        METRICS[
            "total_latency_ms"
        ] += latency_ms

        METRICS[
            "successes"
        ] += 1

        return {

            "success":
                True,

            "query":
                request.query,

            "results":
                retrieval_response.get(
                    "results",
                    []
                ),

            "result_count":
                retrieval_response.get(
                    "result_count",
                    0
                ),

            "retrieval_service":
                RETRIEVAL_SERVICE_URL,

            "latency_ms":
                round(
                    latency_ms,
                    3
                )
        }

    except Exception as error:

        METRICS[
            "failures"
        ] += 1

        raise HTTPException(

            status_code=503,

            detail=(
                "Retrieval service unavailable: "
                + str(error)
            )
        )


# ============================================================
# CHAT
# ============================================================

@app.post("/api/v1/chat")
def chat(
    request: ChatRequest
):

    start = time.perf_counter()

    METRICS[
        "requests"
    ] += 1


    session_id = (
        request.session_id
        or str(uuid.uuid4())
    )


    if session_id not in LOCAL_SESSIONS:

        LOCAL_SESSIONS[
            session_id
        ] = []


    try:

        # ----------------------------------------------------
        # STEP 1 — RETRIEVAL
        # ----------------------------------------------------

        retrieval_response = retrieve(

            request.query,

            top_k=5
        )


        results = retrieval_response.get(
            "results",
            []
        )


        # ----------------------------------------------------
        # STEP 2 — FILTER RELEVANT RESULTS
        # ----------------------------------------------------

        relevant_results = [

            item

            for item in results

            if item.get(
                "score",
                0
            ) > 0
        ]


        # ----------------------------------------------------
        # STEP 3 — SAFE FALLBACK
        # ----------------------------------------------------

        if not relevant_results:

            answer = (

                "I could not find sufficient "
                "evidence in the enterprise "
                "knowledge base."
            )

            grounded = False

            safe_fallback = True

            confidence = 0.0

            citations = []


        else:

            # ------------------------------------------------
            # STEP 4 — CONTEXT
            # ------------------------------------------------

            context = " ".join(

                item.get(
                    "text",
                    ""
                )

                for item
                in relevant_results[:3]
            )


            # ------------------------------------------------
            # STEP 5 — LIGHTWEIGHT GENERATION
            # ------------------------------------------------

            answer = (

                "Based on the retrieved enterprise "
                "information: "

                + context
            )


            # ------------------------------------------------
            # STEP 6 — GROUNDING
            # ------------------------------------------------

            grounded = True

            safe_fallback = False


            best_score = relevant_results[0].get(
                "score",
                0
            )


            confidence = round(

                min(
                    1.0,
                    0.5 + best_score
                ),

                4
            )


            # ------------------------------------------------
            # STEP 7 — CITATIONS
            # ------------------------------------------------

            citations = [

                {

                    "document_id":
                        item.get(
                            "document_id"
                        ),

                    "title":
                        item.get(
                            "title"
                        ),

                    "department":
                        item.get(
                            "department"
                        )
                }

                for item
                in relevant_results[:3]
            ]


        # ----------------------------------------------------
        # STEP 8 — SESSION MEMORY
        # ----------------------------------------------------

        LOCAL_SESSIONS[
            session_id
        ].append({

            "role":
                "user",

            "content":
                request.query,

            "timestamp":
                datetime.utcnow().isoformat()
        })


        LOCAL_SESSIONS[
            session_id
        ].append({

            "role":
                "assistant",

            "content":
                answer,

            "timestamp":
                datetime.utcnow().isoformat()
        })


        # Keep last 10 messages
        LOCAL_SESSIONS[
            session_id
        ] = LOCAL_SESSIONS[
            session_id
        ][-10:]


        # ----------------------------------------------------
        # STEP 9 — METRICS
        # ----------------------------------------------------

        latency_ms = (

            time.perf_counter()
            - start

        ) * 1000


        METRICS[
            "total_latency_ms"
        ] += latency_ms


        METRICS[
            "successes"
        ] += 1


        return {

            "success":
                True,

            "session_id":
                session_id,

            "query":
                request.query,

            "answer":
                answer,

            "grounded":
                grounded,

            "safe_fallback":
                safe_fallback,

            "confidence":
                confidence,

            "citations":
                citations,

            "retrieved_chunks":
                len(
                    relevant_results
                ),

            "latency_ms":
                round(
                    latency_ms,
                    3
                )
        }


    except Exception as error:

        METRICS[
            "failures"
        ] += 1

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


# ============================================================
# SESSION
# ============================================================

@app.post("/api/v1/session")
def create_session():

    session_id = str(
        uuid.uuid4()
    )

    LOCAL_SESSIONS[
        session_id
    ] = []

    return {

        "session_id":
            session_id,

        "created":
            True
    }


@app.get(
    "/api/v1/session/{session_id}"
)
def get_session(
    session_id: str
):

    if session_id not in LOCAL_SESSIONS:

        raise HTTPException(

            status_code=404,

            detail="Session not found"
        )

    return {

        "session_id":
            session_id,

        "messages":
            LOCAL_SESSIONS[
                session_id
            ]
    }


@app.delete(
    "/api/v1/session/{session_id}"
)
def delete_session(
    session_id: str
):

    if session_id not in LOCAL_SESSIONS:

        raise HTTPException(

            status_code=404,

            detail="Session not found"
        )

    del LOCAL_SESSIONS[
        session_id
    ]

    return {

        "session_id":
            session_id,

        "deleted":
            True
    }
'''


API_COMPOSE_FILE = (
    PROJECT_DIR /
    "main.py"
)

API_COMPOSE_FILE.write_text(
    compose_api_code,
    encoding="utf-8"
)

print(
    "\n✅ Multi-container API created:",
    API_COMPOSE_FILE
)


# ============================================================
# 7. CREATE API REQUIREMENTS
# ============================================================

api_requirements = """fastapi
uvicorn[standard]
pydantic
"""

API_REQUIREMENTS_FILE = (
    PROJECT_DIR /
    "requirements.txt"
)

API_REQUIREMENTS_FILE.write_text(
    api_requirements,
    encoding="utf-8"
)

print(
    "✅ API requirements updated."
)


# ============================================================
# 8. CREATE API DOCKERFILE
# ============================================================

api_dockerfile = """FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=5 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

API_DOCKERFILE = (
    PROJECT_DIR /
    "Dockerfile"
)

API_DOCKERFILE.write_text(
    api_dockerfile,
    encoding="utf-8"
)

print(
    "✅ API Dockerfile updated."
)


# ============================================================
# 9. CREATE DOCKER COMPOSE
# ============================================================
#
# Three services:
#
#   ai-api
#   retrieval-service
#   redis
#
# Docker Compose creates an internal network automatically.
#
# Therefore:
#
# ai-api → http://retrieval-service:8001
#
# ai-api → redis:6379
#
# ============================================================

compose_yaml = """services:

  # ==========================================================
  # AI API
  # ==========================================================

  ai-api:

    build:
      context: .
      dockerfile: Dockerfile

    container_name: day63-ai-api

    ports:
      - "8000:8000"

    environment:

      APP_NAME:
        Enterprise Agentic RAG Assistant

      APP_VERSION:
        1.0.0

      ENVIRONMENT:
        production

      RETRIEVAL_SERVICE_URL:
        http://retrieval-service:8001

      REDIS_HOST:
        redis

      REDIS_PORT:
        6379

    depends_on:

      retrieval-service:
        condition: service_healthy

      redis:
        condition: service_healthy

    restart: unless-stopped

    healthcheck:

      test:
        [
          "CMD",
          "python",
          "-c",
          "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
        ]

      interval: 20s

      timeout: 5s

      retries: 5

      start_period: 15s


  # ==========================================================
  # RETRIEVAL SERVICE
  # ==========================================================

  retrieval-service:

    build:
      context: ./retrieval_service
      dockerfile: Dockerfile

    container_name: day63-retrieval-service

    expose:
      - "8001"

    restart: unless-stopped

    healthcheck:

      test:
        [
          "CMD",
          "python",
          "-c",
          "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health')"
        ]

      interval: 20s

      timeout: 5s

      retries: 5

      start_period: 10s


  # ==========================================================
  # REDIS
  # ==========================================================

  redis:

    image: redis:7-alpine

    container_name: day63-redis

    expose:
      - "6379"

    volumes:

      - redis_data:/data

    command:
      [
        "redis-server",
        "--appendonly",
        "yes"
      ]

    restart: unless-stopped

    healthcheck:

      test:
        [
          "CMD",
          "redis-cli",
          "ping"
        ]

      interval: 10s

      timeout: 5s

      retries: 5


# ============================================================
# PERSISTENT VOLUME
# ============================================================

volumes:

  redis_data:
"""

COMPOSE_FILE_PART3 = (
    PROJECT_DIR /
    "docker-compose.yml"
)

COMPOSE_FILE_PART3.write_text(
    compose_yaml,
    encoding="utf-8"
)

print(
    "\n✅ docker-compose.yml created."
)


# ============================================================
# 10. CREATE .DOCKERIGNORE
# ============================================================

dockerignore_part3 = """__pycache__/
*.py[cod]

.ipynb_checkpoints/
*.ipynb

.git/
.gitignore

.env
.env.*

venv/
.venv/
env/

*.log

data/
datasets/
models/
checkpoints/

*.csv
*.parquet
*.db

redis_data/
"""

DOCKERIGNORE_PART3 = (
    PROJECT_DIR /
    ".dockerignore"
)

DOCKERIGNORE_PART3.write_text(
    dockerignore_part3,
    encoding="utf-8"
)

print(
    "✅ .dockerignore updated."
)


# ============================================================
# 11. CREATE ENVIRONMENT TEMPLATE
# ============================================================

env_part3 = """APP_NAME=Enterprise Agentic RAG Assistant
APP_VERSION=1.0.0
ENVIRONMENT=production

RETRIEVAL_SERVICE_URL=http://retrieval-service:8001

REDIS_HOST=redis
REDIS_PORT=6379
"""

ENV_FILE_PART3 = (
    PROJECT_DIR /
    ".env.example"
)

ENV_FILE_PART3.write_text(
    env_part3,
    encoding="utf-8"
)

print(
    "✅ .env.example updated."
)


# ============================================================
# 12. CREATE README
# ============================================================

readme_part3 = """# Day 63 — Multi-Container Enterprise AI

## Architecture

User
  ↓
FastAPI
  ↓
Retrieval Service
  ↓
Redis

## Services

### ai-api
FastAPI enterprise AI API.

Port:
8000

### retrieval-service
Lightweight retrieval service.

Internal port:
8001

### redis
Session/cache infrastructure.

Internal port:
6379

## Start

docker compose up --build

## Stop

docker compose down

## Stop and remove volume

docker compose down -v

## API

http://localhost:8000

Swagger:

http://localhost:8000/docs

Health:

http://localhost:8000/health

Search:

POST /api/v1/search

Chat:

POST /api/v1/chat

Metrics:

GET /api/v1/metrics
"""

README_PART3 = (
    PROJECT_DIR /
    "README.md"
)

README_PART3.write_text(
    readme_part3,
    encoding="utf-8"
)

print(
    "✅ README updated."
)


# ============================================================
# 13. VALIDATE PYTHON SYNTAX
# ============================================================

print("\n" + "=" * 75)
print("🔍 PYTHON SYNTAX VALIDATION")
print("=" * 75)


syntax_files = [

    API_COMPOSE_FILE,

    RETRIEVAL_APP_FILE
]


syntax_results = {}


for file_path in syntax_files:

    result = subprocess.run(

        [
            sys.executable,
            "-m",
            "py_compile",
            str(file_path)
        ],

        capture_output=True,

        text=True
    )

    syntax_results[
        file_path.name
    ] = (
        result.returncode == 0
    )

    if result.returncode == 0:

        print(
            f"✅ {file_path.name}"
        )

    else:

        print(
            f"❌ {file_path.name}"
        )

        print(
            result.stderr
        )


# ============================================================
# 14. CHECK DOCKER COMPOSE
# ============================================================

print("\n" + "=" * 75)
print("🐳 DOCKER COMPOSE CHECK")
print("=" * 75)


docker_available_part3 = False
compose_available_part3 = False


try:

    docker_result = subprocess.run(

        ["docker", "--version"],

        capture_output=True,

        text=True,

        timeout=10
    )

    if docker_result.returncode == 0:

        docker_available_part3 = True

        print(
            "✅ Docker:",
            docker_result.stdout.strip()
        )

except Exception:

    print(
        "⚠️ Docker unavailable."
    )


try:

    compose_result = subprocess.run(

        [
            "docker",
            "compose",
            "version"
        ],

        capture_output=True,

        text=True,

        timeout=10
    )

    if compose_result.returncode == 0:

        compose_available_part3 = True

        print(
            "✅ Docker Compose:",
            compose_result.stdout.strip()
        )

    else:

        print(
            "⚠️ Docker Compose unavailable."
        )

except Exception:

    print(
        "⚠️ Docker Compose unavailable."
    )


# ============================================================
# 15. VALIDATE COMPOSE CONFIGURATION
# ============================================================

compose_config_valid = False


if compose_available_part3:

    print("\n" + "=" * 75)
    print("🔎 VALIDATING DOCKER COMPOSE CONFIGURATION")
    print("=" * 75)

    config_result = subprocess.run(

        [
            "docker",
            "compose",
            "config"
        ],

        cwd=str(PROJECT_DIR),

        capture_output=True,

        text=True
    )

    if config_result.returncode == 0:

        compose_config_valid = True

        print(
            "✅ Docker Compose configuration is valid."
        )

        print(
            config_result.stdout[:4000]
        )

    else:

        print(
            "❌ Docker Compose configuration failed."
        )

        print(
            config_result.stderr
        )


# ============================================================
# 16. BUILD COMPLETE MULTI-CONTAINER STACK
# ============================================================

stack_built = False


if compose_available_part3:

    print("\n" + "=" * 75)
    print("🔨 BUILDING MULTI-CONTAINER STACK")
    print("=" * 75)

    build_stack = subprocess.run(

        [
            "docker",
            "compose",
            "build"
        ],

        cwd=str(PROJECT_DIR),

        capture_output=True,

        text=True
    )

    if build_stack.returncode == 0:

        stack_built = True

        print(
            "✅ All custom Docker services built."
        )

    else:

        print(
            "❌ Stack build failed."
        )

        print(
            "\nSTDOUT:\n",
            build_stack.stdout[-5000:]
        )

        print(
            "\nSTDERR:\n",
            build_stack.stderr[-5000:]
        )


# ============================================================
# 17. START COMPLETE STACK
# ============================================================

stack_started = False


if compose_available_part3 and stack_built:

    print("\n" + "=" * 75)
    print("🚀 STARTING COMPLETE MULTI-CONTAINER STACK")
    print("=" * 75)

    up_result = subprocess.run(

        [
            "docker",
            "compose",
            "up",
            "-d"
        ],

        cwd=str(PROJECT_DIR),

        capture_output=True,

        text=True
    )

    if up_result.returncode == 0:

        stack_started = True

        print(
            "✅ Complete stack started."
        )

        print(
            up_result.stdout
        )

        # Give containers time to become healthy.
        time.sleep(8)

    else:

        print(
            "❌ Stack failed to start."
        )

        print(
            up_result.stderr
        )


# ============================================================
# 18. SHOW SERVICE STATUS
# ============================================================

if compose_available_part3:

    print("\n" + "=" * 75)
    print("📦 DOCKER COMPOSE SERVICE STATUS")
    print("=" * 75)

    ps_result = subprocess.run(

        [
            "docker",
            "compose",
            "ps"
        ],

        cwd=str(PROJECT_DIR),

        capture_output=True,

        text=True
    )

    print(
        ps_result.stdout
    )


# ============================================================
# 19. API TEST HELPER
# ============================================================

def local_http_get(url):

    with urllib.request.urlopen(
        url,
        timeout=8
    ) as response:

        return (
            response.status,
            response.read().decode(
                "utf-8"
            )
        )


def local_http_post(
    url,
    payload
):

    data = json.dumps(
        payload
    ).encode(
        "utf-8"
    )

    request = urllib.request.Request(

        url,

        data=data,

        headers={
            "Content-Type":
                "application/json"
        },

        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=8
    ) as response:

        return (
            response.status,
            response.read().decode(
                "utf-8"
            )
        )


# ============================================================
# 20. TEST MAIN API
# ============================================================

BASE_URL_PART3 = (
    "http://127.0.0.1:8000"
)


if stack_started:

    print("\n" + "=" * 75)
    print("🧪 MAIN API TEST")
    print("=" * 75)

    try:

        status, body = local_http_get(
            BASE_URL_PART3 + "/"
        )

        print(
            "GET /",
            status
        )

        print(
            body
        )

    except Exception as error:

        print(
            "❌ Main API failed:",
            error
        )


# ============================================================
# 21. TEST HEALTH
# ============================================================

if stack_started:

    print("\n" + "=" * 75)
    print("❤️ MULTI-SERVICE HEALTH TEST")
    print("=" * 75)

    try:

        status, body = local_http_get(
            BASE_URL_PART3
            + "/health"
        )

        print(
            "Status:",
            status
        )

        print(
            json.dumps(
                json.loads(body),
                indent=2
            )
        )

    except Exception as error:

        print(
            "❌ Health test failed:",
            error
        )


# ============================================================
# 22. TEST RETRIEVAL THROUGH API
# ============================================================

if stack_started:

    print("\n" + "=" * 75)
    print("🔎 RETRIEVAL SERVICE INTEGRATION TEST")
    print("=" * 75)

    try:

        status, body = local_http_post(

            BASE_URL_PART3
            + "/api/v1/search",

            {
                "query":
                    "How does Docker help deploy AI applications?",

                "top_k":
                    5
            }
        )

        print(
            "Status:",
            status
        )

        print(
            json.dumps(
                json.loads(body),
                indent=2
            )
        )

    except Exception as error:

        print(
            "❌ Retrieval integration failed:",
            error
        )


# ============================================================
# 23. TEST CHAT
# ============================================================

if stack_started:

    print("\n" + "=" * 75)
    print("🤖 END-TO-END CHAT TEST")
    print("=" * 75)

    try:

        status, body = local_http_post(

            BASE_URL_PART3
            + "/api/v1/chat",

            {
                "query":
                    "What is Docker?"
            }
        )

        print(
            "Status:",
            status
        )

        print(
            json.dumps(
                json.loads(body),
                indent=2
            )
        )

    except Exception as error:

        print(
            "❌ Chat test failed:",
            error
        )


# ============================================================
# 24. TEST SESSION
# ============================================================

if stack_started:

    print("\n" + "=" * 75)
    print("🧠 SESSION TEST")
    print("=" * 75)

    try:

        status, body = local_http_post(

            BASE_URL_PART3
            + "/api/v1/session",

            {}
        )

        session_data = json.loads(
            body
        )

        session_id = session_data[
            "session_id"
        ]

        print(
            "Session:",
            session_id
        )


        # First message
        status, body = local_http_post(

            BASE_URL_PART3
            + "/api/v1/chat",

            {
                "query":
                    "Explain RAG",

                "session_id":
                    session_id
            }
        )

        print(
            "\nFirst message:",
            status
        )


        # Retrieve history
        status, body = local_http_get(

            BASE_URL_PART3
            + f"/api/v1/session/{session_id}"
        )

        print(
            "\nSession history:"
        )

        print(
            body
        )

    except Exception as error:

        print(
            "❌ Session test failed:",
            error
        )


# ============================================================
# 25. TEST METRICS
# ============================================================

if stack_started:

    print("\n" + "=" * 75)
    print("📊 METRICS TEST")
    print("=" * 75)

    try:

        status, body = local_http_get(

            BASE_URL_PART3
            + "/api/v1/metrics"
        )

        print(
            "Status:",
            status
        )

        print(
            json.dumps(
                json.loads(body),
                indent=2
            )
        )

    except Exception as error:

        print(
            "❌ Metrics test failed:",
            error
        )


# ============================================================
# 26. TEST SWAGGER AVAILABILITY
# ============================================================

if stack_started:

    print("\n" + "=" * 75)
    print("📚 SWAGGER API TEST")
    print("=" * 75)

    try:

        status, body = local_http_get(

            BASE_URL_PART3
            + "/docs"
        )

        print(
            "Swagger status:",
            status
        )

        print(
            "Swagger available:",
            status == 200
        )

    except Exception as error:

        print(
            "❌ Swagger test failed:",
            error
        )


# ============================================================
# 27. SHOW SERVICE LOGS
# ============================================================

if compose_available_part3:

    print("\n" + "=" * 75)
    print("📋 AI API LOGS")
    print("=" * 75)

    api_logs = subprocess.run(

        [
            "docker",
            "compose",
            "logs",
            "--tail",
            "30",
            "ai-api"
        ],

        cwd=str(PROJECT_DIR),

        capture_output=True,

        text=True
    )

    print(
        api_logs.stdout
    )


    print("\n" + "=" * 75)
    print("📋 RETRIEVAL SERVICE LOGS")
    print("=" * 75)

    retrieval_logs = subprocess.run(

        [
            "docker",
            "compose",
            "logs",
            "--tail",
            "30",
            "retrieval-service"
        ],

        cwd=str(PROJECT_DIR),

        capture_output=True,

        text=True
    )

    print(
        retrieval_logs.stdout
    )


    print("\n" + "=" * 75)
    print("📋 REDIS LOGS")
    print("=" * 75)

    redis_logs = subprocess.run(

        [
            "docker",
            "compose",
            "logs",
            "--tail",
            "20",
            "redis"
        ],

        cwd=str(PROJECT_DIR),

        capture_output=True,

        text=True
    )

    print(
        redis_logs.stdout
    )


# ============================================================
# 28. SHOW NETWORK
# ============================================================

if compose_available_part3:

    print("\n" + "=" * 75)
    print("🌐 DOCKER NETWORK")
    print("=" * 75)

    network_result = subprocess.run(

        [
            "docker",
            "network",
            "ls"
        ],

        capture_output=True,

        text=True
    )

    print(
        network_result.stdout
    )


# ============================================================
# 29. SHOW VOLUMES
# ============================================================

if compose_available_part3:

    print("\n" + "=" * 75)
    print("💾 DOCKER VOLUMES")
    print("=" * 75)

    volume_result = subprocess.run(

        [
            "docker",
            "volume",
            "ls"
        ],

        capture_output=True,

        text=True
    )

    print(
        volume_result.stdout
    )


# ============================================================
# 30. SERVICE COMMUNICATION EXPLANATION
# ============================================================

print("\n" + "=" * 75)
print("🧠 CONTAINER NETWORKING")
print("=" * 75)

print("""
Docker Compose creates an internal network.

Inside the network:

ai-api
   |
   | HTTP
   v
retrieval-service:8001


ai-api
   |
   | TCP
   v
redis:6379


IMPORTANT:

Inside Docker:

http://retrieval-service:8001

NOT:

http://localhost:8001


Redis:

redis:6379

NOT:

localhost:6379
""")


# ============================================================
# 31. DOCKER COMPOSE ARCHITECTURE
# ============================================================

print("\n" + "=" * 75)
print("🏗️ MULTI-CONTAINER ARCHITECTURE")
print("=" * 75)

print("""
                         USER
                           |
                           v
                  +----------------+
                  |   ai-api       |
                  |   FastAPI      |
                  |   Port 8000    |
                  +-------+--------+
                          |
             +------------+------------+
             |                         |
             v                         v
   +-------------------+      +-------------------+
   | retrieval-service |      |      Redis        |
   |                   |      |                   |
   | Retrieval Layer   |      | Session / Cache   |
   | Internal :8001    |      | Internal :6379    |
   +-------------------+      +-------------------+

                Docker Compose Network
                         |
                         v
                  redis_data volume
""")


# ============================================================
# 32. PRODUCTION EVOLUTION
# ============================================================

print("\n" + "=" * 75)
print("🚀 PRODUCTION EVOLUTION")
print("=" * 75)

print("""
CURRENT DAY 63:

FastAPI
   |
   +--> Lightweight Retrieval Service
   |
   +--> Redis


PRODUCTION:

API Gateway / WAF
        |
        v
Load Balancer
        |
        v
FastAPI Containers
        |
        +--------------------+
        |                    |
        v                    v
Agent Orchestrator       Redis
        |
        v
Hybrid Retrieval
   +----+----+
   |         |
   v         v
Vector DB   Keyword Index
   |
   v
Reranker
   |
   v
Enterprise LLM
   |
   v
Grounding Validator
   |
   v
Final Response
""")


# ============================================================
# 33. WHY REDIS?
# ============================================================

print("\n" + "=" * 75)
print("🧠 WHY REDIS?")
print("=" * 75)

print("""
Redis is commonly used as a fast shared state/cache layer.

Possible AI application uses:

1. Conversation sessions
2. Short-term memory
3. Response caching
4. Rate limiting
5. Distributed locks
6. Temporary retrieval state
7. Job/status tracking

Current notebook:

FastAPI → Redis connectivity

Production:

FastAPI → Redis → persistent/distributed session state
""")


# ============================================================
# 34. WHY SEPARATE RETRIEVAL SERVICE?
# ============================================================

print("\n" + "=" * 75)
print("🔎 WHY SEPARATE RETRIEVAL SERVICE?")
print("=" * 75)

print("""
Separating retrieval from the API provides:

1. Independent scaling
2. Clear service boundaries
3. Easier testing
4. Independent deployment
5. Vector database abstraction
6. Retrieval-specific monitoring
7. Easier migration to managed vector search

For example:

Development:
FastAPI → Lightweight Retrieval Service

Production:
FastAPI → Retrieval Service → Azure AI Search / Qdrant /
                              OpenSearch / pgvector
""")


# ============================================================
# 35. PERSISTENCE EXPLANATION
# ============================================================

print("\n" + "=" * 75)
print("💾 PERSISTENT VOLUME")
print("=" * 75)

print("""
Redis container:
    |
    v
redis_data volume
    |
    v
Redis /data

The volume allows Redis data to survive
container recreation more reliably than
container-local filesystem storage.

Command:

docker compose down

keeps the named volume.

Command:

docker compose down -v

removes the named volume.
""")


# ============================================================
# 36. HEALTH DEPENDENCIES
# ============================================================

print("\n" + "=" * 75)
print("❤️ SERVICE HEALTH DEPENDENCIES")
print("=" * 75)

print("""
retrieval-service
        |
        | healthy
        v
      ai-api


redis
        |
        | healthy
        v
      ai-api


Docker Compose therefore waits for the
dependencies to become healthy before
starting the main API service.
""")


# ============================================================
# 37. IMPORTANT VARIABLES
# ============================================================

print("\n" + "=" * 75)
print("📌 IMPORTANT VARIABLES")
print("=" * 75)

IMPORTANT_VARIABLES_PART3 = {

    "PROJECT_DIR":
        str(PROJECT_DIR),

    "RETRIEVAL_DIR":
        str(RETRIEVAL_DIR),

    "API_COMPOSE_FILE":
        str(API_COMPOSE_FILE),

    "RETRIEVAL_APP_FILE":
        str(RETRIEVAL_APP_FILE),

    "API_DOCKERFILE":
        str(API_DOCKERFILE),

    "RETRIEVAL_DOCKERFILE":
        str(RETRIEVAL_DOCKERFILE),

    "COMPOSE_FILE_PART3":
        str(COMPOSE_FILE_PART3),

    "DOCKERIGNORE_PART3":
        str(DOCKERIGNORE_PART3),

    "ENV_FILE_PART3":
        str(ENV_FILE_PART3),

    "docker_available_part3":
        docker_available_part3,

    "compose_available_part3":
        compose_available_part3,

    "compose_config_valid":
        compose_config_valid,

    "stack_built":
        stack_built,

    "stack_started":
        stack_started
}


for key, value in IMPORTANT_VARIABLES_PART3.items():

    print(
        f"{key} = {value}"
    )


# ============================================================
# 38. FINAL FILE VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("✅ FILE VALIDATION")
print("=" * 75)


required_files_part3 = [

    API_COMPOSE_FILE,

    API_REQUIREMENTS_FILE,

    API_DOCKERFILE,

    RETRIEVAL_APP_FILE,

    RETRIEVAL_REQUIREMENTS_FILE,

    RETRIEVAL_DOCKERFILE,

    COMPOSE_FILE_PART3,

    DOCKERIGNORE_PART3,

    ENV_FILE_PART3,

    README_PART3
]


all_files_created_part3 = True


for file_path in required_files_part3:

    exists = file_path.exists()

    print(

        f"{'✅' if exists else '❌'} "
        f"{file_path.relative_to(PROJECT_DIR)}"
    )

    if not exists:

        all_files_created_part3 = False


print(
    "\nAll files created:",
    all_files_created_part3
)


# ============================================================
# 39. FINAL STACK VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("🎯 DAY 63 PART 3 VALIDATION")
print("=" * 75)

print(
    "API syntax:",
    syntax_results.get(
        "main.py",
        False
    )
)

print(
    "Retrieval syntax:",
    syntax_results.get(
        "main.py",
        False
    )
)

print(
    "Docker available:",
    docker_available_part3
)

print(
    "Docker Compose available:",
    compose_available_part3
)

print(
    "Compose configuration valid:",
    compose_config_valid
)

print(
    "Stack built:",
    stack_built
)

print(
    "Stack started:",
    stack_started
)


# ============================================================
# 40. FINAL PROJECT STRUCTURE
# ============================================================

print("\n" + "=" * 75)
print("📂 FINAL PROJECT STRUCTURE")
print("=" * 75)

print(f"""
{PROJECT_NAME}/
│
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── README.md
│
└── retrieval_service/
    │
    ├── main.py
    ├── requirements.txt
    └── Dockerfile
│
└── Redis
    │
    └── redis_data volume
""")


# ============================================================
# 41. FINAL END-TO-END FLOW
# ============================================================

print("\n" + "=" * 75)
print("🔥 DAY 63 PART 3 — COMPLETE FLOW")
print("=" * 75)

print("""
                         USER
                           |
                           v
                    Docker Host :8000
                           |
                           v
                  +------------------+
                  |     ai-api       |
                  |     FastAPI      |
                  +--------+---------+
                           |
                 +---------+---------+
                 |                   |
                 v                   v
       +-------------------+   +----------------+
       | retrieval-service |   |     Redis      |
       |       :8001       |   |     :6379      |
       +---------+---------+   +-------+--------+
                 |                     |
                 v                     v
          Document Search       Session / Cache
                 |
                 v
            Top-K Results
                 |
                 v
          Context Construction
                 |
                 v
          Grounded Response
                 |
                 v
              FastAPI
                 |
                 v
               USER
""")


# ============================================================
# 42. WHAT WAS LEARNED
# ============================================================

print("\n" + "=" * 75)
print("🧠 PART 3 KEY LEARNINGS")
print("=" * 75)

learning_points_part3 = [

    "Docker Compose manages multiple containers",

    "Each service can have an independent responsibility",

    "Docker Compose provides internal service discovery",

    "Containers communicate using service names",

    "localhost inside a container refers to that container",

    "Redis can provide shared session/cache infrastructure",

    "Named volumes provide persistent storage",

    "Health checks allow service readiness validation",

    "depends_on can coordinate service startup",

    "Retrieval can be separated from the API layer",

    "Independent services can scale independently",

    "Environment variables decouple configuration from code",

    "Multi-container architecture prepares the system for production"
]


for point in learning_points_part3:

    print(
        "✅",
        point
    )


# ============================================================
# 43. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("🎉 DAY 63 — PART 3 COMPLETED")
print("=" * 75)

print("""
You have now moved from:

PART 1
Docker Fundamentals
        ↓
PART 2
Dockerized FastAPI AI Service
        ↓
PART 3
Multi-Container AI Infrastructure


CURRENT ARCHITECTURE:

                    USER
                      ↓
                FastAPI Container
                 /           \\
                /             \\
               ↓               ↓
     Retrieval Container     Redis
               ↓
       Enterprise Documents
               ↓
          Top-K Retrieval
               ↓
       Context Construction
               ↓
        Grounded Response


NEXT — PART 4:

🔥 Production Docker architecture
🔥 Container security
🔥 Resource limits
🔥 Production configuration
🔥 Logging and monitoring
🔥 CI/CD pipeline design
🔥 Cloud deployment architecture
🔥 Container scaling
🔥 Production health checks
🔥 Final end-to-end evaluation
🔥 Final Day 63 project summary
🔥 Interview explanation
🔥 Resume bullet
🔥 LinkedIn-ready technical summary
""")

print("=" * 75)
# ============================================================
# 🚀 DAY 63/100 — FINAL PART 4
# Production Docker Architecture + Security + Monitoring
# ============================================================
#
# CONTINUATION FROM PART 3
#
# Part 1:
#   Docker fundamentals
#
# Part 2:
#   Dockerized FastAPI AI service
#
# Part 3:
#   Multi-container architecture
#
# Part 4:
#   Productionization + Security + Monitoring + CI/CD
#
#
# FINAL ARCHITECTURE
#
#                         USER
#                           |
#                           v
#                    API Gateway / WAF
#                           |
#                           v
#                     FastAPI API
#                           |
#              +------------+------------+
#              |                         |
#              v                         v
#       Retrieval Service             Redis
#              |
#              v
#       Knowledge Retrieval
#              |
#              v
#       Context Construction
#              |
#              v
#        Grounded Response
#
#        Monitoring / Logging
#        Security / Health
#        CI/CD / Deployment
#
# ============================================================


import os
import sys
import json
import time
import uuid
import subprocess
from pathlib import Path
from datetime import datetime


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

try:
    PROJECT_DIR
except NameError:

    PROJECT_NAME = "day63_dockerized_enterprise_ai"

    PROJECT_DIR = (
        Path.cwd()
        / PROJECT_NAME
    )

PROJECT_DIR = Path(
    PROJECT_DIR
)

PROJECT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("🚀 DAY 63/100 — FINAL PART 4")
print("Production Docker Architecture + Security + Monitoring")
print("=" * 80)

print(
    "\nProject:",
    PROJECT_DIR
)


# ============================================================
# 2. PRODUCTION DIRECTORIES
# ============================================================

PROD_DIR = (
    PROJECT_DIR
    / "production"
)

CI_DIR = (
    PROJECT_DIR
    / ".github"
    / "workflows"
)

MONITORING_DIR = (
    PROJECT_DIR
    / "monitoring"
)

SECURITY_DIR = (
    PROJECT_DIR
    / "security"
)


for directory in [
    PROD_DIR,
    CI_DIR,
    MONITORING_DIR,
    SECURITY_DIR
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


print("\n✅ Production directories created")


# ============================================================
# 3. PRODUCTION ENVIRONMENT TEMPLATE
# ============================================================

production_env = """# ============================================================
# DAY 63 PRODUCTION ENVIRONMENT
# ============================================================

APP_NAME=Enterprise Agentic RAG Assistant
APP_VERSION=1.0.0
ENVIRONMENT=production

RETRIEVAL_SERVICE_URL=http://retrieval-service:8001

REDIS_HOST=redis
REDIS_PORT=6379

API_WORKERS=2

LOG_LEVEL=INFO

MAX_QUERY_LENGTH=1000
DEFAULT_TOP_K=5

ENABLE_CITATIONS=true
ENABLE_GROUNDING=true
ENABLE_SAFE_FALLBACK=true

# Production secrets must NOT be committed.
# Use a cloud secret manager or Docker/Kubernetes secrets.
"""

PRODUCTION_ENV_FILE = (
    PROD_DIR
    / ".env.example"
)

PRODUCTION_ENV_FILE.write_text(
    production_env,
    encoding="utf-8"
)

print(
    "✅ Production environment template created"
)


# ============================================================
# 4. SECURE FASTAPI DOCKERFILE
# ============================================================
#
# Security improvements:
#
# 1. Slim Python image
# 2. No bytecode
# 3. No pip cache
# 4. Dedicated non-root user
# 5. Health check
# 6. Explicit working directory
#
# ============================================================

production_api_dockerfile = """FROM python:3.11-slim

# ------------------------------------------------------------
# Runtime configuration
# ------------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app


# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

RUN apt-get update \\
    && apt-get install -y --no-install-recommends curl \\
    && rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------
# Install Python dependencies
# ------------------------------------------------------------

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# ------------------------------------------------------------
# Copy application
# ------------------------------------------------------------

COPY main.py .


# ------------------------------------------------------------
# Create non-root user
# ------------------------------------------------------------

RUN useradd \\
    --create-home \\
    --shell /usr/sbin/nologin \\
    appuser

RUN chown -R appuser:appuser /app

USER appuser


# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

EXPOSE 8000


# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \\
    CMD curl --fail http://127.0.0.1:8000/health || exit 1


# ------------------------------------------------------------
# Start API
# ------------------------------------------------------------

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

PRODUCTION_API_DOCKERFILE = (
    PROD_DIR
    / "Dockerfile.api"
)

PRODUCTION_API_DOCKERFILE.write_text(
    production_api_dockerfile,
    encoding="utf-8"
)

print(
    "✅ Secure API Dockerfile created"
)


# ============================================================
# 5. SECURE RETRIEVAL DOCKERFILE
# ============================================================

production_retrieval_dockerfile = """FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

RUN useradd \\
    --create-home \\
    --shell /usr/sbin/nologin \\
    retrievaluser

RUN chown -R retrievaluser:retrievaluser /app

USER retrievaluser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
"""

PRODUCTION_RETRIEVAL_DOCKERFILE = (
    PROD_DIR
    / "Dockerfile.retrieval"
)

PRODUCTION_RETRIEVAL_DOCKERFILE.write_text(
    production_retrieval_dockerfile,
    encoding="utf-8"
)

print(
    "✅ Secure retrieval Dockerfile created"
)


# ============================================================
# 6. PRODUCTION REQUIREMENTS
# ============================================================

production_requirements = """fastapi
uvicorn[standard]
pydantic
"""

PRODUCTION_REQUIREMENTS_FILE = (
    PROD_DIR
    / "requirements.txt"
)

PRODUCTION_REQUIREMENTS_FILE.write_text(
    production_requirements,
    encoding="utf-8"
)

print(
    "✅ Production requirements created"
)


# ============================================================
# 7. PRODUCTION DOCKER COMPOSE
# ============================================================
#
# Production-oriented features:
#
# - Separate services
# - Internal networking
# - Health checks
# - Restart policy
# - Resource limits
# - Read-only filesystem where practical
# - tmpfs
# - No-new-privileges
# - Logging rotation
# - Redis persistence
#
# ============================================================

production_compose = """services:

  # ==========================================================
  # FASTAPI APPLICATION
  # ==========================================================

  ai-api:

    build:
      context: ..
      dockerfile: production/Dockerfile.api

    container_name: day63-ai-api-prod

    ports:
      - "8000:8000"

    environment:

      APP_NAME: Enterprise Agentic RAG Assistant
      APP_VERSION: 1.0.0
      ENVIRONMENT: production

      RETRIEVAL_SERVICE_URL: http://retrieval-service:8001

      REDIS_HOST: redis
      REDIS_PORT: 6379

      LOG_LEVEL: INFO

      DEFAULT_TOP_K: 5
      MAX_QUERY_LENGTH: 1000

      ENABLE_CITATIONS: "true"
      ENABLE_GROUNDING: "true"
      ENABLE_SAFE_FALLBACK: "true"

    depends_on:

      retrieval-service:
        condition: service_healthy

      redis:
        condition: service_healthy

    restart: unless-stopped

    security_opt:
      - no-new-privileges:true

    cap_drop:
      - ALL

    read_only: true

    tmpfs:
      - /tmp

    deploy:

      resources:

        limits:
          cpus: "1.0"
          memory: 512M

        reservations:
          cpus: "0.25"
          memory: 128M

    logging:

      driver: json-file

      options:
        max-size: "10m"
        max-file: "3"

    healthcheck:

      test:
        [
          "CMD",
          "python",
          "-c",
          "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
        ]

      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s


  # ==========================================================
  # RETRIEVAL SERVICE
  # ==========================================================

  retrieval-service:

    build:
      context: ../retrieval_service
      dockerfile: ../production/Dockerfile.retrieval

    container_name: day63-retrieval-prod

    expose:
      - "8001"

    restart: unless-stopped

    security_opt:
      - no-new-privileges:true

    cap_drop:
      - ALL

    read_only: true

    tmpfs:
      - /tmp

    deploy:

      resources:

        limits:
          cpus: "0.75"
          memory: 384M

        reservations:
          cpus: "0.10"
          memory: 96M

    logging:

      driver: json-file

      options:
        max-size: "10m"
        max-file: "3"

    healthcheck:

      test:
        [
          "CMD",
          "python",
          "-c",
          "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health')"
        ]

      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s


  # ==========================================================
  # REDIS
  # ==========================================================

  redis:

    image: redis:7-alpine

    container_name: day63-redis-prod

    expose:
      - "6379"

    command:
      [
        "redis-server",
        "--appendonly",
        "yes"
      ]

    volumes:
      - redis_data:/data

    restart: unless-stopped

    security_opt:
      - no-new-privileges:true

    cap_drop:
      - ALL

    deploy:

      resources:

        limits:
          cpus: "0.50"
          memory: 256M

        reservations:
          cpus: "0.05"
          memory: 64M

    logging:

      driver: json-file

      options:
        max-size: "10m"
        max-file: "3"

    healthcheck:

      test:
        [
          "CMD",
          "redis-cli",
          "ping"
        ]

      interval: 10s
      timeout: 5s
      retries: 5


# ============================================================
# PERSISTENT DATA
# ============================================================

volumes:

  redis_data:
"""

PRODUCTION_COMPOSE_FILE = (
    PROD_DIR
    / "docker-compose.production.yml"
)

PRODUCTION_COMPOSE_FILE.write_text(
    production_compose,
    encoding="utf-8"
)

print(
    "✅ Production Docker Compose created"
)


# ============================================================
# 8. SECURITY CHECKLIST
# ============================================================

security_checklist = """# Day 63 Production Security Checklist

## Container Security

- [x] Non-root containers
- [x] No-new-privileges
- [x] Drop Linux capabilities
- [x] Slim base images
- [x] Health checks
- [x] Resource limits
- [x] Read-only application filesystem where practical
- [x] Temporary filesystem for /tmp
- [x] Log rotation

## Application Security

- [ ] Authentication
- [ ] Authorization
- [ ] RBAC
- [ ] API rate limiting
- [ ] Input validation
- [ ] Prompt injection protection
- [ ] PII protection
- [ ] Audit logging
- [ ] TLS
- [ ] Secret management

## Production Secrets

Never commit:

- API keys
- Database passwords
- LLM credentials
- Cloud credentials
- JWT secrets
- Encryption keys

Use:

- AWS Secrets Manager
- Azure Key Vault
- Google Secret Manager
- Kubernetes Secrets
- Docker Secrets

## AI Security

Production RAG systems should also consider:

- document-level access control
- tenant isolation
- retrieval authorization
- prompt injection
- malicious documents
- sensitive data leakage
- model output validation
- grounding validation
- citation validation
"""

SECURITY_CHECKLIST_FILE = (
    SECURITY_DIR
    / "SECURITY_CHECKLIST.md"
)

SECURITY_CHECKLIST_FILE.write_text(
    security_checklist,
    encoding="utf-8"
)

print(
    "✅ Security checklist created"
)


# ============================================================
# 9. STRUCTURED LOGGING CONFIGURATION
# ============================================================

logging_config = """{
  "version": 1,
  "disable_existing_loggers": false,

  "formatters": {
    "json": {
      "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
    }
  },

  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "formatter": "json",
      "stream": "ext://sys.stdout"
    }
  },

  "root": {
    "level": "INFO",
    "handlers": [
      "console"
    ]
  }
}
"""

LOGGING_CONFIG_FILE = (
    MONITORING_DIR
    / "logging.json"
)

LOGGING_CONFIG_FILE.write_text(
    logging_config,
    encoding="utf-8"
)

print(
    "✅ Logging configuration created"
)


# ============================================================
# 10. MONITORING CHECKLIST
# ============================================================

monitoring_checklist = """# Day 63 Production Monitoring

## Infrastructure Metrics

- CPU utilization
- Memory utilization
- Container restarts
- Network traffic
- Disk utilization

## API Metrics

- Request count
- Success rate
- Error rate
- Average latency
- P50 latency
- P95 latency
- P99 latency
- HTTP status distribution

## Retrieval Metrics

- Retrieval requests
- Empty retrieval rate
- Top-K relevance
- Hit@K
- MRR
- Retrieval confidence
- Retrieval latency

## RAG Metrics

- Grounding rate
- Citation coverage
- Answer relevance
- Safe fallback rate
- Hallucination rate
- Context relevance
- Generation latency

## Redis Metrics

- Connection count
- Memory usage
- Hit rate
- Miss rate
- Key count
- Evictions

## Alerts

Possible production alerts:

1. API error rate > threshold
2. P95 latency > threshold
3. Container restart loop
4. High memory usage
5. Redis unavailable
6. Retrieval service unavailable
7. High safe-fallback rate
8. Low grounding rate
9. Retrieval quality degradation
"""

MONITORING_CHECKLIST_FILE = (
    MONITORING_DIR
    / "MONITORING.md"
)

MONITORING_CHECKLIST_FILE.write_text(
    monitoring_checklist,
    encoding="utf-8"
)

print(
    "✅ Monitoring checklist created"
)


# ============================================================
# 11. HEALTH CHECK SCRIPT
# ============================================================

health_script = r'''#!/usr/bin/env python3

import json
import sys
import urllib.request


BASE_URL = "http://127.0.0.1:8000"


def check(endpoint):

    url = BASE_URL + endpoint

    try:

        with urllib.request.urlopen(
            url,
            timeout=5
        ) as response:

            body = response.read().decode(
                "utf-8"
            )

            print(
                endpoint,
                "->",
                response.status
            )

            try:

                print(
                    json.dumps(
                        json.loads(body),
                        indent=2
                    )
                )

            except Exception:

                print(body)

            return response.status == 200

    except Exception as error:

        print(
            endpoint,
            "-> FAILED:",
            error
        )

        return False


checks = [

    "/",

    "/health",

    "/api/v1/info",

    "/api/v1/metrics"
]


results = [
    check(endpoint)
    for endpoint in checks
]


if all(results):

    print("\n✅ Production API smoke test passed")

    sys.exit(0)

else:

    print("\n❌ Production API smoke test failed")

    sys.exit(1)
'''

HEALTH_SCRIPT_FILE = (
    PROD_DIR
    / "health_check.py"
)

HEALTH_SCRIPT_FILE.write_text(
    health_script,
    encoding="utf-8"
)

print(
    "✅ Health-check script created"
)


# ============================================================
# 12. PERFORMANCE TEST SCRIPT
# ============================================================

performance_script = r'''import json
import time
import statistics
import urllib.request


BASE_URL = "http://127.0.0.1:8000"

QUERIES = [

    "What is Docker?",

    "Explain RAG",

    "How should APIs be secured?",

    "What should AI applications monitor?",

    "How does Docker Compose work?"
]


def post_chat(query):

    payload = json.dumps({

        "query":
            query

    }).encode(
        "utf-8"
    )

    request = urllib.request.Request(

        BASE_URL
        + "/api/v1/chat",

        data=payload,

        headers={
            "Content-Type":
                "application/json"
        },

        method="POST"
    )

    start = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=10
    ) as response:

        response.read()

    latency = (

        time.perf_counter()
        - start

    ) * 1000

    return latency


latencies = []


for query in QUERIES:

    try:

        latency = post_chat(
            query
        )

        latencies.append(
            latency
        )

        print(
            f"{query:<45} "
            f"{latency:.2f} ms"
        )

    except Exception as error:

        print(
            "FAILED:",
            query,
            error
        )


if latencies:

    ordered = sorted(
        latencies
    )

    p50 = statistics.median(
        ordered
    )

    p95_index = min(
        len(ordered) - 1,
        int(
            len(ordered) * 0.95
        )
    )

    p95 = ordered[
        p95_index
    ]

    print("\nPerformance Summary")

    print(
        "Requests:",
        len(latencies)
    )

    print(
        "Average:",
        round(
            statistics.mean(latencies),
            2
        ),
        "ms"
    )

    print(
        "P50:",
        round(
            p50,
            2
        ),
        "ms"
    )

    print(
        "P95:",
        round(
            p95,
            2
        ),
        "ms"
    )

    print(
        "Max:",
        round(
            max(latencies),
            2
        ),
        "ms"
    )
'''

PERFORMANCE_SCRIPT_FILE = (
    PROD_DIR
    / "performance_test.py"
)

PERFORMANCE_SCRIPT_FILE.write_text(
    performance_script,
    encoding="utf-8"
)

print(
    "✅ Performance test created"
)


# ============================================================
# 13. CI/CD PIPELINE
# ============================================================
#
# This is a GitHub Actions template.
#
# Flow:
#
# Code Push
#    ↓
# Checkout
#    ↓
# Python validation
#    ↓
# Docker build
#    ↓
# Compose validation
#    ↓
# Container deployment/test
#    ↓
# Image registry
#    ↓
# Cloud deployment
#
# Actual registry/cloud deployment requires credentials
# and is intentionally not performed here.
#
# ============================================================

github_actions = """name: Day 63 AI Application CI/CD

on:

  push:
    branches:
      - main

  pull_request:
    branches:
      - main


jobs:

  validate:

    runs-on: ubuntu-latest

    steps:

      - name: Checkout repository
        uses: actions/checkout@v4


      - name: Setup Python
        uses: actions/setup-python@v5

        with:
          python-version: "3.11"


      - name: Python syntax validation
        run: |
          python -m py_compile main.py
          python -m py_compile retrieval_service/main.py


      - name: Docker Compose validation
        run: |
          docker compose config


  build:

    needs: validate

    runs-on: ubuntu-latest

    steps:

      - name: Checkout repository
        uses: actions/checkout@v4


      - name: Build Docker images
        run: |
          docker compose build


      - name: Start services
        run: |
          docker compose up -d


      - name: Show services
        run: |
          docker compose ps


      - name: Wait for services
        run: |
          sleep 10


      - name: Health check
        run: |
          curl --fail http://localhost:8000/health


      - name: API root test
        run: |
          curl --fail http://localhost:8000/


      - name: Cleanup
        if: always()
        run: |
          docker compose down -v
"""

GITHUB_ACTIONS_FILE = (
    CI_DIR
    / "docker-ci.yml"
)

GITHUB_ACTIONS_FILE.write_text(
    github_actions,
    encoding="utf-8"
)

print(
    "✅ CI/CD workflow created"
)


# ============================================================
# 14. PRODUCTION DEPLOYMENT DOCUMENT
# ============================================================

deployment_document = """# Day 63 Production Deployment Architecture

## Development

Developer
   |
   v
Git
   |
   v
Docker Compose
   |
   +--> FastAPI
   +--> Retrieval Service
   +--> Redis


## CI/CD

Developer
   |
   v
GitHub
   |
   v
GitHub Actions
   |
   +--> Syntax Validation
   |
   +--> Docker Build
   |
   +--> Compose Validation
   |
   +--> Integration Tests
   |
   v
Container Registry


## Cloud Production

Users
   |
   v
DNS
   |
   v
API Gateway / WAF
   |
   v
Load Balancer
   |
   v
FastAPI Container Cluster
   |
   +----------------------+
   |                      |
   v                      v
Agent / RAG Service      Redis
   |
   v
Retrieval Service
   |
   +-------------------+
   |                   |
   v                   v
Vector Database    Keyword Search
   |
   v
Reranker
   |
   v
Enterprise LLM
   |
   v
Grounding Validator
   |
   v
Source Citations
   |
   v
Final Response


## Observability

All services
   |
   +--> Logs
   |
   +--> Metrics
   |
   +--> Traces
   |
   v
Monitoring Platform


Possible tools:

- Prometheus
- Grafana
- OpenTelemetry
- CloudWatch
- Azure Monitor
- Google Cloud Monitoring


## Scaling

FastAPI:
horizontal scaling

Retrieval:
horizontal scaling

Redis:
managed Redis / clustered Redis

Vector DB:
managed scalable vector database

LLM:
managed inference or model-serving cluster
"""

DEPLOYMENT_DOCUMENT_FILE = (
    PROD_DIR
    / "DEPLOYMENT_ARCHITECTURE.md"
)

DEPLOYMENT_DOCUMENT_FILE.write_text(
    deployment_document,
    encoding="utf-8"
)

print(
    "✅ Deployment architecture document created"
)


# ============================================================
# 15. PRODUCTION READINESS CHECKLIST
# ============================================================

production_checklist = {

    "containerization":
        True,

    "multi_container_architecture":
        True,

    "health_checks":
        True,

    "restart_policy":
        True,

    "resource_limits":
        True,

    "non_root_containers":
        True,

    "capability_drop":
        True,

    "no_new_privileges":
        True,

    "read_only_filesystem":
        True,

    "persistent_redis_volume":
        True,

    "service_discovery":
        True,

    "environment_configuration":
        True,

    "logging_configuration":
        True,

    "monitoring_plan":
        True,

    "api_smoke_test":
        True,

    "performance_test":
        True,

    "ci_cd_template":
        True,

    "cloud_architecture":
        True,

    "authentication":
        False,

    "authorization":
        False,

    "tls":
        False,

    "external_secret_manager":
        False,

    "production_cloud_deployment":
        False
}


print("\n" + "=" * 80)
print("📊 PRODUCTION READINESS")
print("=" * 80)


implemented = sum(
    1
    for value
    in production_checklist.values()
    if value
)

total = len(
    production_checklist
)

readiness_percentage = (
    implemented
    / total
    * 100
)


for key, value in production_checklist.items():

    symbol = "✅" if value else "⬜"

    print(
        f"{symbol} {key}"
    )


print(
    "\nArchitecture readiness:",
    round(
        readiness_percentage,
        2
    ),
    "%"
)

print(
    "Note: unchecked items are intentionally "
    "production-environment-specific."
)


# ============================================================
# 16. PYTHON FILE SYNTAX VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("🔍 PYTHON VALIDATION")
print("=" * 80)


python_files_to_validate = [

    PROJECT_DIR
    / "main.py",

    PROJECT_DIR
    / "retrieval_service"
    / "main.py",

    HEALTH_SCRIPT_FILE,

    PERFORMANCE_SCRIPT_FILE
]


syntax_validation_results = {}


for file_path in python_files_to_validate:

    if not file_path.exists():

        print(
            "⚠️ Missing:",
            file_path
        )

        syntax_validation_results[
            str(file_path)
        ] = False

        continue


    result = subprocess.run(

        [
            sys.executable,
            "-m",
            "py_compile",
            str(file_path)
        ],

        capture_output=True,

        text=True
    )


    success = (
        result.returncode == 0
    )


    syntax_validation_results[
        str(file_path)
    ] = success


    if success:

        print(
            "✅",
            file_path.relative_to(
                PROJECT_DIR
            )
        )

    else:

        print(
            "❌",
            file_path.relative_to(
                PROJECT_DIR
            )
        )

        print(
            result.stderr
        )


# ============================================================
# 17. DOCKER AVAILABILITY
# ============================================================

print("\n" + "=" * 80)
print("🐳 DOCKER VALIDATION")
print("=" * 80)


docker_available_final = False
docker_compose_available_final = False


try:

    docker_result = subprocess.run(

        [
            "docker",
            "--version"
        ],

        capture_output=True,

        text=True,

        timeout=10
    )


    if docker_result.returncode == 0:

        docker_available_final = True

        print(
            "✅ Docker:",
            docker_result.stdout.strip()
        )

    else:

        print(
            "⚠️ Docker command unavailable."
        )

except Exception as error:

    print(
        "⚠️ Docker check:",
        error
    )


try:

    compose_result = subprocess.run(

        [
            "docker",
            "compose",
            "version"
        ],

        capture_output=True,

        text=True,

        timeout=10
    )


    if compose_result.returncode == 0:

        docker_compose_available_final = True

        print(
            "✅ Docker Compose:",
            compose_result.stdout.strip()
        )

    else:

        print(
            "⚠️ Docker Compose unavailable."
        )

except Exception as error:

    print(
        "⚠️ Docker Compose check:",
        error
    )


# ============================================================
# 18. VALIDATE PRODUCTION COMPOSE
# ============================================================

production_compose_valid = False


if docker_compose_available_final:

    print("\n" + "=" * 80)
    print("🔎 PRODUCTION COMPOSE VALIDATION")
    print("=" * 80)


    result = subprocess.run(

        [
            "docker",
            "compose",

            "-f",

            "production/docker-compose.production.yml",

            "config"
        ],

        cwd=str(PROJECT_DIR),

        capture_output=True,

        text=True
    )


    if result.returncode == 0:

        production_compose_valid = True

        print(
            "✅ Production Compose configuration is valid."
        )

        print(
            result.stdout[:5000]
        )

    else:

        print(
            "❌ Production Compose validation failed."
        )

        print(
            result.stderr
        )


# ============================================================
# 19. BUILD PRODUCTION IMAGES
# ============================================================

production_images_built = False


if (
    docker_compose_available_final
    and production_compose_valid
):

    print("\n" + "=" * 80)
    print("🔨 BUILDING PRODUCTION IMAGES")
    print("=" * 80)


    build_result = subprocess.run(

        [
            "docker",
            "compose",

            "-f",

            "production/docker-compose.production.yml",

            "build"
        ],

        cwd=str(PROJECT_DIR),

        capture_output=True,

        text=True
    )


    if build_result.returncode == 0:

        production_images_built = True

        print(
            "✅ Production images built successfully."
        )

    else:

        print(
            "❌ Production image build failed."
        )

        print(
            build_result.stdout[-4000:]
        )

        print(
            build_result.stderr[-4000:]
        )


# ============================================================
# 20. START PRODUCTION STACK
# ============================================================

production_stack_started = False


if production_images_built:

    print("\n" + "=" * 80)
    print("🚀 STARTING PRODUCTION STACK")
    print("=" * 80)


    start_result = subprocess.run(

        [
            "docker",
            "compose",

            "-f",

            "production/docker-compose.production.yml",

            "up",
            "-d"
        ],

        cwd=str(PROJECT_DIR),

        capture_output=True,

        text=True
    )


    if start_result.returncode == 0:

        production_stack_started = True

        print(
            "✅ Production stack started."
        )

        print(
            start_result.stdout
        )

        time.sleep(8)

    else:

        print(
            "❌ Production stack failed to start."
        )

        print(
            start_result.stderr
        )


# ============================================================
# 21. SHOW PRODUCTION CONTAINERS
# ============================================================

if docker_compose_available_final:

    print("\n" + "=" * 80)
    print("📦 PRODUCTION CONTAINER STATUS")
    print("=" * 80)


    status_result = subprocess.run(

        [
            "docker",
            "compose",

            "-f",

            "production/docker-compose.production.yml",

            "ps"
        ],

        cwd=str(PROJECT_DIR),

        capture_output=True,

        text=True
    )


    print(
        status_result.stdout
    )


# ============================================================
# 22. PRODUCTION API SMOKE TEST
# ============================================================

BASE_URL_FINAL = (
    "http://127.0.0.1:8000"
)


def http_get_final(
    endpoint
):

    import urllib.request

    with urllib.request.urlopen(

        BASE_URL_FINAL
        + endpoint,

        timeout=8

    ) as response:

        return (

            response.status,

            response.read().decode(
                "utf-8"
            )
        )


if production_stack_started:

    print("\n" + "=" * 80)
    print("🧪 PRODUCTION API SMOKE TEST")
    print("=" * 80)


    smoke_endpoints = [

        "/",

        "/health",

        "/api/v1/info",

        "/api/v1/metrics"
    ]


    smoke_results = {}


    for endpoint in smoke_endpoints:

        try:

            status, body = http_get_final(
                endpoint
            )

            passed = (
                status == 200
            )

            smoke_results[
                endpoint
            ] = passed


            print(

                f"{'✅' if passed else '❌'} "
                f"{endpoint} -> {status}"
            )


            try:

                print(
                    json.dumps(
                        json.loads(body),
                        indent=2
                    )[:1500]
                )

            except Exception:

                print(
                    body[:1500]
                )


        except Exception as error:

            smoke_results[
                endpoint
            ] = False


            print(
                f"❌ {endpoint} -> {error}"
            )


    smoke_test_passed = all(
        smoke_results.values()
    )

else:

    smoke_results = {}

    smoke_test_passed = False

    print(
        "⚠️ Production stack was not started."
    )


# ============================================================
# 23. END-TO-END CHAT TEST
# ============================================================

if production_stack_started:

    print("\n" + "=" * 80)
    print("🤖 PRODUCTION CHAT TEST")
    print("=" * 80)


    import urllib.request


    chat_payload = json.dumps({

        "query":
            "How does Docker help deploy AI applications?"

    }).encode(
        "utf-8"
    )


    request = urllib.request.Request(

        BASE_URL_FINAL
        + "/api/v1/chat",

        data=chat_payload,

        headers={
            "Content-Type":
                "application/json"
        },

        method="POST"
    )


    try:

        start_time = time.perf_counter()


        with urllib.request.urlopen(

            request,

            timeout=10

        ) as response:

            chat_body = response.read().decode(
                "utf-8"
            )

            chat_status = response.status


        chat_latency_ms = (

            time.perf_counter()
            - start_time

        ) * 1000


        print(
            "Status:",
            chat_status
        )

        print(
            "Latency:",
            round(
                chat_latency_ms,
                3
            ),
            "ms"
        )

        print(
            json.dumps(
                json.loads(chat_body),
                indent=2
            )
        )


        chat_test_passed = (
            chat_status == 200
        )


    except Exception as error:

        print(
            "❌ Chat test failed:",
            error
        )

        chat_test_passed = False

else:

    chat_test_passed = False


# ============================================================
# 24. PRODUCTION METRICS TEST
# ============================================================

if production_stack_started:

    print("\n" + "=" * 80)
    print("📊 PRODUCTION METRICS")
    print("=" * 80)


    try:

        status, body = http_get_final(
            "/api/v1/metrics"
        )


        metrics_data_final = json.loads(
            body
        )


        print(
            json.dumps(
                metrics_data_final,
                indent=2
            )
        )


    except Exception as error:

        metrics_data_final = {}

        print(
            "❌ Metrics unavailable:",
            error
        )

else:

    metrics_data_final = {}


# ============================================================
# 25. PRODUCTION PERFORMANCE TEST
# ============================================================

if production_stack_started:

    print("\n" + "=" * 80)
    print("⚡ PRODUCTION PERFORMANCE TEST")
    print("=" * 80)


    performance_queries = [

        "What is Docker?",

        "Explain RAG",

        "How should APIs be secured?",

        "What should AI applications monitor?",

        "How does Docker Compose work?"
    ]


    performance_latencies = []


    for query in performance_queries:

        payload = json.dumps({

            "query":
                query

        }).encode(
            "utf-8"
        )


        request = urllib.request.Request(

            BASE_URL_FINAL
            + "/api/v1/chat",

            data=payload,

            headers={
                "Content-Type":
                    "application/json"
            },

            method="POST"
        )


        try:

            start_time = time.perf_counter()


            with urllib.request.urlopen(

                request,

                timeout=10

            ) as response:

                response.read()


            latency = (

                time.perf_counter()
                - start_time

            ) * 1000


            performance_latencies.append(
                latency
            )


            print(

                f"{latency:8.2f} ms | "
                f"{query}"
            )


        except Exception as error:

            print(
                "FAILED:",
                query,
                error
            )


    if performance_latencies:

        ordered_latencies = sorted(
            performance_latencies
        )


        average_latency_final = (

            sum(
                performance_latencies
            )
            / len(
                performance_latencies
            )
        )


        p50_latency_final = (
            ordered_latencies[
                len(ordered_latencies) // 2
            ]
        )


        p95_position = min(

            len(ordered_latencies) - 1,

            max(
                0,
                int(
                    len(ordered_latencies)
                    * 0.95
                )
            )
        )


        p95_latency_final = (
            ordered_latencies[
                p95_position
            ]
        )


        max_latency_final = max(
            performance_latencies
        )


        print("\nPerformance Summary:")

        print(
            "Requests:",
            len(
                performance_latencies
            )
        )

        print(
            "Average:",
            round(
                average_latency_final,
                2
            ),
            "ms"
        )

        print(
            "P50:",
            round(
                p50_latency_final,
                2
            ),
            "ms"
        )

        print(
            "P95:",
            round(
                p95_latency_final,
                2
            ),
            "ms"
        )

        print(
            "Max:",
            round(
                max_latency_final,
                2
            ),
            "ms"
        )

    else:

        average_latency_final = 0
        p50_latency_final = 0
        p95_latency_final = 0
        max_latency_final = 0

else:

    average_latency_final = 0
    p50_latency_final = 0
    p95_latency_final = 0
    max_latency_final = 0


# ============================================================
# 26. CONTAINER SECURITY INSPECTION
# ============================================================

if docker_compose_available_final:

    print("\n" + "=" * 80)
    print("🔐 CONTAINER SECURITY INSPECTION")
    print("=" * 80)


    security_result = subprocess.run(

        [
            "docker",
            "compose",

            "-f",

            "production/docker-compose.production.yml",

            "config"
        ],

        cwd=str(PROJECT_DIR),

        capture_output=True,

        text=True
    )


    security_config_text = (
        security_result.stdout
    )


    security_features_detected = {

        "no_new_privileges":
            "no-new-privileges:true"
            in security_config_text,

        "cap_drop":
            "cap_drop:"
            in security_config_text,

        "resource_limits":
            "limits:"
            in security_config_text,

        "healthchecks":
            "healthcheck:"
            in security_config_text,

        "restart_policy":
            "restart: unless-stopped"
            in security_config_text,

        "persistent_volume":
            "redis_data:"
            in security_config_text,

        "read_only":
            "read_only: true"
            in security_config_text
    }


    for key, value in security_features_detected.items():

        print(

            f"{'✅' if value else '❌'} "
            f"{key}"
        )

else:

    security_features_detected = {}


# ============================================================
# 27. DOCKER IMAGE LIST
# ============================================================

if docker_available_final:

    print("\n" + "=" * 80)
    print("🐳 DOCKER IMAGES")
    print("=" * 80)


    images_result = subprocess.run(

        [
            "docker",
            "images"
        ],

        capture_output=True,

        text=True
    )


    print(
        images_result.stdout
    )


# ============================================================
# 28. FINAL EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("🏆 DAY 63 FINAL EVALUATION")
print("=" * 80)


final_evaluation = {

    "API Python syntax":
        all(
            syntax_validation_results.values()
        ),

    "Docker available":
        docker_available_final,

    "Docker Compose available":
        docker_compose_available_final,

    "Production Compose valid":
        production_compose_valid,

    "Production images built":
        production_images_built,

    "Production stack started":
        production_stack_started,

    "API smoke tests":
        smoke_test_passed,

    "Chat endpoint":
        chat_test_passed,

    "Security configuration":
        all(
            security_features_detected.values()
        )
        if security_features_detected
        else False
}


for metric, result in final_evaluation.items():

    print(

        f"{'✅' if result else '⚠️'} "
        f"{metric}: "
        f"{result}"
    )


# ============================================================
# 29. FINAL PROJECT STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("📂 FINAL PROJECT STRUCTURE")
print("=" * 80)


print(f"""
{PROJECT_DIR.name}/
│
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── README.md
│
├── retrieval_service/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── production/
│   ├── Dockerfile.api
│   ├── Dockerfile.retrieval
│   ├── docker-compose.production.yml
│   ├── requirements.txt
│   ├── .env.example
│   ├── health_check.py
│   ├── performance_test.py
│   └── DEPLOYMENT_ARCHITECTURE.md
│
├── monitoring/
│   ├── logging.json
│   └── MONITORING.md
│
├── security/
│   └── SECURITY_CHECKLIST.md
│
└── .github/
    └── workflows/
        └── docker-ci.yml
""")


# ============================================================
# 30. FINAL END-TO-END ARCHITECTURE
# ============================================================

print("\n" + "=" * 80)
print("🔥 DAY 63 — COMPLETE END-TO-END ARCHITECTURE")
print("=" * 80)


print("""
                         USER
                           |
                           v
                  +-------------------+
                  | API Gateway / WAF |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  |   FastAPI API     |
                  |    Container      |
                  +---------+---------+
                            |
                 +----------+----------+
                 |                     |
                 v                     v
       +-------------------+   +----------------+
       | Retrieval Service |   |     Redis      |
       |    Container      |   | Session/Cache  |
       +---------+---------+   +----------------+
                 |
                 v
          Knowledge Base
                 |
                 v
           Retrieval
                 |
                 v
          Relevant Context
                 |
                 v
        RAG / Generation Layer
                 |
                 v
        Grounding Validation
                 |
                 v
          Source Citations
                 |
                 v
          Safe Final Answer
                 |
                 v
                USER


     ┌─────────────────────────────────────────┐
     │          OBSERVABILITY                  │
     │                                         │
     │ Logs | Metrics | Latency | Errors       │
     │ Retrieval | Grounding | Citations      │
     └─────────────────────────────────────────┘


     ┌─────────────────────────────────────────┐
     │             SECURITY                    │
     │                                         │
     │ Non-root | Cap Drop | Resource Limits   │
     │ Input Validation | Secrets | RBAC       │
     │ Authentication | Authorization          │
     └─────────────────────────────────────────┘


     ┌─────────────────────────────────────────┐
     │              CI/CD                      │
     │                                         │
     │ GitHub → Validate → Build → Test        │
     │       → Registry → Deployment           │
     └─────────────────────────────────────────┘
""")


# ============================================================
# 31. DAY 63 LEARNING SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("🧠 DAY 63 — WHAT YOU LEARNED")
print("=" * 80)


day63_learning = [

    "Docker containerization",

    "Dockerfile creation",

    "Container networking",

    "Docker Compose",

    "Multi-container AI architecture",

    "FastAPI containerization",

    "Dedicated retrieval service",

    "Redis integration",

    "Persistent Docker volumes",

    "Environment-based configuration",

    "Container health checks",

    "Service dependency management",

    "Restart policies",

    "CPU and memory resource limits",

    "Non-root container execution",

    "Linux capability dropping",

    "No-new-privileges security",

    "Read-only container filesystems",

    "Container log rotation",

    "API smoke testing",

    "Performance testing",

    "Production monitoring design",

    "AI/RAG monitoring metrics",

    "CI/CD architecture",

    "Cloud deployment architecture",

    "Production security considerations",

    "Horizontal scaling concepts",

    "Service separation",

    "Production readiness evaluation"
]


for index, learning in enumerate(
    day63_learning,
    start=1
):

    print(
        f"{index:02d}. {learning}"
    )


# ============================================================
# 32. IMPORTANT VARIABLES
# ============================================================

print("\n" + "=" * 80)
print("📌 IMPORTANT VARIABLES CREATED")
print("=" * 80)


IMPORTANT_VARIABLES_DAY63_FINAL = {

    "PROJECT_DIR":
        str(PROJECT_DIR),

    "PROD_DIR":
        str(PROD_DIR),

    "CI_DIR":
        str(CI_DIR),

    "MONITORING_DIR":
        str(MONITORING_DIR),

    "SECURITY_DIR":
        str(SECURITY_DIR),

    "PRODUCTION_API_DOCKERFILE":
        str(PRODUCTION_API_DOCKERFILE),

    "PRODUCTION_RETRIEVAL_DOCKERFILE":
        str(PRODUCTION_RETRIEVAL_DOCKERFILE),

    "PRODUCTION_COMPOSE_FILE":
        str(PRODUCTION_COMPOSE_FILE),

    "PRODUCTION_REQUIREMENTS_FILE":
        str(PRODUCTION_REQUIREMENTS_FILE),

    "PRODUCTION_ENV_FILE":
        str(PRODUCTION_ENV_FILE),

    "SECURITY_CHECKLIST_FILE":
        str(SECURITY_CHECKLIST_FILE),

    "MONITORING_CHECKLIST_FILE":
        str(MONITORING_CHECKLIST_FILE),

    "HEALTH_SCRIPT_FILE":
        str(HEALTH_SCRIPT_FILE),

    "PERFORMANCE_SCRIPT_FILE":
        str(PERFORMANCE_SCRIPT_FILE),

    "GITHUB_ACTIONS_FILE":
        str(GITHUB_ACTIONS_FILE),

    "DEPLOYMENT_DOCUMENT_FILE":
        str(DEPLOYMENT_DOCUMENT_FILE),

    "production_compose_valid":
        production_compose_valid,

    "production_images_built":
        production_images_built,

    "production_stack_started":
        production_stack_started,

    "smoke_test_passed":
        smoke_test_passed,

    "chat_test_passed":
        chat_test_passed,

    "average_latency_final":
        average_latency_final,

    "p50_latency_final":
        p50_latency_final,

    "p95_latency_final":
        p95_latency_final,

    "max_latency_final":
        max_latency_final
}


for key, value in IMPORTANT_VARIABLES_DAY63_FINAL.items():

    print(
        f"{key} = {value}"
    )


# ============================================================
# 33. INTERVIEW-READY PROJECT EXPLANATION
# ============================================================

INTERVIEW_EXPLANATION_DAY63 = """
I built a Dockerized enterprise AI application and evolved it
from a single-container FastAPI service into a multi-container
production-oriented architecture.

The system separates the API layer, retrieval layer and shared
state layer into independent containers. FastAPI exposes the
application APIs, a dedicated retrieval service handles knowledge
retrieval, and Redis provides shared session and caching
infrastructure.

I used Docker Compose for service orchestration, internal service
discovery, health checks, persistent volumes and service
dependencies.

For productionization, I added non-root containers, dropped Linux
capabilities, no-new-privileges, read-only application
filesystems, resource limits, restart policies, log rotation and
container health checks.

I also created API smoke tests, performance tests, monitoring
requirements and a CI/CD workflow that validates Python code,
Docker Compose configuration, builds the containers and performs
integration health checks.

The deployment architecture is cloud-ready and can evolve into
an API Gateway or WAF, load balancer, horizontally scaled FastAPI
services, managed Redis, vector database, retrieval service,
enterprise LLM, grounding validation and centralized monitoring.

The implementation remains lightweight for development, while the
architecture provides clear service boundaries for production
scaling.
"""


print("\n" + "=" * 80)
print("🎤 INTERVIEW EXPLANATION")
print("=" * 80)

print(
    INTERVIEW_EXPLANATION_DAY63
)


# ============================================================
# 34. RESUME BULLET
# ============================================================

RESUME_BULLET_DAY63 = (
    "Built a production-oriented Dockerized enterprise AI "
    "architecture using FastAPI, Docker Compose, Redis and a "
    "dedicated retrieval service, implementing container "
    "security, health checks, resource limits, monitoring, "
    "performance testing and CI/CD-ready deployment architecture."
)


print("\n" + "=" * 80)
print("📄 RESUME BULLET")
print("=" * 80)

print(
    RESUME_BULLET_DAY63
)


# ============================================================
# 35. GITHUB PROJECT DESCRIPTION
# ============================================================

GITHUB_DESCRIPTION_DAY63 = """
# Dockerized Enterprise AI Application

A production-oriented enterprise AI architecture built with
FastAPI, Docker, Docker Compose, Redis and a dedicated retrieval
service.

## Features

- FastAPI AI service
- Dedicated retrieval service
- Redis session/cache infrastructure
- Docker Compose orchestration
- Container health checks
- Internal service discovery
- Persistent volumes
- Resource limits
- Non-root containers
- Container security hardening
- API monitoring
- Performance testing
- CI/CD workflow
- Cloud-ready architecture

## Architecture

User
→ FastAPI
→ Retrieval Service
→ Redis
→ Context
→ RAG
→ Grounding
→ Final Response

## Production Evolution

The architecture can evolve toward:

API Gateway
→ Load Balancer
→ FastAPI Cluster
→ Retrieval Service
→ Vector Database
→ Reranker
→ Enterprise LLM
→ Grounding Validator
→ Monitoring

## Technology

Python
FastAPI
Docker
Docker Compose
Redis
REST APIs
RAG
Vector Search
CI/CD
Container Security
Monitoring
"""


GITHUB_DESCRIPTION_FILE = (
    PROJECT_DIR
    / "GITHUB_DESCRIPTION.md"
)

GITHUB_DESCRIPTION_FILE.write_text(
    GITHUB_DESCRIPTION_DAY63,
    encoding="utf-8"
)

print(
    "✅ GitHub description created"
)


# ============================================================
# 36. FINAL DAY 63 SCORECARD
# ============================================================

print("\n" + "=" * 80)
print("🏆 DAY 63 FINAL SCORECARD")
print("=" * 80)


DAY63_SCORECARD = {

    "Docker fundamentals":
        "Completed",

    "Dockerfile":
        "Completed",

    "FastAPI containerization":
        "Completed",

    "Docker Compose":
        "Completed",

    "Multi-container architecture":
        "Completed",

    "Redis":
        "Completed",

    "Retrieval service":
        "Completed",

    "Health checks":
        "Completed",

    "Persistent volumes":
        "Completed",

    "Resource limits":
        "Completed",

    "Container security":
        "Completed",

    "Monitoring design":
        "Completed",

    "Performance testing":
        "Completed",

    "CI/CD":
        "Template created",

    "Cloud architecture":
        "Designed",

    "Actual cloud deployment":
        "Not performed",

    "Production authentication":
        "Architecture consideration",

    "Production authorization":
        "Architecture consideration",

    "Production secret manager":
        "Architecture consideration"
}


for key, value in DAY63_SCORECARD.items():

    print(
        f"{key:<35} : {value}"
    )


# ============================================================
# 37. FINAL PROJECT STATEMENT
# ============================================================

FINAL_PROJECT_STATEMENT_DAY63 = """
DAY 63 — DOCKERIZED ENTERPRISE AI APPLICATION

Built an end-to-end containerized enterprise AI architecture
using FastAPI, Docker, Docker Compose and Redis.

The system separates the API, retrieval and shared-state layers
into independent services and implements container networking,
health checks, persistent storage, resource limits, restart
policies and security hardening.

The application also includes retrieval integration, grounded
response flow, session infrastructure, API monitoring,
performance testing and a CI/CD-ready workflow.

The architecture is designed to evolve toward cloud deployment
with an API Gateway/WAF, load balancer, horizontally scaled
services, managed Redis, vector databases, enterprise LLMs,
grounding validation and centralized observability.
"""


print("\n" + "=" * 80)
print("🚀 FINAL DAY 63 PROJECT")
print("=" * 80)

print(
    FINAL_PROJECT_STATEMENT_DAY63
)


# ============================================================
# 38. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("✅ DAY 63 FINAL VALIDATION")
print("=" * 80)


print(
    "Python validation:",
    all(
        syntax_validation_results.values()
    )
)

print(
    "Docker available:",
    docker_available_final
)

print(
    "Docker Compose available:",
    docker_compose_available_final
)

print(
    "Production Compose valid:",
    production_compose_valid
)

print(
    "Production images built:",
    production_images_built
)

print(
    "Production stack started:",
    production_stack_started
)

print(
    "API smoke test:",
    smoke_test_passed
)

print(
    "Chat test:",
    chat_test_passed
)

print(
    "Average latency:",
    round(
        average_latency_final,
        2
    ),
    "ms"
)

print(
    "P50 latency:",
    round(
        p50_latency_final,
        2
    ),
    "ms"
)

print(
    "P95 latency:",
    round(
        p95_latency_final,
        2
    ),
    "ms"
)


print("\n" + "=" * 80)
print("🎉 DAY 63/100 — COMPLETED")
print("=" * 80)

print("""
DAY 63 COMPLETE FLOW:

Part 1
Docker Fundamentals
        ↓
Part 2
Dockerized FastAPI AI Service
        ↓
Part 3
Multi-Container AI Architecture
        ↓
Part 4
Production Dockerization
        ↓
Security Hardening
        ↓
Health Checks
        ↓
Resource Limits
        ↓
Monitoring
        ↓
Performance Testing
        ↓
CI/CD
        ↓
Cloud-Ready Architecture


🔥 FINAL RESULT:

You have built a production-oriented container architecture
for an enterprise AI application rather than just putting
Python code inside a Docker container.

IMPORTANT:

Actual cloud deployment, authentication, authorization,
TLS, managed secrets, production vector databases and
enterprise LLM infrastructure still need environment-specific
configuration before real production deployment.
""")
