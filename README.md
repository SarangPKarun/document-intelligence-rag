# Document Intelligence RAG

A local, agentic Retrieval-Augmented Generation (RAG) system designed for document intelligence. This project leverages **LangGraph**, **Ollama**, **Weaviate**, and **FastAPI** to provide a private, efficient, and conversational interface for querying your documents.

## 🚀 Features

- **Local & Private**: Runs entirely locally using Docker, ensuring data privacy.
- **Multi-Format Ingestion**: Supports uploading and indexing of PDF and Text files.
- **Advanced Retrieval**: Implements **HyDe (Hypothetical Document Embeddings)** to improve retrieval accuracy by generating hypothetical answers before searching.
- **Agentic Workflow**: Uses **LangGraph** to orchestrate retrieval and generation steps.
- **Vector Search**: Powered by **Weaviate** for efficient semantic search.
- **Conversational UI**: Simple web interface for chatting with your documents.

## 🏗️ Architecture

The system consists of three main services orchestrated via Docker Compose:

1.  **App Service (`app`)**:
    *   **Backend**: FastAPI server handling API requests (`/ingest`, `/ask`).
    *   **Agent**: LangGraph state machine managing the RAG flow (HyDe Retrieve -> Generate).
    *   **Frontend**: Static HTML/JS interface served by FastAPI.
2.  **Vector Database (`weaviate`)**:
    *   Stores document chunks and embeddings.
3.  **LLM Service (`ollama`)**:
    *   Hosts the local LLM (`smollm2:1.7b`) for embedding generation and answer synthesis.

## 🛠️ Prerequisites

- **Docker** and **Docker Compose** installed on your machine.

## 📦 Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd document-intelligence-rag
    ```

2.  **Start the services**:
    ```bash
    docker-compose up -d --build app
    ```

3.  **Pull the LLM model**:
    Once the containers are running, you need to pull the specific model into the Ollama container.
    ```bash
    docker exec -it document-intelligence-rag-ollama-1 ollama pull smollm2:1.7b
    ```
    *Note: The container name `document-intelligence-rag-ollama-1` depends on your directory name. Check `docker ps` if this fails.*

## 💻 Usage

### Web Interface
Open your browser and navigate to:
**[http://localhost:8000](http://localhost:8000)**

1.  **Upload**: Use the file picker to upload a PDF or TXT file.
2.  **Chat**: Type your question in the chat box to retrieve answers based on the uploaded content.

### AAPI Endpoints

You can also interact directly with the API (Swagger UI available at `http://localhost:8000/docs`).

-   **Ingest Document**:
    ```bash
    curl -X 'POST' \
      'http://localhost:8000/ingest' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'file=@/path/to/your/document.pdf'
    ```

-   **Ask Question**:
    ```bash
    curl -X 'POST' \
      'http://localhost:8000/ask' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{ "question": "What is the summary of this document?" }'
    ```

-   **Clear Context**:
    ```bash
    curl -X 'POST' 'http://localhost:8000/delete_context'
    ```

## 📊 Evaluation

The `evaluation/` directory contains scripts and datasets for evaluating the RAG pipeline using **RAGAS**.
-   `evaluate_custom.py`: Script to run evaluations.
-   `test_dataset.json`: Ground truth dataset.