# ✈️ AeroMind: Cloud-Native Aviation AI Agent

AeroMind is an intelligent, multi-agent Retrieval-Augmented Generation (RAG) platform designed specifically for the aviation industry. It processes complex aviation technical documents and provides highly accurate, context-aware answers to users through a real-time web interface.

🌐 **Live Demo:** [https://aeromind-tau.vercel.app/](https://aeromind-tau.vercel.app/)

## 🏗️ Architecture & Tech Stack

AeroMind is built with a modern, cloud-native architecture prioritizing performance, scalability, and automated deployment.

- **Frontend:** Vanilla HTML/JS/CSS (Deployed globally on **Vercel**).
- **Backend API:** **FastAPI** & Python 3.12 (Containerized with **Docker** & served via **Nginx**).
- **Cloud Infrastructure:** Hosted on **AWS EC2**, utilizing **Amazon S3** for scalable document storage.
- **AI & Orchestration:** 
  - **LangGraph** & **LangChain** for multi-agent workflows.
  - **Groq** for high-speed LLM inference.
  - **ChromaDB** for vector embeddings and semantic search.
- **CI/CD:** Automated testing and integration via **GitHub Actions**.

## 🧠 How It Works

1. **Cloud Ingestion:** A background pipeline securely authenticates with AWS via `boto3`, pulls aviation manuals (PDFs) from an S3 bucket into a temporary hidden memory buffer, and extracts the text using `PyMuPDF`.
2. **Vectorization:** The text is chunked and embedded into a ChromaDB vector database using Sentence Transformers.
3. **Multi-Agent RAG:** When a user asks a question, LangGraph dynamically routes the query through specialized AI agents:
   - **🔍 Retriever Agent:** Analyzes the user's query, transforms it for optimal search, and pulls the most relevant context from the vector database.
   - **📝 Generator Agent:** Synthesizes the retrieved aviation technical data and formulates a highly accurate, professional response.
   - **⚖️ Evaluator Agent:** (Optional/Future) Cross-checks the generated answer against the original source documents to prevent hallucinations.
4. **CORS & Proxying:** To ensure seamless communication across domains, the system is designed with a decoupled architecture. The Vercel frontend securely fetches data from the AWS EC2 instance.

## 🚀 Running Locally

This project uses a `Makefile` to simplify Docker commands for developers.

1. Clone the repository.
2. Create a `.env` file based on `.env.example` and add your AWS and Groq credentials.
3. Build and start the containers:
```bash
make up
```
4. To ingest new documents from S3 into your local vector database:
```bash
python scripts/ingest_docs.py
```
5. To view the local web interface, navigate to `http://localhost`.

## 🧪 Testing
Automated testing is handled via `pytest`. To run the test suite locally:
```bash
make test
```
