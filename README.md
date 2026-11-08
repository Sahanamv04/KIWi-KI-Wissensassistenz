# KIWi — KI-Wissensassistenz

**AI-Powered Knowledge Assistance for the THWS CampusPortal**

KIWi (KI-Wissensassistenz) is a project focused on developing a locally hosted Retrieval-Augmented Generation (RAG) assistant for the THWS CampusPortal (HISinOne). It helps users find answers to questions about university processes and documentation through natural-language queries, with answers grounded in relevant sources.

The project aims to keep data processing within approved university infrastructure and provide transparent, source-supported responses.

## Project Objectives

- **Natural-language question answering:** Allow users to ask questions about university procedures in everyday language.
- **Retrieval-Augmented Generation:** Retrieve relevant information from approved knowledge sources and use it to generate context-aware answers.
- **Source citations:** Provide references to source records and URLs supporting each answer.
- **Local deployment:** Explore an architecture that avoids sending university data to external LLM APIs.
- **Reliable responses:** Reduce unsupported answers and allow the system to abstain when information is insufficient.
- **Evaluation:** Assess retrieval quality, answer correctness, citation accuracy, and response time.

## Proposed Architecture

The planned system follows a Retrieval-Augmented Generation workflow:

1. **Data ingestion** — Load approved university knowledge records from CSV files.
2. **Text preprocessing** — Prepare record questions, answers, and source details.
3. **Chunking and metadata** — Divide content into passages and retain source information.
4. **Embedding generation** — Convert passages into numerical representations for semantic search.
5. **Vector storage** — Store embeddings, text, and metadata in ChromaDB.
6. **Question processing** — Convert a user's question into a searchable representation.
7. **Information retrieval** — Retrieve relevant passages from the knowledge base.
8. **Answer generation** — Use a locally hosted language model to generate an answer grounded in retrieved evidence.
9. **Citation and quality checks** — Provide source references and handle insufficient evidence.
10. **Response delivery** — Display the answer and its supporting sources.

The architecture and technology choices are preliminary and will be refined during development.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Python CSV module |
| Text splitting | LangChain Text Splitters |
| Embedding model | BGE-M3 |
| Vector database | ChromaDB |
| Local LLM runtime | Ollama |
| Language model | Qwen 2.5 7B Instruct |
| Interface | Interactive terminal |
| Evaluation | Custom tests and human review |
| Future deployment | FastAPI and Docker Compose (under consideration) |

## Project Structure

```text
KIWi-KI-Wissensassistenz/
├── data/              # Source CSV data
├── chroma_db/         # Persistent vector database
├── docs/              # Project documentation
├── ingest.py          # Document processing and indexing
├── rag.py             # Retrieval and answer generation
├── config.py          # Model and application settings
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## Prerequisites

Before running KIWi, ensure that you have:

- Python 3 installed.
- Ollama installed and running.
- The required language and embedding models downloaded.
- The project dependencies installed.
- Approved CSV files available for indexing.

The commands below use `python3`, as in the current development environment.

## Installation and Setup

### 1. Open the project directory

Open a terminal and navigate to the project folder:

```bash
cd KIWi-KI-Wissensassistenz
```

If the project is already open in VS Code, you can use its integrated terminal.

### 2. Install Python dependencies

Install the packages listed in `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

### 3. Install and start Ollama

Install Ollama from the official website if it is not already installed:

https://ollama.com/

Make sure the Ollama service is running. Then download the models:

```bash
ollama pull bge-m3
ollama pull qwen2.5:7b-instruct
```

Verify the installed models:

```bash
ollama list
```

The model names must match the settings in `config.py`.

### 4. Configure the application

Check `config.py` and confirm that the model settings match the installed models:

```python
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
```

The default configuration also specifies the Ollama host, CSV data directory,
ChromaDB location, chunking settings, and number of retrieved passages.

### 5. Add CSV data

Place the approved CSV files you want KIWi to search inside the `data/` directory.
The ingestion script reads the question, answer, key fact, evidence, and source
fields from each record. It uses UTF-8 CSV files (including files with a UTF-8 BOM).
PDF files in `data/` are not used by the current ingestion script.

For example:

```text
data/
└── Data.csv
```

Use only data that is permitted for your intended development and deployment environment.

## How to Run KIWi

### Step 1: Build the knowledge base

Whenever you add or update source CSV files, run:

```bash
python3 ingest.py
```

The ingestion script combines each CSV record's question, answer, evidence, and
source details, splits long records into chunks, generates embeddings, and stores
the results in ChromaDB.

A successful run should display output similar to:

```text
995 chunks from 857 CSV records
indexed 32/995
indexed 64/995
...
Done. Stored in ChromaDB: 995 chunks
```

The exact numbers depend on the CSV records and their lengths.

**Important:** The current ingestion script rebuilds the collection when it runs. Make sure the source CSV files and configuration are correct before running it.

### Step 2: Start KIWi

Run the following command:

```bash
python3 rag.py
```

The assistant starts in interactive terminal mode.

Example:

```text
KIWi - THWS Knowledge Assistant
Ask a question or type 'exit' to quit.

You: How can students contact the ITSC Helpdesk in Schweinfurt?

KIWi: [Generated answer based on retrieved CSV records]

Sources:
  [1] THWS ITSC; record ADD-0032; ITSC > Student services > ITSC helpdesk Schweinfurt; verified 2026-11-07; https://itsc.thws.de/fuer-studierende/
```

The answer and source shown above are illustrative; actual output depends on the indexed data and model response.
KIWi is instructed to give a useful explanation grounded in the retrieved records
and include a direct website link in the answer when a source URL is available.

### Step 3: Ask more questions

You can enter multiple questions without restarting the program:

```text
You: What services does the ITSC provide for students?

You: How do I contact the ITSC Helpdesk?

You: What is the student services portal path?

You: exit
```

KIWi retrieves relevant passages for each question and generates an answer using the configured local language model.

Typing `exit` or `quit` ends the session.

### Step 4: Test retrieval separately

To inspect the retrieved document chunks without generating an LLM answer, run:

```bash
python3 rag.py --chunks
```

To test retrieval for a particular question:

```bash
python3 rag.py --chunks "What is the deadline for HAW-ForschungsSchub?"
```

This mode helps investigate whether the correct document passages are being retrieved.

## Configuration

The main configuration options are defined in `config.py`.

| Setting | Purpose |
|---|---|
| `OLLAMA_HOST` | Address of the Ollama service |
| `LLM_MODEL` | Language model used to generate answers |
| `EMBED_MODEL` | Model used to create embeddings |
| `DATA_DIR` | Directory containing source CSV files |
| `CHROMA_DIR` | Directory used for persistent vector storage |
| `COLLECTION` | Name of the ChromaDB collection |
| `CHUNK_SIZE` | Maximum text chunk size used during splitting |
| `CHUNK_OVERLAP` | Overlap between adjacent chunks |
| `TOP_K` | Number of relevant chunks returned to the answer-generation step |

Some settings can be overridden using environment variables.

## Troubleshooting

**Model not found**

Check the installed model names:

```bash
ollama list
```

If a configured model is missing, download it with `ollama pull` or update the corresponding setting in `config.py`.

**Python module not found**

Install the project dependencies:

```bash
python3 -m pip install -r requirements.txt
```

**No collection or database found**

Run CSV ingestion first:

```bash
python3 ingest.py
```

**Poor or irrelevant answers**

Answer quality depends on the CSV content, chunking, embeddings, retrieval, and language model.

**Incorrect or unsupported answers**

Check the cited records and source URLs. Displayed citations do not, by themselves, guarantee that every generated claim is supported.

## Design Principles

- **Privacy first:** Respect university data-handling and deployment requirements.
- **Evidence-based answers:** Ground responses in retrieved documentation.
- **Transparency:** Show users the sources behind generated answers.
- **Modularity:** Keep retrieval, embeddings, generation, and the interface separable.
- **Evaluation-driven development:** Use tests and evidence to guide technical decisions.

## Future Development

Potential future improvements include:

- A web interface for CampusPortal users.
- A backend API using FastAPI.
- More robust citation validation and insufficient-evidence detection.
- Evaluation datasets for retrieval and answer quality.
- Response-time and resource-usage measurements.
- Containerized deployment, subject to university infrastructure requirements.

## Project Status

KIWi is under development. The current prototype supports CSV ingestion, embedding generation, semantic retrieval, local LLM-based answer generation, source references, and interactive terminal questions.

The architecture and technology stack remain preliminary and subject to change as requirements are clarified.
