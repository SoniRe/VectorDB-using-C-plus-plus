<div align="center">

# Vector DB

**A Vector Database built from scratch in C++**

![C++](https://img.shields.io/badge/C++-17-00599C?style=flat&logo=cplusplus)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows-blue)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-orange)

Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms from scratch with a RAG pipeline powered by a local LLM via Ollama. No cloud. No API keys. Everything runs on your machine.

</div>

---

## What This Project Does

| Feature | Description |
|---|---|
| 3 Search Algorithms | HNSW (production-grade), KD-Tree, Brute Force — compare speed side by side |
| 3 Distance Metrics | Cosine similarity, Euclidean distance, Manhattan distance |
| 16D Demo Vectors | 20 pre-loaded vectors across 4 categories (CS, Math, Food, Sports) |
| 2D PCA Scatter Plot | Live visualization of semantic space — watch clusters form |
| Real Document Embedding | Paste any text → Ollama embeds it with nomic-embed-text (768D) |
| RAG Pipeline | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| Full REST API | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)     ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (C++)              ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search               ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)             ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

---

## Project Structure

```
VecDB/
├── main.cpp          ← C++ backend: REST API, seeds 20 demo vectors, serves index.html
├── httplib.h         ← Single-header HTTP server library
├── structs.h         ← Shared structs: Result, Document, HNSWNode, KDNode
├── math_utils.h      ← Vector math: dot, norm, cosine, euclidean, manhattan, distance
├── brute_force.h     ← BruteForce class: O(N·d) exact search
├── kdtree.h          ← KDTree class: O(log N) axis-aligned partitioning
├── hnsw.h            ← HNSW class: O(log N) multilayer graph search
├── vector_db.h       ← VectorDB: unified interface over all 3 algorithms (16D)
├── ollama.h          ← OllamaClient: HTTP calls to /api/embeddings + /api/generate
├── document_db.h     ← DocumentDB: chunking + 768D embeddings + RAG pipeline
└── index.html        ← Frontend: PCA scatter plot, search UI, Ask AI tab
```

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)
The same algorithm used by **Pinecone, Weaviate, Chroma, and Milvus**.

```
Layer 2 (Highway):    A-----------E              ← few nodes, long jumps
Layer 1 (Main road):  A----C------E----G         ← medium jumps
Layer 0 (Street):     A-B--C--D---E-F--G         ← all nodes, fine search
```

Insert: Greedily descend from top layer finding the nearest node, drop layers, beam search at each layer to find M best neighbors, connect bidirectionally.

Search: Same greedy descent. At layer 0 run full beam search with `ef` candidates. Returns top-K.

**Why it's fast:** Upper layers = highway to the right neighborhood. Layer 0 = fine-grained exact search.

### KD-Tree (K-Dimensional Tree)
Binary space partitioning. Splits space along one dimension per level, cycling through all dimensions. Prunes subtrees when the closest possible point can't beat the current best.

**Weakness:** Degrades at high dimensions (curse of dimensionality). Works well ≤20D, approaches brute force at 768D.

### Why HNSW Wins at High Dimensions
KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions, almost all space is near the hypersphere boundary — no subtrees get pruned. HNSW's graph-based approach doesn't have this problem.

| Algorithm | Complexity | Type |
|---|---|---|
| Brute Force | O(N·d) | Exact |
| KD-Tree | O(log N) | Exact (low dims) |
| HNSW | O(log N) | Approximate |

---

## Prerequisites

- **g++ 13+** with C++17 support
- **Ollama** with `nomic-embed-text` and `llama3.2` pulled
- **Windows** (uses `-lws2_32` for networking)

---

## Setup

### Step 1 — Install g++ (if not already)

Check if installed:
```powershell
g++ --version
```

If not, install via MSYS2 from https://www.msys2.org:
```bash
pacman -S mingw-w64-ucrt-x86_64-gcc
```
Add `C:\msys64\ucrt64\bin` to Windows PATH.

### Step 2 — Install Ollama

Download from https://ollama.com and pull the models:
```powershell
ollama pull nomic-embed-text   # ~274MB — embedding model
ollama pull llama3.2           # ~2GB  — language model
```

Verify:
```powershell
ollama list
```

### Step 3 — Clone the repo

```powershell
git clone https://github.com/SoniRe/VecDB.git
cd VecDB
```

### Step 4 — Compile

```powershell
g++ -std=c++17 -O2 main.cpp -o vecdb.exe -lws2_32 -lwsock32
```

### Step 5 — Run

**Terminal 1:**
```powershell
ollama serve
```

**Terminal 2:**
```powershell
./vecdb.exe
```

Open browser at `http://localhost:8080`

---

## Using the App

### Search Tab
- 20 pre-loaded 16D demo vectors across 4 categories: CS, Math, Food, Sports
- Type a query concept or paste a raw vector
- Pick algorithm (HNSW / KD-Tree / Brute Force) and metric (Cosine / Euclidean / Manhattan)
- See results with distances + matching point highlighted on PCA scatter plot

### Documents Tab
- Paste any text with a title
- Ollama embeds it as 768D vectors, auto-chunked at 200 words
- Stored in a separate HNSW index

### Ask AI Tab
- Add documents first in Documents tab
- Type any question about your documents
- VecDB embeds your question → HNSW finds top-3 most relevant chunks → llama3.2 generates an answer

---

## REST API

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/search` | K-NN search with body `{query, k, algo, metric}` |
| POST | `/insert` | Insert a vector `{id, vector}` |
| GET | `/vectors` | Get vector count |
| GET | `/status` | Ollama online status |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/documents` | `{"title":"...","text":"..."}` | Embed and store document |
| GET | `/documents` | — | List all stored document chunks |
| POST | `/ask` | `{"question":"..."}` | RAG: retrieve + generate answer |

### Example

```powershell
# Search
curl -X POST http://localhost:8080/search `
  -H "Content-Type: application/json" `
  -d '{"query":[0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],"k":3,"algo":"hnsw","metric":"cosine"}'

# Ask AI
curl -X POST http://localhost:8080/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is dynamic programming?"}'
```

---

## Common Issues

| Problem | Fix |
|---|---|
| `undefined reference to WSA...` | Add `-lws2_32 -lwsock32` to compile command |
| `g++: command not found` | Add `C:\msys64\ucrt64\bin` to Windows PATH |
| Ollama offline | Run `ollama serve` in a terminal |
| Port 8080 in use | `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F` |
| LLM answer is slow | Normal on CPU — use `llama3.2:1b` for faster responses |

### Use a Smaller/Faster LLM
```powershell
ollama pull llama3.2:1b
```
Then change `gen_model` in `ollama.h`:
```cpp
this -> gen_model = "llama3.2:1b";
```
Recompile and restart.

---

## Built With

- **C++17** — core vector database engine
- **cpp-httplib** — single-header HTTP server
- **Ollama** — local LLM inference (nomic-embed-text + llama3.2)
- **Vanilla JS + CSS** — frontend UI with PCA scatter plot

---

<div align="center">
Built from scratch as an educational deep-dive into how production vector databases work internally.
</div>
