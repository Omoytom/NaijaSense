# NaijaSense: Context-Aware Localized Multi-Agent E-Commerce Simulation & Recommendation Engine

---

##  Overview

**NaijaSense** is an intelligent e-commerce personalization and behavioral simulation framework engineered to resolve the deep mismatch between global recommendation logic and localized developing market realities (e.g., the West African commercial ecosystem).

Standard storefront engines optimize linearly for high purchase conversion or vector text similarity under the assumption of perfect infrastructure. When deployed in environments bounded by **intermittent power grids, volatile utility stability, and metered, high-cost mobile data bundles**, these legacy models collapse—frequently recommending unviable, high-wattage, or bandwidth-heavy assets.

NaijaSense addresses this by decoupling recommendation and review generation into localized context layers, processing product manifests via competitive multi-agent arbitration loops.

---

##  System Architecture & Core Capabilities

The repository is split into two interconnected execution phases:

### Task A: Decoupled Context & Linguistic Review Simulation

* **The Critic Head:** Maps incoming consumer historical transactions to extract latent socioeconomic and infrastructure risk constraints. It acts as an analytical filter, adjusting standard 5-star product ratings based on real-world constraints (e.g., docking an electronics product score if it demands excessive power overhead).
* **The Voice Head:** Translates core product utility vectors into an age-stratified, hyper-localized first-person textual review, embedding localized phrasing and regional slang while preserving structural grading intent.

### Task B: Multi-Agent Storefront Arbitration

Instead of relying on single-prompt ranking models susceptible to cognitive drift, Task B implements a competitive panel of specialized micro-agents:

1. **The Infrastructure Realist:** Evaluates items on the candidate shelf space against grid volatility boundaries and data caps (e.g., *"Will this asset trip a shared 0.9kVA generator or exhaust a midnight data bundle?"*).
2. **The Value Budget Hawk:** Critiques price-to-utility returns tailored to the cohort's dynamic spending metrics.
3. **The Technical/Narrative Visionary:** Assesses component longevity, hardware specifications, and narrative storytelling values.

A centralized **Master Arbitrator** synthesizes this qualitative debate, resolving conflicting feedback loops to generate a fully optimized, structured recommendations array.

---

## 📁 Repository Structure

```text
├── data/
│   ├── staging_dataset.json      # Offline compiled high-fidelity dataset cache
│   ├── user_registry.json        # Extracted psychographic behavioral identity registry
│   └── evaluation_report.json    # Auto-logged accuracy precision summaries
├── src/
│   ├── agents/
│   │   ├── user_modelling.py     # Task A: Dynamic user modeling simulation engine
│   │   └── recommender.py        # Task B: Multi-agent storefront arbitrator loops
│   ├── core/
│   │   └── registry.py           # Identity mapping framework & footprint aggregator
│   ├── evaluation/
│   │   └── evaluator.py          # Validation engine tracking RMSE and ROUGE-L precision
│   ├── models/
│   │   └── schemas.py            # Strict structural Pydantic data contracts
│   ├── utils/
│   │   └── data_loader.py        # High-speed data loading & title extraction pipeline
│   └── main.py                   # Production FastAPI service root
├── Dockerfile                    # Production environment containerization configuration
├── requirements.txt              # Project pipeline dependency manifest
└── README.md                     # Technical repository documentation

```

---

##  Data Engineering & Ingestion Pipeline

NaijaSense fast-streams targeted segment slices from the **McAuley-Lab Amazon Reviews 2023 dataset** across three distinct domains: `Books`, `Electronics`, and `Movies_and_TV`.

### The Structural Metadata Challenge

Raw review streams are highly normalized, containing abstract alphanumeric tracking handles (**ASIN codes**) rather than clean, explicit product text. Compiling multi-gigabyte production metadata tables during tight validation deployment cycles would result in system resource exhaustion.

### The Solution: Zero-API Offline Text-Mining Title Reconstructor

To eliminate text-twisting hallucinations (e.g., converting arbitrary reviews into broken product strings like `"The Always Read Edition"`), `data_loader.py` includes a native token frequency parsing pipeline that acts completely offline:

1. Strips unescaped punctuation, emojis, and broken text fragments.
2. Filters out structural stop-words (`this`, `that`, `with`, `have`, `they`).
3. Extracts the top 3 dominant capitalized contextual keywords directly from the review body.
4. Appends a predictable, domain-specific suffix handle (`System Gear`, `Cinema Release`, `Edition`).

This approach achieves **zero-latency catalog labeling** while maintaining **absolute data purity** by completely stripping out broken product image links, optimizing token consumption and frontend network overhead.

---

##  Local Setup & Execution Installation

### 1. Environment Activation

Clone this repository to your local runtime partition and install the required tracking packages:

```bash
git clone https://github.com/your-username/NaijaSense.git
cd NaijaSense
pip install -r requirements.txt

```

### 2. Configuration Settings

Create a `.env` file in the root environment directory and mount your access key safely:

```env
GEMINI_API_KEY="your_GemininAPI_Key"

```

### 3. Compilation of Data Layers

Run the automated pipeline execution sequentially from your terminal window to populate your database staging files:

```bash
# A. Ingest records and generate non-hallucinatory descriptive titles
python -m src.utils.data_loader

# B. Build the behavioral identity mapping matrix registry
python -m src.core.registry

```

### 4. Running the Local API Server

Spin up the backend routing engine using Uvicorn to host your API endpoints locally:

```bash
uvicorn src.main:app --reload --host 127.0.0.1 --port 8000

```

Open your browser and navigate to `http://127.0.0.1:8000/docs` to access the auto-generated interactive Swagger UI portal.

---

##  Docker Production Deployment

To ensure environment consistency across separate engineering workstations and cloud services (such as Render), you can build and run the entire application stack as an isolated container containerized over a lightweight base image.

### Build the Image Context

```bash
docker build -t naijasense-backend .

```

### Spin Up the Container Globally

```bash
docker run -p 8000:8000 -e GEMINI_API_KEY="your_actual_key_here" naijasense-backend

```

---

##  Verification & Validation Suite

The engine includes an automated statistical evaluation framework (`src/evaluation/evaluator.py`) that tests systemic divergence against original behavioral benchmarks.

The evaluator includes an integrated **4-second pacing middleware** to prevent request exhaustion and maintain execution stability under free-tier API rate limits (15 Requests Per Minute).

To run a verification sweep across random catalog profile vectors, run:

```bash
python -m src.evaluation.evaluator

```

### Key Performance Metric Targets

* **Rating Calibration Variance (RMSE):** Captures the score friction applied by the *Critic Head* when docking high-rated assets due to local utility limitations.
* **Linguistic Overlap (ROUGE-L):** Verifies style-shifting, local phrase injection, and original review transformation efficiency. All execution traces log directly into `data/evaluation_report.json` for validation analysis.
