# HFrEF AI Agent — Sensor-Driven Medication Titration
### PIXEL MINDS · University at Buffalo · CDA Project · Spring 2026
### Client: Dr. Ciprian Ionita, QAS.AI

> An AI agent that reads wearable sensor data and automatically decides when to adjust 4 life-saving medications for Heart Failure (HFrEF) patients — 24 hours a day, no clinic visit needed.

🌐 **Live Website:** [jayachandragg.github.io/HfrEF_3](https://jayachandragg.github.io/HfrEF_3)

---

## Team

| Name | Role |
|------|------|
| Jayachandra Galda | AI/ML Engineering · Agent Architecture · Dataset Pipeline |
| Hema Priya Balaji | Decision Logic · 7-Step IF-ELSE Rule Formalisation |
| Srinivasa Rao Tummalapalli | Data Engineering · MIMIC-IV SQL Pipeline |

**Client:** Dr. Ciprian Ionita — QAS.AI

---

## What This Project Does

Heart failure patients are prescribed 4 drugs and told to return in 2 weeks. Only **1% of HFrEF patients** ever reach all 4 drugs at target doses simultaneously. A lot can go wrong in between — fluid builds silently, blood pressure drops, kidneys strain — and nobody is watching.

This system uses **5 wearable sensors** and a **7-step clinical logic engine** to continuously monitor patients, automatically decide when to adjust each of the 4 GDMT drugs, and alert clinicians when emergency thresholds are breached.

---

## Repository Contents

### Website
| File | Description |
|------|-------------|
| `index.html` | Full project showcase website (single HTML file). Contains the interactive live demo, auto-playing logic animation, RAG chat widget, project book PDF viewer, and all project sections. Hosted via GitHub Pages. |
| `logo-qas.jpg` | QAS.AI client logo used in the website. |

### Logic Engine
| File | Description |
|------|-------------|
| `logic_engine.py` | **Runnable Python script** — applies the complete 7-step medication titration logic to any patient dataset CSV. Accepts command-line arguments, prints a full decision summary, and saves results. Only requires `pandas` and `numpy`. |

### RAG Chatbot Backend
| File | Description |
|------|-------------|
| `main.py` | FastAPI backend server. Exposes `/ask`, `/upload`, and `/health` endpoints. Pre-loads all project knowledge at startup in a background thread. Deployed on Render. |
| `rag.py` | RAG pipeline using TF-IDF retrieval (no local model needed) and Groq Llama-3.3-70B for answering questions about the project. |
| `requirements.txt` | Python dependencies for the RAG backend — `fastapi`, `uvicorn`, `groq`, `pdfplumber`, `faiss-cpu`, `numpy`. |

### Documents
| File | Description |
|------|-------------|
| `Project_Book.pdf` | Full 40-page project documentation covering problem statement, clinical background, system design, dataset pipeline, 7-step logic engine, results on 100 patients, challenges, and next steps. |
| `HFrEF_Logic_Flowchart.pdf` | A2 portrait flowchart showing the complete 7-step IF-ELSE decision flow with all diamond gates, YES/NO branches, and drug outcomes for every scenario. Print at A2 for demo use. |
| `HFrEF_Full_Project_Presentation.pptx` | Full 12-slide project presentation deck covering the problem, 5 sensors, 7 research papers, system workflow, MIMIC-IV dataset, and logic engine with results. |
| `MIMIC_Dataset_Slides.pptx` | 4-slide focused deck on the MIMIC-IV dataset extraction and modification pipeline — ICD filtering funnel, 4 SQL extractions, final dataset schema, and key clinical findings. |

---

## Logic Engine — Quick Start

### Install
```bash
git clone https://github.com/Jayachandragg/HfrEF_3.git
cd HfrEF_3
pip install pandas numpy
```

### Run
```bash
# Run on first 100 patients (default)
python logic_engine.py

# Run on your own CSV
python logic_engine.py --input your_data.csv

# Run on all patients
python logic_engine.py --input hfref_final_dataset.csv --all

# Run on 500 patients and save to specific file
python logic_engine.py --patients 500 --output my_results.csv
```

### Required CSV Columns
```
subject_id, hadm_id, charttime,
heart_rate, sbp, spo2, resp_rate,
creatinine, potassium, egfr,
has_afib, has_t1dm, has_copd
```
All columns support missing values — the engine handles NaN safely.

### Output Format
One row per patient timestamp:
```
subject_id | hadm_id | charttime |
step1_emergency | step2_fluid | step3_diuretic |
step4_raas | step5_bb | step6_sglt2 | step6_mra |
step7_trajectory | alert | alert_reason
```

---

## The 7-Step Logic Engine

| Step | Name | Rule |
|------|------|------|
| 1 | Emergency Gates | SpO2 < 90%, SBP < 90, K+ > 6.0, Creat > 3.5, HR < 40, eGFR < 15 → STOP ALL |
| 2 | Fluid Classification | Impedance → WET (>35%) / BORDERLINE (30–35%) / DRY (<30%) |
| 3 | Diuretic | WET + safe → INCREASE · WET + Creat rose >50% → ESCALATE IV · DRY + Creat rising → REDUCE |
| 4 | RAAS Inhibitor | 3 gates: SBP ≥ 100 AND K+ < 5.5 AND eGFR ≥ 30 → UPTITRATE (ARNI preferred) |
| 5 | Beta Blocker | DRY only · HR > target → UPTITRATE · COPD → Bisoprolol/Metoprolol only |
| 6 | SGLT2 + MRA | SGLT2: eGFR ≥ 20, fixed 10mg, T1DM = CONTRAINDICATED · MRA: K+ < 5.0 + eGFR ≥ 30 |
| 7 | Trajectory | Last 3 readings worsening → ESCALATE · Fluid ↓ + stable → IMPROVING |

---

## Dataset — MIMIC-IV

Access requires credentialed PhysioNet account: [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/3.1/)

| Stage | Records | Description |
|-------|---------|-------------|
| Raw MIMIC-IV | ~500,000 | Full hospital database 2008–2022 |
| After ICD filter | 33,131 admissions | HFrEF only (I50.20, I50.22, I50.42...) |
| After sensor filter | 6,319 patients | Has complete sensor + drug + lab data |
| **Final dataset** | **1,599,150 rows · 43 columns** | Decision-ready |

---

## RAG Chatbot Backend

The website includes a live chat widget powered by the FastAPI backend on Render.

**To run locally:**
```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_groq_key_here"
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Endpoints:**
- `GET  /health` — check status and whether knowledge base is loaded
- `POST /ask`    — `{"question": "What is the 7-step logic?"}` → answer
- `POST /upload` — upload additional PDF or TXT to extend knowledge base

---

## Clinical Evidence Base

Every rule is sourced from peer-reviewed publications:

1. **AHA/ACC/HFSA 2022 HF Guidelines** — 4 GDMT drugs, titration targets
2. **ACC 2024 Expert Consensus Pathway** — Wet/dry concept, 3-gate RAAS, trajectory
3. **Wearable Sensors & Remote HF Monitoring** (JACC 2023) — SpO2, HRV, impedance thresholds
4. **Diuretic Titration & Kidney Function** (ESC/CKJ 2022–23) — Creatinine, K+ tiers
5. **Beta Blocker Titration** (Circulation/Frontiers 2023) — Dry-before-you-try, HR 70/110
6. **RAAS Inhibitors: ACEi vs ARB vs ARNI** (ACC 2024/StatPearls) — ARNI, 36hr washout
7. **SGLT2 + MRA in HFrEF** (DAPA-HF/EMPEROR-Reduced) — Fixed 10mg, T1DM, eGFR ≥ 20

---

## Key Results — 100 Patient Sample

| Decision | Outcome | % |
|----------|---------|---|
| Emergency triggered | EMERGENCY | 21.8% |
| Fluid status | WET | 52.4% |
| Diuretic | INCREASE | 44.1% |
| RAAS | HOLD — SBP gate | 48.3% |
| Beta Blocker | SKIP — not dry | 58.2% |
| Trajectory | STABLE | 54.3% |

---

## Disclaimer

This is a **research prototype** for academic purposes only. It is **not** a certified medical device and must **not** be used for real clinical decisions without physician oversight.

---

*PIXEL MINDS · University at Buffalo · MS Artificial Intelligence · Spring 2026*
