from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os

# ── INIT ──────────────────────────────────────────────────────
model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# Global state
chunks = []
index  = None

# ── PROJECT KNOWLEDGE BASE ────────────────────────────────────
# All project content pre-loaded so users can ask without uploading
PROJECT_CONTEXT = """
PIXEL MINDS — HFrEF AI Agent Project
Team: Jayachandra Galda, Hema Priya Balaji, Srinivasa Rao Tummalapalli
Client: Dr. Ciprian Ionita, QAS.AI
University at Buffalo — CDA Project — Spring 2026
Project Full Name: Trajectory-Integrated Decision Engine for Heart Failure (TIDE-HF) Phase 1A

=== PROJECT OVERVIEW ===
Our project builds a decision support system for patients with chronic heart failure (HFrEF).
It uses wearable sensor data and clinical records to support safe and timely medication decisions.
Currently most patients are monitored during hospital visits and medication adjustments are done every few weeks.
This creates a gap where a patient's condition may worsen without timely intervention.
The goal is not to replace doctors, but to support them with better, continuous insights.

=== THE PROBLEM ===
Heart failure patients are prescribed 4 drugs and told to return in 2 weeks.
A lot can go wrong in between — fluid builds silently in the lungs, blood pressure drops, kidneys strain.
Only 1% of HFrEF patients are on all 4 drugs at target doses simultaneously.
Annual US healthcare cost is $30 billion. 6.7 million Americans have heart failure.
50% 5-year mortality rate — worse than many cancers.

=== OUR SOLUTION ===
5 wearable sensors monitor the patient continuously.
An AI agent reads the sensors every cycle and applies 7-step clinical logic.
Drug decisions are made in real time — no clinic visit needed.
System continuously collects real-time physiological data, applies clinical guidelines,
provides safe medication recommendations, and alerts clinicians in case of risk.

=== 5 WEARABLE SENSORS ===
1. ECG Patch — Heart rhythm + arrhythmia detection. Drives Beta Blocker logic. Identifies AF which changes HR target from 70 to 110 bpm.
2. Blood Pressure Sensor — Systolic BP gate. Must be >= 100 mmHg before any RAAS uptitration.
3. Heart Rate + HRV — BPM target drives beta blocker decisions. HRV sudden drop is early warning.
4. SpO2 Sensor — Oxygen saturation. Below 90% is emergency override — all automated logic stops.
5. Impedance Patch — Lung fluid %. Above 35% = WET, below 30% = DRY. This classification drives ALL drug decisions.

=== 4 GDMT DRUG CLASSES ===
1. RAAS Inhibitor — Lisinopril, Sacubitril/Valsartan (ARNI preferred). Blocks harmful RAAS activation.
2. Beta Blocker — Carvedilol, Metoprolol, Bisoprolol. Reduces heart rate and oxygen demand.
3. Diuretic — Furosemide, Torsemide. Removes excess fluid from lungs and body.
4. MRA — Spironolactone, Eplerenone. Controls electrolytes and fluid, watch potassium.
5. SGLT2 Inhibitor — Dapagliflozin, Empagliflozin. Kidney + heart protection, fixed 10mg dose.

=== 7-STEP LOGIC ENGINE ===
Step 1 — Emergency Gates: Check SpO2<90%, SBP<90mmHg, K+>6.0, Creatinine>3.5, HR<40, eGFR<15. Any trigger = STOP ALL drugs, alert clinician.
Step 2 — Fluid Classification: Impedance patch reads lung fluid %. >35% = WET (prioritise diuresis). 30-35% = BORDERLINE (monitor). <30% = DRY (BB allowed).
Step 3 — Diuretic Decision: WET + labs safe = INCREASE. WET + Creat rose >50% = ESCALATE to IV. DRY + Creat rising = REDUCE (over-diuresis). K+<3.5 = REDUCE.
Step 4 — RAAS Inhibitor: Three gates must ALL pass: SBP>=100 AND K+<5.5 AND eGFR>=30. All pass = UPTITRATE (ARNI preferred). ACEi to ARNI needs 36hr washout.
Step 5 — Beta Blocker: Patient must be DRY first — dry before you try. WET/BORDERLINE = SKIP. DRY + HR>target = UPTITRATE. HR<50 = REDUCE. COPD = Bisoprolol/Metoprolol only.
Step 6 — SGLT2 + MRA: SGLT2 fixed 10mg if eGFR>=20. T1DM = CONTRAINDICATED. MRA: K+<5.0 + eGFR>=30 = ADD/MAINTAIN. K+>5.5 = REDUCE.
Step 7 — Trajectory: Last 3 readings all worsening = ESCALATE. HRV sudden drop = early warning. SBP dropped >20pts = flag. Fluid dropping + stable vitals = IMPROVING.

=== DATASET — MIMIC-IV ===
MIMIC-IV v3.1 from MIT + Beth Israel Deaconess Medical Center. 2008-2022. ~500,000 admissions.
Filtered using ICD codes: ICD-10 I50.20, I50.21, I50.22, I50.23, I50.40-I50.43. ICD-9 428.20-428.43.
Final pipeline: 5 scripts (k.py, k3.py, k4.py, k5.py, k6.py, k7.py)
Results: 33,131 HFrEF admissions → 6,319 patients with full sensor+drug data.
Final dataset: 1,599,150 rows, 43 columns, 8,255 admissions.
Lab records: 2,698,813 records. Sensor readings: 4,185,662. Drug records: 267,301.
Comorbidity flags: has_afib (10,952), has_ckd (15,479), has_t1dm (861), has_copd (5,258), has_t2dm (7,827).
Emergency flags: flag_emergency_spo2 (21,270 readings), flag_low_sbp (123,796), flag_high_potassium (5,743).

=== 43 COLUMNS IN FINAL DATASET ===
Identity: subject_id, hadm_id, charttime
Sensors: heart_rate, sbp, spo2, resp_rate, weight_kg
Labs: creatinine, potassium, egfr, sodium, bun, anion_gap, ph, hematocrit, platelets, wbc
Drug doses: dose_diuretic, dose_raas, dose_betablocker, dose_mra, dose_sglt2
Drug names: drug_diuretic, drug_raas, drug_betablocker, drug_mra, drug_sglt2
Comorbidity: has_afib, has_ckd, has_t1dm, has_t2dm, has_copd, has_cardiomyopathy, has_hypertensive_hd
Logic flags: flag_emergency_spo2, flag_low_sbp, flag_high_potassium, flag_critical_potassium, flag_high_creatinine, flag_low_hr, flag_high_hr, flag_low_spo2

=== 7 RESEARCH PAPERS ===
1. AHA/ACC/HFSA 2022 HF Guidelines — Foundation: 4 GDMT drugs, titration frequency, dose targets.
2. ACC 2024 Expert Consensus Pathway — Wet/dry concept, trajectory monitoring, 3-gate RAAS check.
3. Wearable Sensors & Remote HF Monitoring (JACC 2023) — SpO2 thresholds, HRV as early warning, impedance monitoring.
4. Diuretic Titration & Kidney Function (Michigan Med/ESC/CKJ 2022-23) — Creatinine safety thresholds, K+ tiers.
5. Beta Blocker Titration (Circulation/Frontiers CV 2023) — Dry-before-you-try, HR targets 70/110, COPD restriction.
6. RAAS Inhibitors: ACEi vs ARB vs ARNI (ACC 2024/StatPearls) — ARNI preference, 36hr washout rule.
7. SGLT2 + MRA in HFrEF (StatPearls 2025/ESC 2023/DAPA-HF/EMPEROR-Reduced) — Fixed 10mg, T1DM contraindication, eGFR>=20.

=== COMPETITIVE LANDSCAPE ===
CardioMEMS (Abbott): FDA approved Feb 2026. Monitors pulmonary pressure only. Invasive surgical implant. No drug decisions.
Medly (Toronto): Rules-based. Nurse confirms every decision. No continuous wearables.
Story Health (US): Health coaching. Patient-reported data. No sensors.
Our System: Continuous wearables + automated 4-drug logic + non-invasive + 24/7. Only system combining all.

=== TECH STACK ===
LangChain + LangGraph StateGraph, Python IF-ELSE + Pydantic, Claude Sonnet / GPT-4o,
Redis/SQLite state, MIMIC-IV + Sensor API, FastAPI + WebSocket, LangSmith observability.
RAG: SentenceTransformers all-MiniLM-L6-v2, FAISS vector index, Groq Llama-3.3-70B.

=== AGENT WORKFLOW — 9 TOOL CALLS PER CYCLE ===
1. fetch_sensor_data() — Reads all 5 sensors
2. check_emergency_gates() — Safety first, stop if danger
3. classify_fluid_status() — WET / BORDERLINE / DRY
4. call_diuretic_tool() — Furosemide up/down/hold
5. call_raas_tool() — 3-gate SBP + K+ + eGFR check
6. call_bb_tool() — Beta Blocker DRY only
7. call_sglt2_mra_tool() — SGLT2 10mg + MRA K+ gate
8. check_trajectory() — Trend of last 3 readings
9. log_and_output() — Structured JSON decision to EHR

=== KEY CLINICAL FINDINGS FROM DATA ===
47% of patients have CKD — kidney safety gates are critical for nearly half the population.
33% have Atrial Fibrillation — HR target changes from 70 to 110 bpm for these patients.
21,270 readings triggered SpO2 emergency gate — logic fires correctly on real ICU data.
Only 437 SGLT2 records — drug approved 2020, MIMIC ends 2022, expected limitation.
RAAS held on most rows due to SBP<100 — common in ICU, gate working correctly.
Beta blockers skipped on most rows — patients WET or BORDERLINE, dry-before-you-try firing.

=== CHALLENGES AND LIMITATIONS ===
Fluid Proxy: No thoracic impedance in MIMIC-IV. Used respiratory rate as proxy (resp>22=WET, <16=DRY).
ICU vs Outpatient: Retrospective ICU data, real titration happens post-discharge in clinic.
SGLT2 Coverage: Only 2 years of data (2020-2022) due to late approval.
Trajectory: Current version checks single row, not rolling 3-reading window yet.

=== NEXT STEPS ===
Fix trajectory rolling window logic.
Compare logic decisions vs actual clinician decisions in MIMIC.
Build prediction model (deterioration risk, readmission likelihood) on top of dataset.
Validate respiratory rate as fluid proxy or find impedance dataset.
Clinical pilot or simulation environment testing.
Phase 3: Reinforcement learning agent design.

=== MEETINGS ===
2/26 - Introductory Meeting
3/5 - Weekly Check-in and Scope Clarification
3/13 - Weekly Check-in and Dataset Discussion
3/26 - Weekly Check-in, Initial Dataset Findings and Titration Discussion

=== PROJECT RESULTS — 100 PATIENT SAMPLE ===
Emergency: 21.8% EMERGENCY, 78.2% SAFE
Fluid Status: 52.4% WET, 28.1% BORDERLINE, 19.5% DRY
Diuretic: 44.1% INCREASE, 31.2% HOLD, 17.6% REDUCE, 7.1% ESCALATE
RAAS: 48.3% HOLD(SBP<100), 29.7% UPTITRATE, 13.4% HOLD(K+), 8.6% HOLD(eGFR)
Beta Blocker: 58.2% SKIP(not dry), 21.4% HOLD, 14.8% UPTITRATE, 5.6% REDUCE
Trajectory: 54.3% STABLE, 32.1% WORSENING, 13.6% IMPROVING
"""

# ── CHUNKING ──────────────────────────────────────────────────
def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    result = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            result.append(chunk)
    return result

# ── INDEXING ──────────────────────────────────────────────────
def build_index(text):
    global chunks, index
    chunks = chunk_text(text)
    embeddings = model.encode(chunks).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return len(chunks)

# ── PRE-LOAD PROJECT KNOWLEDGE AT STARTUP ────────────────────
def load_project_knowledge():
    """Called once at startup — indexes all project content."""
    num = build_index(PROJECT_CONTEXT)
    print(f"✅ Project knowledge base loaded — {num} chunks indexed")
    return num

# ── RETRIEVAL ─────────────────────────────────────────────────
def retrieve(question, k=5):
    if index is None:
        return []
    q_vec = model.encode([question]).astype('float32')
    distances, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# ── ANSWER ────────────────────────────────────────────────────
def answer(question):
    retrieved = retrieve(question)

    if not retrieved:
        return "Knowledge base not loaded yet. Please try again in a moment."

    context = "\n\n".join(retrieved)

    prompt = f"""You are an expert assistant for the PIXEL MINDS HFrEF AI Agent project at University at Buffalo.
Answer questions about the project clearly and accurately using the context below.
Be concise but complete. Use bullet points for lists. If something is not in the context, say so honestly.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
