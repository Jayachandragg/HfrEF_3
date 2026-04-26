import os
import numpy as np

# ── Lazy-loaded globals ───────────────────────────────────────
_model = None
_client = None
chunks = []
index = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return _client

PROJECT_CONTEXT = """
PIXEL MINDS — HFrEF AI Agent Project
Team: Jayachandra Galda, Hema Priya Balaji, Srinivasa Rao Tummalapalli
Client: Dr. Ciprian Ionita, QAS.AI
University at Buffalo — CDA Project — Spring 2026
Project Full Name: Trajectory-Integrated Decision Engine for Heart Failure (TIDE-HF) Phase 1A

THE PROBLEM: Heart failure patients are prescribed 4 drugs and told to return in 2 weeks.
Only 1% of HFrEF patients are on all 4 drugs at target doses. 6.7M US patients. $30B annual cost.

OUR SOLUTION: 5 wearable sensors monitor continuously. AI agent reads sensors every cycle.
7-step clinical logic decides drug adjustments in real time. No clinic visit needed.

5 SENSORS: ECG Patch (heart rhythm, drives BB), Blood Pressure (RAAS gate SBP>=100),
Heart Rate+HRV (BB titration, HRV early warning), SpO2 (emergency <90%),
Impedance Patch (lung fluid% WET>35% BORDERLINE 30-35% DRY<30% drives ALL drugs).

4 GDMT DRUGS: RAAS Inhibitor (Sacubitril/Valsartan ARNI preferred),
Beta Blocker (Carvedilol/Metoprolol/Bisoprolol), Diuretic (Furosemide/Torsemide),
MRA (Spironolactone/Eplerenone), SGLT2 (Dapagliflozin/Empagliflozin fixed 10mg).

7-STEP LOGIC ENGINE:
Step 1 Emergency: SpO2<90 SBP<90 K+>6.0 Creat>3.5 HR<40 eGFR<15 = STOP ALL alert clinician.
Step 2 Fluid: Impedance >35%=WET 30-35%=BORDERLINE <30%=DRY.
Step 3 Diuretic: WET+safe=INCREASE. WET+Creat rose 50%=ESCALATE IV. DRY+Creat rising=REDUCE. K+<3.5=REDUCE.
Step 4 RAAS: 3 gates ALL must pass: SBP>=100 AND K+<5.5 AND eGFR>=30. ARNI preferred. ACEi->ARNI 36hr washout.
Step 5 BB: DRY only (dry before you try). WET/BORDERLINE=SKIP. DRY+HR>target=UPTITRATE. HR<50=REDUCE. COPD=Bisoprolol/Metoprolol only.
Step 6 SGLT2+MRA: SGLT2 eGFR>=20 fixed 10mg. T1DM=CONTRAINDICATED. MRA K+<5.0+eGFR>=30=ADD. K+>5.5=REDUCE.
Step 7 Trajectory: Last 3 worsening=ESCALATE. HRV drop=early warning. Fluid down+stable=IMPROVING.

DATASET MIMIC-IV: 500000 admissions filtered to 33131 HFrEF -> 6319 patients with full data.
Final: 1599150 rows, 43 columns, 8255 admissions. Labs 2698813 records. Sensors 4185662. Drugs 267301.
Comorbidities: has_afib 10952, has_ckd 15479, has_t1dm 861, has_copd 5258.
43 columns: subject_id hadm_id charttime heart_rate sbp spo2 resp_rate weight_kg
creatinine potassium egfr sodium bun anion_gap ph hematocrit platelets wbc
dose_diuretic dose_raas dose_betablocker dose_mra dose_sglt2
drug_diuretic drug_raas drug_betablocker drug_mra drug_sglt2
has_afib has_ckd has_t1dm has_t2dm has_copd has_cardiomyopathy has_hypertensive_hd
flag_emergency_spo2 flag_low_sbp flag_high_potassium flag_critical_potassium
flag_high_creatinine flag_low_hr flag_high_hr flag_low_spo2

7 RESEARCH PAPERS:
1. AHA/ACC/HFSA 2022 HF Guidelines - foundation 4 GDMT drugs titration targets
2. ACC 2024 Expert Consensus Pathway - wet/dry concept 3-gate RAAS trajectory monitoring
3. Wearable Sensors Remote HF Monitoring JACC 2023 - SpO2 thresholds HRV impedance
4. Diuretic Titration Kidney Function Michigan Med ESC CKJ 2022-23 - creatinine K+ thresholds
5. Beta Blocker Titration Circulation Frontiers 2023 - dry-before-you-try HR targets 70/110 COPD
6. RAAS Inhibitors ACEi ARB ARNI ACC 2024 StatPearls - ARNI preference 36hr washout
7. SGLT2 MRA HFrEF StatPearls 2025 ESC 2023 DAPA-HF EMPEROR-Reduced - fixed 10mg T1DM eGFR>=20

COMPETITIVE: CardioMEMS monitors only invasive no drug logic.
Medly rules-based nurse confirms no wearables. Story Health coaching no sensors.
Our system: continuous wearables + automated 4-drug logic + non-invasive + 24/7.

TECH STACK: LangChain LangGraph Python Pydantic Claude Sonnet MIMIC-IV DuckDB FastAPI LangSmith
RAG: SentenceTransformers all-MiniLM-L6-v2 FAISS Groq Llama-3.3-70B.

RESULTS 100 PATIENTS: Emergency 21.8%. WET 52.4% BORDERLINE 28.1% DRY 19.5%.
Diuretic INCREASE 44.1% HOLD 31.2% REDUCE 17.6% ESCALATE 7.1%.
RAAS HOLD-SBP 48.3% UPTITRATE 29.7%. BB SKIP 58.2% UPTITRATE 14.8%.
Trajectory STABLE 54.3% WORSENING 32.1% IMPROVING 13.6%.

CHALLENGES: No thoracic impedance in MIMIC-IV used resp_rate as proxy.
ICU retrospective data not outpatient. SGLT2 only 2020-2022 data. Trajectory single row not rolling window yet.

NEXT STEPS: Fix trajectory rolling window. Compare vs clinician decisions.
Build prediction model deterioration risk readmission. Clinical pilot simulation.
"""

def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    result = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            result.append(chunk)
    return result

def build_index(text):
    global chunks, index
    import faiss
    m = get_model()
    chunks = chunk_text(text)
    embeddings = m.encode(chunks).astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return len(chunks)

def load_project_knowledge():
    num = build_index(PROJECT_CONTEXT)
    print(f"Project knowledge loaded — {num} chunks")
    return num

def retrieve(question, k=5):
    if index is None:
        return []
    import faiss
    m = get_model()
    q_vec = m.encode([question]).astype('float32')
    _, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def answer(question):
    retrieved = retrieve(question)
    if not retrieved:
        return "Knowledge base not ready yet."
    context = "\n\n".join(retrieved)
    prompt = f"""You are an expert assistant for the PIXEL MINDS HFrEF AI Agent project at University at Buffalo.
Answer clearly using only the context below. Be concise and use bullet points for lists.

Context:
{context}

Question: {question}
Answer:"""
    client = get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
