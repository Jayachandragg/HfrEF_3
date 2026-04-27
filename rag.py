
import os, math, re
from groq import Groq

chunks = []
tfidf_matrix = []
vocab = {}
_idf = {}
index = True
_client = None

def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return _client

def tokenize(text):
    return re.findall(r'\b[a-z]{2,}\b', text.lower())

def build_tfidf(corpus):
    global vocab, tfidf_matrix, _idf
    tokenized = [tokenize(doc) for doc in corpus]
    all_words = set(w for doc in tokenized for w in doc)
    vocab = {w: i for i, w in enumerate(sorted(all_words))}
    N = len(corpus)
    for w in vocab:
        df = sum(1 for doc in tokenized if w in doc)
        _idf[w] = math.log((N+1)/(df+1)) + 1
    tfidf_matrix = []
    for doc_tokens in tokenized:
        tf = {}
        for w in doc_tokens: tf[w] = tf.get(w,0)+1
        total = len(doc_tokens) or 1
        tfidf_matrix.append({w:(tf.get(w,0)/total)*_idf[w] for w in vocab})

def cosine_sim(v1, v2):
    dot = sum(v1.get(w,0)*v2.get(w,0) for w in v1)
    m1 = math.sqrt(sum(x**2 for x in v1.values()))
    m2 = math.sqrt(sum(x**2 for x in v2.values()))
    return dot/(m1*m2+1e-9)

def chunk_text(text, size=120, overlap=20):
    words = text.split()
    result = []
    for i in range(0, len(words), size-overlap):
        c = " ".join(words[i:i+size])
        if c.strip(): result.append(c)
    return result

def build_index(text):
    global chunks
    chunks = chunk_text(text)
    build_tfidf(chunks)
    return len(chunks)

def retrieve(question, k=5):
    if not tfidf_matrix: return []
    q_tokens = tokenize(question)
    total = len(q_tokens) or 1
    tf = {}
    for w in q_tokens: tf[w] = tf.get(w,0)+1
    q_vec = {w:(tf.get(w,0)/total)*_idf.get(w,1) for w in vocab}
    scores = [(cosine_sim(q_vec,dv),i) for i,dv in enumerate(tfidf_matrix)]
    scores.sort(reverse=True)
    return [chunks[i] for _,i in scores[:k]]

PROJECT_CONTEXT = """
PIXEL MINDS HFrEF AI Agent Project University at Buffalo CDA Spring 2026
Team Jayachandra Galda Hema Priya Balaji Srinivasa Rao Tummalapalli
Client Dr Ciprian Ionita QAS.AI TIDE-HF Phase 1A

PROBLEM heart failure patients prescribed 4 drugs told return 2 weeks only 1 percent on all 4 drugs at target doses 6.7 million US patients 30 billion annual cost 50 percent 5 year mortality.

SOLUTION 5 wearable sensors monitor continuously AI agent reads sensors every cycle 7 step clinical logic decides drug adjustments real time no clinic visit needed.

SENSORS ECG Patch heart rhythm arrhythmia drives beta blocker atrial fibrillation HR target 110. Blood Pressure systolic BP gate 100 mmHg before RAAS. Heart Rate HRV BPM target beta blocker HRV early warning. SpO2 emergency below 90 stop all drugs. Impedance Patch lung fluid percent above 35 WET 30 to 35 BORDERLINE below 30 DRY drives all drugs.

DRUGS RAAS Inhibitor Sacubitril Valsartan ARNI preferred Lisinopril. Beta Blocker Carvedilol Metoprolol Bisoprolol. Diuretic Furosemide Torsemide removes fluid. MRA Spironolactone Eplerenone watch potassium. SGLT2 Dapagliflozin Empagliflozin fixed 10mg eGFR above 20.

LOGIC Step 1 Emergency SpO2 below 90 SBP below 90 potassium above 6 creatinine above 3.5 HR below 40 eGFR below 15 STOP ALL alert clinician. Step 2 Fluid above 35 WET 30 to 35 BORDERLINE below 30 DRY. Step 3 Diuretic WET INCREASE WET creatinine rose 50 percent ESCALATE IV DRY creatinine rising REDUCE potassium below 3.5 REDUCE. Step 4 RAAS three gates SBP above 100 AND potassium below 5.5 AND eGFR above 30 UPTITRATE ARNI preferred 36 hour washout ACEi to ARNI. Step 5 Beta Blocker DRY only dry before you try WET SKIP DRY HR above target UPTITRATE HR below 50 REDUCE COPD Bisoprolol Metoprolol only. Step 6 SGLT2 eGFR above 20 10mg T1DM CONTRAINDICATED MRA potassium below 5 eGFR above 30 ADD potassium above 5.5 REDUCE. Step 7 Trajectory 3 readings worsening ESCALATE HRV drop warning fluid down stable IMPROVING.

DATASET MIMIC-IV 500000 admissions filtered to 33131 HFrEF then 6319 patients 1599150 rows 43 columns. Labs 2698813 drug records 267301 sensors 4185662. Comorbidities afib 10952 ckd 15479 t1dm 861 copd 5258.

PAPERS AHA ACC HFSA 2022 guidelines. ACC 2024 Expert Consensus wet dry 3 gate RAAS. JACC 2023 wearables SpO2 HRV impedance. Michigan Med ESC diuretic kidney creatinine. Circulation Frontiers beta blocker dry before you try HR 70 110 COPD. StatPearls ARNI 36hr washout. DAPA-HF EMPEROR-Reduced SGLT2 fixed 10mg T1DM eGFR.

RESULTS 100 patients emergency 21.8 percent WET 52.4 BORDERLINE 28.1 DRY 19.5. Diuretic INCREASE 44.1 HOLD 31.2. RAAS HOLD SBP 48.3 UPTITRATE 29.7. BB SKIP 58.2 UPTITRATE 14.8. Trajectory STABLE 54.3 WORSENING 32.1 IMPROVING 13.6.

TECH STACK LangChain LangGraph Python FastAPI Groq Llama 3.3 70B TF-IDF retrieval MIMIC-IV DuckDB.
COMPETITIVE CardioMEMS monitors only invasive. Medly rules based nurse confirms. Story Health coaching no sensors. Our system continuous wearables automated 4 drug logic non invasive 24 7.
"""

def load_project_knowledge():
    num = build_index(PROJECT_CONTEXT)
    print(f"Knowledge loaded {num} chunks")
    return num

def answer(question):
    retrieved = retrieve(question)
    if not retrieved:
        return "Knowledge base not ready."
    context = "\n\n".join(retrieved)
    prompt = f"""You are an expert assistant for the PIXEL MINDS HFrEF AI Agent project at University at Buffalo. Answer clearly using only the context below. Be concise and use bullet points.

Context:
{context}

Question: {question}
Answer:"""
    response = get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

