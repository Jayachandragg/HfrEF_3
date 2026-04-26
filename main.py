from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pdfplumber
import io
import threading
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ──────────────────────────────────────────────────────
_rag = None
_ready = False

def load_rag():
    global _rag, _ready
    import rag as r
    r.load_project_knowledge()
    _rag = r
    _ready = True
    print("RAG ready!")

# Start loading after a short delay so uvicorn binds the port first
def delayed_load():
    import time
    time.sleep(3)
    load_rag()

threading.Thread(target=delayed_load, daemon=True).start()

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "PIXEL MINDS RAG API", "ready": _ready}

@app.get("/health")
def health():
    return {"status": "running", "ready": _ready}

@app.post("/ask")
def ask(request: QuestionRequest):
    if not _ready:
        return {"answer": "Knowledge base is still loading, please wait 30 seconds and try again."}
    return {"answer": _rag.answer(request.question)}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not _ready:
        return {"error": "Still loading, try again shortly."}
    contents = await file.read()
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            text = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
    else:
        text = contents.decode("utf-8")
    existing = " ".join(_rag.chunks)
    num = _rag.build_index(existing + "\n\n" + text)
    return {"message": "Added", "chunks_created": num}
