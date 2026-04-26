from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
import io
import threading
import rag

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load knowledge in background so server starts immediately ──
def init_knowledge():
    print("Loading model and knowledge base in background...")
    rag.load_project_knowledge()
    print("Knowledge base ready!")

threading.Thread(target=init_knowledge, daemon=True).start()

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "PIXEL MINDS — HFrEF RAG API is running"}

@app.get("/health")
def health():
    return {
        "status": "running",
        "knowledge_loaded": rag.index is not None,
        "chunks": len(rag.chunks)
    }

@app.post("/ask")
def ask(request: QuestionRequest):
    if rag.index is None:
        return {"answer": "Still loading knowledge base, please try again in 30 seconds."}
    response = rag.answer(request.question)
    return {"answer": response}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            text = "\n".join(
                page.extract_text() for page in pdf.pages
                if page.extract_text()
            )
    else:
        text = contents.decode("utf-8")

    existing_text = " ".join(rag.chunks)
    combined = existing_text + "\n\n" + text
    num_chunks = rag.build_index(combined)
    return {"message": "Document added", "chunks_created": num_chunks}
