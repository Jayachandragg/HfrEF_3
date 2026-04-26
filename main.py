from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pdfplumber
import io
import rag

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load project knowledge at startup
    rag.load_project_knowledge()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "PIXEL MINDS — HFrEF RAG API is running"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload an additional document to extend the knowledge base."""
    contents = await file.read()

    if file.filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            text = "\n".join(
                page.extract_text() for page in pdf.pages
                if page.extract_text()
            )
    else:
        text = contents.decode("utf-8")

    # Append to existing knowledge base
    existing_text = " ".join(rag.chunks)
    combined = existing_text + "\n\n" + text
    num_chunks = rag.build_index(combined)

    return {
        "message": "Document added to knowledge base",
        "chunks_created": num_chunks
    }

@app.post("/ask")
def ask(request: QuestionRequest):
    response = rag.answer(request.question)
    return {"answer": response}

@app.get("/health")
def health():
    return {
        "status": "running",
        "knowledge_loaded": rag.index is not None,
        "chunks": len(rag.chunks)
    }
