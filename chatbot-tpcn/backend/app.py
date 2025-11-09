# backend/app.py  (bản có CORS)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from rag import MiniRAG
from domain import suggest_for_query

app = FastAPI(title="TPCN Advisor Bot", version="0.1.0")

# CORS: cho phép gọi từ file:// và localhost
origins = ["http://localhost", "http://127.0.0.1", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # có thể thu hẹp lại khi lên production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAG = MiniRAG()

class Profile(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    pregnant: Optional[bool] = False
    ulcer: Optional[bool] = False

class AskReq(BaseModel):
    query: str = Field(..., examples=["Khách bị đau lưng"])
    profile: Optional[Profile] = Profile()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskReq):
    result = suggest_for_query(RAG, req.query, req.profile.model_dump())
    return result

@app.post("/admin/reindex")
def reindex():
    stats = RAG.reload()
    return {"status":"reindexed", **stats}