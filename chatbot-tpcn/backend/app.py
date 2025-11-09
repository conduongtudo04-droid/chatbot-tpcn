from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
from rag import MiniRAG
from domain import suggest_for_query

app = FastAPI(title="TPCN Advisor Bot", version="0.1.0")

# CORS
ALLOWED = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,      # production: set domain Netlify của anh
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAG = MiniRAG()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")

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
    return {"status":"ok"}

@app.post("/ask")
def ask(req: AskReq):
    return suggest_for_query(RAG, req.query, req.profile.model_dump())

@app.post("/admin/reindex")
def reindex(x_admin_token: str = Header(default="")):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    stats = RAG.reload()
    return {"status":"reindexed", **stats}
