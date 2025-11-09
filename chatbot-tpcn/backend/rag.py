# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Tuple
import json, os, re, logging
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Cấu hình nguồn dữ liệu ===
PRODUCTS_URL: str = "https://script.google.com/macros/s/AKfycbyJts3RIVGN5WH5fICTy4lLAs-qHBazygK1FR_mK_Adwy8QCGj594bThi6W-7wCIu-qhw/exec"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOCAL_PRODUCTS = os.path.join(DATA_DIR, "products.json")
LOCAL_COMBOS   = os.path.join(DATA_DIR, "combos.json")
LOCAL_SYMPTOMS = os.path.join(DATA_DIR, "symptoms.json")
HTTP_TIMEOUT = 20

log = logging.getLogger("RAG")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

def _load_json_local(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_products_from_url():
    if (not PRODUCTS_URL) or ("REPLACE_ME" in PRODUCTS_URL):
        return _load_json_local(LOCAL_PRODUCTS)
    try:
        r = requests.get(
            PRODUCTS_URL,
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": "GW-AdvisorBot/1.0"}
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            data = data.get("items", [])
        return data or []
    except Exception as e:
        log.warning(f"[RAG] load products from URL failed: {e}")
        return _load_json_local(LOCAL_PRODUCTS)

def _norm(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def _join(xs) -> str:
    if not xs:
        return ""
    if isinstance(xs, (list, tuple)):
        return " ".join(_norm(str(x)) for x in xs)
    return _norm(str(xs))

class MiniRAG:
    def __init__(self) -> None:
        self.vectorizer = None
        self.matrix = None
        self.index_docs: List[str] = []
        self.meta: List[Dict[str, Any]] = []
        self.products: List[Dict[str, Any]] = []
        self.combos: List[Dict[str, Any]] = []
        self.symptoms: List[Dict[str, Any]] = []
        self._load_all()

    def reload(self) -> Dict[str, Any]:
        self._load_all()
        return {
            "ok": True,
            "counts": {
                "products": len(self.products),
                "combos": len(self.combos),
                "symptoms": len(self.symptoms),
            },
        }

    def search(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        if (not query.strip()) or (not self.index_docs):
            return []
        qv = self.vectorizer.transform([_norm(query)])
        sims = cosine_similarity(qv, self.matrix)[0]
        idxs = sims.argsort()[::-1][:max(1, topk)]
        return [{"score": float(sims[i]), "meta": self.meta[i]} for i in idxs]

    def get_product(self, sku: str):
        return next((p for p in self.products if p.get("sku") == sku), None)

    def get_combo(self, cid: str):
        return next((c for c in self.combos if c.get("id") == cid), None)

    def get_symptom(self, sid: str):
        return next((s for s in self.symptoms if s.get("id") == sid), None)

    def _load_all(self) -> None:
        self.products = _load_products_from_url()
        self.combos = _load_json_local(LOCAL_COMBOS)
        self.symptoms = _load_json_local(LOCAL_SYMPTOMS)

        self.index_docs, self.meta = self._build_corpus()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.index_docs or [""])
        log.info(
            f"[RAG] P/C/S={len(self.products)}/{len(self.combos)}/{len(self.symptoms)} "
            f"docs={len(self.index_docs)}"
        )

    def _build_corpus(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        docs: List[str] = []
        meta: List[Dict[str, Any]] = []

        for p in self.products:
            docs.append(" | ".join([
                _norm(p.get("name", "")),
                _norm(p.get("description", "")),
                _join(p.get("benefits", [])),
                _norm(p.get("directions", "")),
                _norm(p.get("warnings", "")),
                _join(p.get("tags", [])),
                _norm(p.get("brand", "")),
                _norm(p.get("price_text", "")),
                _join(p.get("category_path", [])),
            ]))
            meta.append({"type": "product", "id": p.get("sku")})

        for c in self.combos:
            docs.append(" | ".join([
                _norm(c.get("name", "")),
                _join(c.get("targets", [])),
                _norm(c.get("protocol", "")),
                _join([i.get("sku") for i in c.get("items", [])]),
                _norm(c.get("notes", "")),
            ]))
            meta.append({"type": "combo", "id": c.get("id")})

        for s in self.symptoms:
            docs.append(" | ".join([
                _norm(s.get("symptom", "")),
                _join(s.get("keywords", [])),
                _join(s.get("triage_questions", [])),
                _join(s.get("red_flags", [])),
                _norm(s.get("protocol", "")),
            ]))
            meta.append({"type": "symptom", "id": s.get("id")})

        return docs, meta
