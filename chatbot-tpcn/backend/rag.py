# -*- coding: utf-8 -*-
"""
rag.py — Mini RAG cho Smart Advisor TPCN
- Ưu tiên nạp sản phẩm từ URL (Google Apps Script Web App)
- Fallback sang data/products.json nếu URL lỗi
- Hỗ trợ reload nóng qua RAG.reload() (được gọi tại /admin/reindex)
"""

from typing import List, Dict, Any, Tuple
import json, os, re, logging
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ====== Cấu hình nguồn dữ liệu ======
# Đổi URL này thành Web App của Apps Script (Deploy → Web app → URL)
PRODUCTS_URL: str = "https://script.google.com/macros/s/AKfycbyJts3RIVGN5WH5fICTy4lLAs-qHBazygK1FR_mK_Adwy8QCGj594bThi6W-7wCIu-qhw/exec"

# Thư mục data local (fallback)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOCAL_PRODUCTS = os.path.join(DATA_DIR, "products.json")
LOCAL_COMBOS   = os.path.join(DATA_DIR, "combos.json")
LOCAL_SYMPTOMS = os.path.join(DATA_DIR, "symptoms.json")

HTTP_TIMEOUT = 20  # giây

log = logging.getLogger("RAG")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)


# ---------- Helpers ----------
def _load_json_local(path: str) -> Any:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_products_from_url() -> List[Dict[str, Any]]:
    if not PRODUCTS_URL or PRODUCTS_URL.startswith("https://script.google.com/macros/s/REPLACE"):
        # chưa cấu hình URL => dùng local
        return _load_json_local(LOCAL_PRODUCTS)
    try:
        r = requests.get(PRODUCTS_URL, timeout=HTTP_TIMEOUT, headers={"User-Agent": "GW-AdvisorBot/1.0"})
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            # phòng trường hợp trả về object {items:[...]}
            data = data.get("items", [])
        return data or []
    except Exception as e:
        log.warning(f"[RAG] load products from URL failed: {e}")
        return _load_json_local(LOCAL_PRODUCTS)

def _norm_text(s: str) -> str:
    """Chuẩn hoá chuỗi để index (gọn—ổn định)."""
    if not s: return ""
    s = re.sub(r"<[^>]+>", " ", s)      # bỏ thẻ HTML
    s = re.sub(r"\s+", " ", s)          # gom khoảng trắng
    return s.strip().lower()

def _join_list(xs: Any) -> str:
    if not xs: return ""
    if isinstance(xs, (list, tuple)):
        return " ".join([_norm_text(str(x)) for x in xs])
    return _norm_text(str(xs))


# ---------- Lõi RAG ----------
class MiniRAG:
    """
    - products: list[{sku,name,description,benefits[],directions,warnings,tags[],link,...}]
    - combos: list[{id,name,items,targets[],protocol,notes}]
    - symptoms: list[{id,symptom,keywords[],triage_questions[],red_flags[],first_line_products[],combos[],protocol}]
    """

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer = None  # type: ignore
        self.matrix = None
        self.index_docs: List[str] = []
        self.meta: List[Dict[str, Any]] = []
        self.products: List[Dict[str, Any]] = []
        self.combos: List[Dict[str, Any]] = []
        self.symptoms: List[Dict[str, Any]] = []
        self._load_all()

    # ---- Public API ----
    def reload(self) -> Dict[str, Any]:
        """Nạp lại toàn bộ dữ liệu & rebuild index (gọi từ /admin/reindex)."""
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
        if not query.strip():
            return []
        if self.matrix is None or not len(self.index_docs):
            return []
        qv = self.vectorizer.transform([_norm_text(query)])
        sims = cosine_similarity(qv, self.matrix)[0]
        idxs = sims.argsort()[::-1][:max(1, topk)]
        out: List[Dict[str, Any]] = []
        for i in idxs:
            out.append({"score": float(sims[i]), "meta": self.meta[i]})
        return out

    # Convenience getters
    def get_product(self, sku: str) -> Dict[str, Any] | None:
        return next((p for p in self.products if p.get("sku") == sku), None)

    def get_combo(self, cid: str) -> Dict[str, Any] | None:
        return next((c for c in self.combos if c.get("id") == cid), None)

    def get_symptom(self, sid: str) -> Dict[str, Any] | None:
        return next((s for s in self.symptoms if s.get("id") == sid), None)

    # ---- Internal ----
    def _load_all(self) -> None:
        # Nguồn sản phẩm: URL (Apps Script) -> fallback file local
        self.products = _load_products_from_url()
        self.combos   = _load_json_local(LOCAL_COMBOS)
        self.symptoms = _load_json_local(LOCAL_SYMPTOMS)

        # Xây lại corpus TF-IDF
        self.index_docs, self.meta = self._build_corpus()
        if self.index_docs:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self.matrix = self.vectorizer.fit_transform(self.index_docs)
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self.matrix = self.vectorizer.fit_transform([""])

        log.info(
            f"[RAG] loaded P/C/S = {len(self.products)}/{len(self.combos)}/{len(self.symptoms)} | docs={len(self.index_docs)}"
        )

    def _build_corpus(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        docs: List[str] = []
        meta: List[Dict[str, Any]] = []

        # Products → one doc per product
        for p in self.products:
            text = " | ".join([
                _norm_text(p.get("name", "")),
                _norm_text(p.get("description", "")),
                _join_list(p.get("benefits", [])),
                _norm_text(p.get("directions", "")),
                _norm_text(p.get("warnings", "")),
                _join_list(p.get("tags", [])),
                _norm_text(p.get("brand", "")),
                _norm_text(p.get("price_text", "")),
                _norm_text(p.get("category_path", " > ".join(p.get("category_path", [])))),
            ])
            docs.append(text)
            meta.append({"type": "product", "id": p.get("sku")})

        # Combos → one doc per combo
        for c in self.combos:
            text = " | ".join([
                _norm_text(c.get("name", "")),
                _join_list(c.get("targets", [])),
                _norm_text(c.get("protocol", "")),
                _join_list([i.get("sku") for i in c.get("items", [])]),
                _norm_text(c.get("notes", "")),
            ])
            docs.append(text)
            meta.append({"type": "combo", "id": c.get("id")})

        # Symptoms → one doc per symptom
        for s in self.symptoms:
            text = " | ".join([
                _norm_text(s.get("symptom", "")),
                _join_list(s.get("keywords", [])),
                _join_list(s.get("triage_questions", [])),
                _join_list(s.get("red_flags", [])),
                _norm_text(s.get("protocol", "")),
            ])
            docs.append(text)
            meta.append({"type": "symptom", "id": s.get("id")})

        return docs, meta
