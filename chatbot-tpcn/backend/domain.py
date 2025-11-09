# backend/domain.py
from typing import Dict, Any, List
from rag import MiniRAG

SAFETY = {
    "pregnancy": ["NSAID", "quế chi"],   # ví dụ minh hoạ
    "ulcer": ["NSAID"],
}

def guard_notes(profile: Dict[str, Any]) -> List[str]:
    notes = []
    if profile.get("pregnant"):
        notes.append("Lưu ý: phụ nữ có thai tránh nhóm hoạt chất nhạy cảm; ưu tiên sản phẩm an toàn thai kỳ.")
    if profile.get("ulcer"):
        notes.append("Lưu ý: tiền sử loét dạ dày – cần tránh hoạt chất gây kích ứng dạ dày.")
    return notes

def build_protocol_text(proto: str) -> str:
    return proto or "Phác đồ tham khảo 7–14 ngày; theo dõi đáp ứng sau 3–5 ngày."

def suggest_for_query(rag: MiniRAG, query: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    hits = rag.search(query, topk=6)
    found_symptom = next((h for h in hits if h["meta"]["type"]=="symptom"), None)
    combos, products = [], []

    if found_symptom:
        s = rag.get_symptom(found_symptom["meta"]["id"])
        # Lấy combos ưu tiên
        for cid in s.get("combos", [])[:3]:
            c = rag.get_combo(cid)
            if c: combos.append(c)
        # Lấy sản phẩm ưu tiên
        for sku in s.get("first_line_products", [])[:3]:
            p = rag.get_product(sku)
            if p: products.append(p)
        response_type = "symptom"
        protocol = build_protocol_text(s.get("protocol"))
        triage = s.get("triage_questions", [])
        red_flags = s.get("red_flags", [])
    else:
        # fallback: trả sản phẩm gần nhất + combos gần nhất
        prod_hits = [h for h in hits if h["meta"]["type"]=="product"][:3]
        combo_hits = [h for h in hits if h["meta"]["type"]=="combo"][:2]
        for h in prod_hits:
            p = rag.get_product(h["meta"]["id"])
            if p: products.append(p)
        for h in combo_hits:
            c = rag.get_combo(h["meta"]["id"])
            if c: combos.append(c)
        response_type = "fallback"
        protocol, triage, red_flags = "", [], []

    return {
        "type": response_type,
        "query": query,
        "triage_questions": triage,
        "red_flags": red_flags,
        "products": [{
            "sku": p["sku"], "name": p["name"],
            "benefits": p.get("benefits", []),
            "directions": p.get("directions",""),
            "warnings": p.get("warnings",""),
            "link": p.get("link","")
        } for p in products],
        "combos": [{
            "id": c["id"], "name": c["name"],
            "targets": c.get("targets", []),
            "items": c.get("items", []),
            "protocol": c.get("protocol",""),
            "notes": c.get("notes","")
        } for c in combos],
        "protocol": protocol,
        "safety_notes": guard_notes(profile),
        "disclaimer": "Thông tin tham khảo theo tài liệu nội bộ TPCN; không thay thế tư vấn y tế khi có dấu hiệu bất thường."
    }
