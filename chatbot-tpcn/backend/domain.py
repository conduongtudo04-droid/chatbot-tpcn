from typing import Dict, Any, List
from rag import MiniRAG

def guard_notes(profile: Dict[str, Any]) -> List[str]:
    notes = []
    if profile.get("pregnant"): notes.append("Lưu ý thai kỳ: ưu tiên sản phẩm an toàn; tránh hoạt chất nhạy cảm.")
    if profile.get("ulcer"): notes.append("Lưu ý loét dạ dày: tránh hoạt chất gây kích ứng dạ dày.")
    return notes

def build_protocol_text(proto: str) -> str:
    return proto or "Phác đồ tham khảo 7–14 ngày; theo dõi đáp ứng sau 3–5 ngày."

def suggest_for_query(rag: MiniRAG, query: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    hits = rag.search(query, topk=6)
    found_symptom = next((h for h in hits if h["meta"]["type"]=="symptom"), None)
    combos, products = [], []

    if found_symptom:
        s = rag.get_symptom(found_symptom["meta"]["id"])
        for cid in s.get("combos", [])[:3]:
            c = rag.get_combo(cid);  combos.append(c) if c else None
        for sku in s.get("first_line_products", [])[:3]:
            p = rag.get_product(sku); products.append(p) if p else None
        response_type = "symptom"
        protocol = build_protocol_text(s.get("protocol"))
        triage = s.get("triage_questions", [])
        red_flags = s.get("red_flags", [])
    else:
        for h in [x for x in hits if x["meta"]["type"]=="product"][:3]:
            p = rag.get_product(h["meta"]["id"]); products.append(p) if p else None
        for h in [x for x in hits if x["meta"]["type"]=="combo"][:2]:
            c = rag.get_combo(h["meta"]["id"]); combos.append(c) if c else None
        response_type = "fallback"; protocol=""; triage=[]; red_flags=[]

    return {
        "type": response_type,
        "query": query,
        "triage_questions": triage,
        "red_flags": red_flags,
        "products": [{
            "sku": p.get("sku"), "name": p.get("name"),
            "benefits": p.get("benefits", []),
            "directions": p.get("directions",""),
            "warnings": p.get("warnings",""),
            "price_text": p.get("price_text",""),
            "pv": p.get("pv"),
            "link": p.get("link","")
        } for p in products],
        "combos": [{
            "id": c.get("id"), "name": c.get("name"),
            "targets": c.get("targets", []),
            "items": c.get("items", []),
            "protocol": c.get("protocol",""),
            "notes": c.get("notes","")
        } for c in combos if c],
        "protocol": protocol,
        "safety_notes": guard_notes(profile),
        "disclaimer": "Thông tin tham khảo nội bộ; không thay thế tư vấn y tế khi có dấu hiệu bất thường."
    }
