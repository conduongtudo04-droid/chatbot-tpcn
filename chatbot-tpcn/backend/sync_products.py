# backend/sync_products.py
import os, re, json, time, hashlib, datetime as dt
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from slugify import slugify

BASE_URL = "https://example.com/"  # 🔁 ĐỔI thành website công ty
SITEMAP_PATHS = ["sitemap.xml", "product-sitemap.xml", "sitemap_products.xml"]  # thử lần lượt

# ====== HTML SELECTORS fallback (chỉnh theo site) ======
SEL = {
    "name": ["h1.product-title", "h1.entry-title"],
    "sku":  [".sku", "span[itemprop='sku']"],
    "desc": [".product-short-description", ".entry-content p"],
    "benefits": [".benefits li"],
    "directions": [".directions, .usage, .huong-dan"],
    "warnings": [".warnings, .canh-bao"],
    "tags": [".product-tags a", ".tags a"]
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_FILE = os.path.join(DATA_DIR, "products.json")
BACKUP = os.path.join(DATA_DIR, f"products.{dt.datetime.now():%Y%m%d%H%M}.bak.json")
TIMEOUT = 20

def http_get(url):
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent":"TPCN-Bot/1.0"})
    r.raise_for_status()
    return r

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return []

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def sha256(data)->str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def pick_text(soup, selectors):
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            t = " ".join(el.get_text(" ", strip=True).split())
            if t: return t
    return ""

def pick_list(soup, selectors):
    for sel in selectors:
        els = soup.select(sel)
        if els:
            vals = []
            for e in els:
                t = " ".join(e.get_text(" ", strip=True).split())
                if t: vals.append(t)
            if vals: return vals
    return []

def parse_jsonld(soup):
    for tag in soup.find_all("script", {"type":"application/ld+json"}):
        try:
            data = json.loads(tag.string or "")
            # có thể là list hoặc object
            items = data if isinstance(data, list) else [data]
            for it in items:
                if (isinstance(it, dict) and it.get("@type") in ["Product", "DietarySupplement", "ProductModel"]):
                    name = it.get("name") or ""
                    sku = it.get("sku") or it.get("mpn") or ""
                    desc = it.get("description") or ""
                    offers = it.get("offers") or {}
                    return {
                        "name": name, "sku": sku, "description": desc,
                        "price": (offers.get("price") if isinstance(offers, dict) else None)
                    }
        except Exception:
            pass
    return None

def is_product_url(url):
    # heuristics: URL chứa /product/ hoặc /san-pham/
    return any(k in url for k in ["/product/", "/san-pham/", "/products/"])

def collect_from_sitemap():
    urls = set()
    for path in SITEMAP_PATHS:
        try:
            sm_url = urljoin(BASE_URL, path)
            r = http_get(sm_url)
            soup = BeautifulSoup(r.text, "xml")
            for loc in soup.find_all("loc"):
                u = loc.get_text(strip=True)
                if is_product_url(u):
                    urls.add(u)
        except Exception:
            continue
    return sorted(urls)

def parse_product(url):
    try:
        r = http_get(url)
        soup = BeautifulSoup(r.text, "lxml")

        # 1) JSON-LD trước
        jd = parse_jsonld(soup) or {}

        # 2) Fallback selectors
        name = jd.get("name") or pick_text(soup, SEL["name"]) or ""
        sku  = jd.get("sku")  or pick_text(soup, SEL["sku"]) or ""
        desc = jd.get("description") or pick_text(soup, SEL["desc"])
        benefits = pick_list(soup, SEL["benefits"])
        directions = pick_text(soup, SEL["directions"])
        warnings = pick_text(soup, SEL["warnings"])
        tags = pick_list(soup, SEL["tags"])

        if not sku:
            # tạo mã tạm dựa trên slug name (giữ ổn định)
            sku = ("SKU-" + slugify(name)[:24]).upper()

        item = {
            "sku": sku,
            "name": name,
            "description": desc,
            "benefits": benefits,
            "directions": directions,
            "warnings": warnings,
            "tags": tags,
            "link": url
        }
        return item
    except Exception as e:
        print(f"[ERR] {url} -> {e}")
        return None

def main():
    print("[SYNC] Collect product URLs from sitemap...")
    urls = collect_from_sitemap()
    if not urls:
        print("[WARN] Không tìm thấy URL sản phẩm từ sitemap. Có thể site dùng cấu trúc khác. Thử BASE_URL/products/ ...")
        # fallback nhẹ: crawl trang danh mục phổ biến (tuỳ biến nếu cần)
        urls = []

    products = []
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        p = parse_product(url)
        if p: products.append(p)
        time.sleep(0.1)  # lịch sự

    if not products:
        print("[DONE] Không thu được sản phẩm mới.")
        return

    # Hợp nhất với products.json cũ theo SKU (update theo SKU)
    old = {p["sku"]: p for p in load_json(OUT_FILE)}
    for p in products:
        old[p["sku"]] = p
    merged = list(old.values())

    # Chỉ ghi khi khác nội dung (để tránh reindex thừa)
    old_hash = sha256(load_json(OUT_FILE))
    new_hash = sha256(merged)
    if old_hash != new_hash:
        if os.path.exists(OUT_FILE):
            save_json(BACKUP, load_json(OUT_FILE))
            print(f"[BACKUP] -> {BACKUP}")
        save_json(OUT_FILE, merged)
        print(f"[WRITE] Updated {OUT_FILE} ({len(merged)} items)")
    else:
        print("[SKIP] No change detected, keep current products.json")

if __name__ == "__main__":
    main()
