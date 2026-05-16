"""
Robust scraper for SHL Individual Test Solutions catalog.
Restricted to type=2 tab. Writes data/catalog.json as UTF-8.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
CATALOG_URL = f"{BASE}/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SHL-Recommender-Intern-Assignment/1.0)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

TEST_TYPE_LEGEND = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def _polite_get(client: httpx.Client, url: str, *, max_retries: int = 5) -> httpx.Response | None:
    for attempt in range(max_retries):
        try:
            r = client.get(url, timeout=30.0)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 503):
                wait = 2 ** attempt
                print(f"  rate-limited ({r.status_code}), sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  HTTP {r.status_code} for {url}", file=sys.stderr)
            return None
        except httpx.RequestError as e:
            wait = 2 ** attempt
            print(f"  network error: {e}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None


def _extract_test_types(cell) -> list[str]:
    letters: list[str] = []
    # Strategy 1: each letter is a single-char span/p
    for el in cell.find_all(["span", "p", "div"]):
        txt = el.get_text(strip=True)
        if len(txt) == 1 and txt.upper() in TEST_TYPE_LEGEND:
            letters.append(txt.upper())
    # Strategy 2: regex over the full cell text
    if not letters:
        full = cell.get_text(" ", strip=True)
        for m in re.finditer(r"\b([ABCDEKPS])\b", full):
            letters.append(m.group(1).upper())
    return sorted(set(letters))


def _parse_listing_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []

    tables = soup.find_all("table")
    target_tables = []
    for tbl in tables:
        header = tbl.find("th")
        header_text = header.get_text(" ", strip=True) if header else ""
        if "Individual Test Solutions" in header_text:
            target_tables.append(tbl)
    if not target_tables:
        target_tables = tables  # fallback

    for tbl in target_tables:
        for tr in tbl.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 4:
                continue
            name_cell, remote_cell, adaptive_cell, types_cell = cells[0], cells[1], cells[2], cells[3]
            a = name_cell.find("a")
            if not a or not a.get("href"):
                continue
            name = a.get_text(" ", strip=True)
            href = a["href"]
            if not name:
                continue
            url = urljoin(BASE, href)

            remote = bool(remote_cell.find(class_=re.compile(r"-?yes|catalogue__circle")))
            adaptive = bool(adaptive_cell.find(class_=re.compile(r"-?yes|catalogue__circle")))
            type_letters = _extract_test_types(types_cell)

            rows.append({
                "name": name,
                "url": url,
                "remote_testing": remote,
                "adaptive_irt": adaptive,
                "test_types": type_letters,
            })
    return rows


def _parse_detail_page(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    detail = {
        "description": "",
        "job_levels": [],
        "languages": [],
        "assessment_length": "",
    }
    for h in soup.find_all(["h4", "h2", "h3"]):
        label = h.get_text(strip=True).lower()
        nxt = h.find_next(["p", "div"])
        if not nxt:
            continue
        text = nxt.get_text(" ", strip=True)
        if "description" in label and not detail["description"]:
            detail["description"] = text
        elif "job level" in label:
            detail["job_levels"] = [s.strip() for s in re.split(r",|·|\|", text) if s.strip()]
        elif "language" in label:
            detail["languages"] = [s.strip() for s in re.split(r",|·|\|", text) if s.strip()]
        elif "assessment length" in label or "completion time" in label:
            detail["assessment_length"] = text
    if not detail["description"]:
        for p in soup.find_all("p"):
            t = p.get_text(" ", strip=True)
            if len(t) > 80:
                detail["description"] = t
                break
    return detail


def scrape_all(client: httpx.Client, *, limit: int | None, max_empty_pages: int = 3) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    page_size = 12
    start = 0
    empty_streak = 0
    while True:
        if limit and len(items) >= limit:
            break
        url = f"{CATALOG_URL}?start={start}&type=2"
        print(f"[listing] start={start} (collected so far: {len(items)})", file=sys.stderr)
        resp = _polite_get(client, url)
        if resp is None:
            print(f"  giving up on start={start}, advancing", file=sys.stderr)
            empty_streak += 1
            if empty_streak >= max_empty_pages:
                print("  too many empty/failed pages in a row; stopping", file=sys.stderr)
                break
            start += page_size
            continue
        rows = _parse_listing_page(resp.text)
        new_rows = [r for r in rows if r["url"] not in seen]
        if not new_rows:
            empty_streak += 1
            print(f"  no new rows ({empty_streak}/{max_empty_pages})", file=sys.stderr)
            if empty_streak >= max_empty_pages:
                break
            start += page_size
            time.sleep(0.4)
            continue
        empty_streak = 0
        for r in new_rows:
            seen.add(r["url"])
            items.append(r)
            if limit and len(items) >= limit:
                break
        start += page_size
        time.sleep(0.4)
    return items


def enrich(client: httpx.Client, row: dict) -> dict:
    resp = _polite_get(client, row["url"], max_retries=3)
    if resp is None:
        row.setdefault("description", "")
        row.setdefault("job_levels", [])
        row.setdefault("languages", [])
        row.setdefault("assessment_length", "")
        return row
    try:
        row.update(_parse_detail_page(resp.text))
    except Exception as e:
        print(f"  detail parse failed for {row['name']!r}: {e}", file=sys.stderr)
        row.setdefault("description", "")
        row.setdefault("job_levels", [])
        row.setdefault("languages", [])
        row.setdefault("assessment_length", "")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/catalog.json")
    ap.add_argument("--skip-detail", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        listing = scrape_all(client, limit=args.limit)
        print(f"Listing collected: {len(listing)} items. Enriching...", file=sys.stderr)
        items = []
        for i, row in enumerate(listing, 1):
            if not args.skip_detail:
                enrich(client, row)
                time.sleep(0.2)
            items.append(row)
            if i % 25 == 0:
                print(f"  enriched {i}/{len(listing)}", file=sys.stderr)

    payload = {
        "source": CATALOG_URL,
        "scope": "Individual Test Solutions",
        "test_type_legend": TEST_TYPE_LEGEND,
        "count": len(items),
        "items": items,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(items)} items to {out_path}")


if __name__ == "__main__":
    main()
