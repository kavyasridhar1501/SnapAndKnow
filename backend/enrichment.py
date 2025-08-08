import re
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

ASIN_PATTERN = re.compile(r"\b([A-Z0-9]{10})\b")
DP_ASIN_PATTERN = re.compile(r"/dp/([A-Z0-9]{10})")

def extract_asin(text: str | None):
    """Return the first ASIN‐like token (10 uppercase letters/digits), or None."""
    if not text:
        return None
    m = ASIN_PATTERN.search(text)
    return m.group(1) if m else None

def scrape_amazon_asin(asin: str) -> dict:
    """Scrape Amazon product page for title & price for a given ASIN."""
    url = f"https://www.amazon.com/dp/{asin}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title_el = soup.select_one("#productTitle")
    # cover a few price selectors that Amazon commonly uses
    price_el = soup.select_one(
        "#corePrice_feature_div span.a-price span.a-offscreen, "
        "#snsBasePrice span.a-offscreen, "
        "#priceblock_ourprice, #priceblock_dealprice, #priceblock_saleprice"
    )

    title = title_el.get_text(strip=True) if title_el else None
    price = price_el.get_text(strip=True) if price_el else None

    return {"asin": asin, "title": title, "price": price}

def find_asin_via_search(query: str | None) -> str | None:
    """
    Try Amazon site search and return the first plausible ASIN from results.
    Uses the 'data-asin' attribute Amazon places on search result items.
    """
    if not query:
        return None
    try:
        params = {"k": query}
        resp = requests.get("https://www.amazon.com/s", headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # primary: cards have data-asin
        for div in soup.select("div.s-result-item[data-asin]"):
            asin = (div.get("data-asin") or "").strip()
            if asin and len(asin) == 10:
                return asin

        # fallback: first link with /dp/ASIN
        a = soup.select_one("a.a-link-normal[href*='/dp/']")
        if a and a.has_attr("href"):
            m = DP_ASIN_PATTERN.search(a["href"])
            if m:
                return m.group(1)
    except Exception as e:
        print(f"[enrichment] find_asin_via_search error: {e}")
    return None

def enrich_from_free_text(*texts: str) -> dict:
    """
    Best-effort enrichment from arbitrary text:
      1) regex ASIN
      2) Amazon search for ASIN using combined text
      3) if ASIN found → scrape title/price
    """
    joined = " ".join([t for t in texts if t])
    asin = extract_asin(joined)
    if not asin:
        asin = find_asin_via_search(joined)

    if asin:
        try:
            meta = scrape_amazon_asin(asin) or {}
            meta.setdefault("asin", asin)
            return meta
        except Exception as e:
            print(f"[enrichment] enrich_from_free_text scrape error: {e}")
    return {}
