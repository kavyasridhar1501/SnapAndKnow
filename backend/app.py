import os
import re
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
from flask import Flask, request, jsonify, session, send_from_directory
from PIL import Image

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
load_dotenv()

FORCE_AGENT_FOR_IMAGE = os.environ.get("FORCE_AGENT_FOR_IMAGE", "false").lower() in {"1", "true", "yes"}

import image_pipeline 
from agent import agent, rag_answer
from enrichment import extract_asin, scrape_amazon_asin, find_asin_via_search, enrich_from_free_text

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), os.pardir, "frontend"),
    static_url_path=""
)
secret_key = os.environ.get("FLASK_SECRET_KEY")
if not secret_key:
    raise RuntimeError("Missing FLASK_SECRET_KEY environment variable")
app.secret_key = secret_key

@app.route("/", methods=["GET"])
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")


IMAGE_QUESTION_KEYWORDS = (
    "what is this", "what’s this", "what is it", "identify",
    "brand", "maker", "make", "manufacturer", "model", "name",
    "color", "colour",
)
REFERS_TO_IMAGE_TERMS = ("this", "this product", "this item", "this one")

OPINION_QUESTION_KEYWORDS = (
    "what do people think", "people think", "reviews", "review",
    "rating", "ratings", "feedback", "worth it", "worth buying",
    "recommend", "pros and cons", "good or bad", "overall opinion",
)

BUY_VOLUME_KEYWORDS = (
    "how many people have bought", "how many bought", "units sold",
    "how many sold", "sold in the last year", "sales in the last year",
)

PRICE_QUESTION_KEYWORDS = (
    "price", "cost", "how much", "current price", "what is the price",
    "how much is this", "how much is it"
)

def _contains_term(text: str, term: str) -> bool:
    text = (text or "").lower()
    term = term.lower()
    if " " in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text) is not None

def _any(text: str, terms) -> bool:
    return any(_contains_term(text, t) for t in terms)

def _is_image_question(q: str) -> bool:
    return _any(q, IMAGE_QUESTION_KEYWORDS)

def _refers_to_uploaded_image(q: str) -> bool:
    return _any(q, REFERS_TO_IMAGE_TERMS)

def _is_opinion_question(q: str) -> bool:
    return _any(q, OPINION_QUESTION_KEYWORDS)

def _asks_buy_volume(q: str) -> bool:
    return _any(q, BUY_VOLUME_KEYWORDS)

def _is_price_question(q: str) -> bool:
    return _any(q, PRICE_QUESTION_KEYWORDS)


def _maybe_enrich_from_strings(*texts) -> Tuple[str, Dict[str, Any]]:
    """
    Enhanced enrichment:
      - regex ASIN
      - else Amazon search to find ASIN
      - then scrape title/price
      - caches last_asin
    """
    joined = " ".join([t for t in texts if t])
    meta = enrich_from_free_text(joined)  
    if meta.get("asin"):
        session["last_asin"] = meta["asin"]

    parts = []
    if meta.get("asin"): parts.append(f"asin: {meta['asin']}")
    if meta.get("title"): parts.append(f"title: {meta['title']}")
    if meta.get("price"): parts.append(f"price: {meta['price']}")
    ctx = f"Context from product page: {'; '.join(parts)}. " if parts else ""
    if ctx:
        print(f"[enrichment] {ctx}")
    return ctx, meta

def _price_line(meta: Dict[str, Any]) -> str | None:
    price = meta.get("price")
    if not price:
        return None
    title = meta.get("title") or "this item"
    asin = meta.get("asin")
    suffix = f" (ASIN {asin})" if asin else ""
    return f"The current listed price on Amazon is {price} for {title}{suffix}. Prices change frequently."

def _build_seed_from_image(pil_img: Image.Image) -> Dict[str, Any]:
    out = {"brand": None, "caption": None, "color": None, "seed_text": None}
    brand = image_pipeline.detect_brand_via_ocr(pil_img)
    caption = image_pipeline.image_blurb(pil_img, "") or ""
    out["brand"] = brand
    out["caption"] = caption
    out["seed_text"] = " ".join((f"{brand} {caption}".strip() if brand else (caption or "this product")).split()[:20])
    return out


def _gather_signals(user_q: str, last_img: Image.Image | None) -> Dict[str, Any]:
    signals: Dict[str, Any] = {
        "user_q": user_q,
        "has_image": last_img is not None,
        "image": {},
        "enrichment": {"ctx": "", "meta": {}},
        "rag": "",
        "agent": "",
    }

    if last_img is not None and (_is_image_question(user_q) or _refers_to_uploaded_image(user_q) or FORCE_AGENT_FOR_IMAGE):
        img = _build_seed_from_image(last_img)
        signals["image"] = img

        if _contains_term(user_q, "color") or _contains_term(user_q, "colour"):
            try:
                img["color"] = image_pipeline.get_dominant_color(last_img)
            except Exception as e:
                print(f"[image] color error: {e}")

        enrich_ctx, meta = _maybe_enrich_from_strings(user_q, img.get("seed_text"))
        signals["enrichment"] = {"ctx": enrich_ctx, "meta": meta}

        try:
            rag_prompt = (
                f"{enrich_ctx}"
                f"User question: {user_q}\n"
                f"Image hints: brand={img.get('brand')}, caption=\"{img.get('caption')}\".\n"
                "Answer succinctly and include sentiment from reviews if relevant."
            )
            signals["rag"] = rag_answer(rag_prompt)
        except Exception as e:
            print(f"[rag] error: {e}")

        try:
            agent_q = (enrich_ctx + user_q) if enrich_ctx else user_q
            agent_out = agent.invoke({"input": agent_q})
            signals["agent"] = (
                (agent_out or {}).get("output")
                or (agent_out or {}).get("text")
                or str(agent_out)
                or ""
            ).strip()
            session["last_agent_text"] = signals["agent"]
        except Exception as e:
            print(f"[agent] error: {e}")

    else:
        enrich_ctx, meta = _maybe_enrich_from_strings(user_q)
        signals["enrichment"] = {"ctx": enrich_ctx, "meta": meta}

        try:
            signals["rag"] = rag_answer(f"{enrich_ctx}User question: {user_q}\nUse reviews to answer reliably and concisely.")
        except Exception as e:
            print(f"[rag] error: {e}")

        try:
            agent_out = agent.invoke({"input": (enrich_ctx + user_q) if enrich_ctx else user_q})
            signals["agent"] = (
                (agent_out or {}).get("output")
                or (agent_out or {}).get("text")
                or str(agent_out)
                or ""
            ).strip()
            session["last_agent_text"] = signals["agent"]
        except Exception as e:
            print(f"[agent] error: {e}")

    return signals

def _compose_answer(signals: Dict[str, Any]) -> str:
    q = signals["user_q"]
    img = signals.get("image") or {}
    meta = signals.get("enrichment", {}).get("meta") or {}
    enrich_ctx = signals.get("enrichment", {}).get("ctx") or ""
    rag_txt = (signals.get("rag") or "").strip()
    agent_txt = (signals.get("agent") or "").strip()

    if _is_price_question(q):
        price_line = _price_line(meta)
        if price_line:
            return price_line

        retry_meta = enrich_from_free_text(" ".join([
            agent_txt, img.get("seed_text") or "", rag_txt
        ]))
        price_line = _price_line(retry_meta)
        if price_line:
            return price_line

        retry_meta2 = enrich_from_free_text(" ".join([img.get("seed_text") or "", q]))
        price_line = _price_line(retry_meta2)
        if price_line:
            return price_line

        return ("I couldn’t fetch a live price for this item right now. "
                "Prices vary by seller and options, but reviews suggest it’s reasonably priced.")

    if _any(q, ("brand", "maker", "manufacturer", "make", "name", "model", "identify")):
        if img.get("brand"):
            return f"The visible brand appears to be {img['brand']}."
        guess = _extract_brand_like(agent_txt) or _extract_brand_like(rag_txt)
        if guess:
            return f"It looks like the brand might be {guess}."
        return "I can’t read a visible brand from this image."

    if _contains_term(q, "color") or _contains_term(q, "colour"):
        if img.get("color"):
            return img["color"]
        return "I can’t confidently determine a single dominant color."

    if _any(q, ("what is this", "what’s this", "what is it", "identify")) and signals.get("has_image"):
        if img.get("brand") and img.get("caption"):
            return f"{img['brand']} — {img['caption']}"
        return img.get("caption") or "Looks like a product image."

    if _asks_buy_volume(q):
        base = ("Exact sales numbers aren’t public. "
                "Based on reviews and public signals, here’s the popularity snapshot:")
        return f"{base}\n\n{_tidy(rag_txt) or 'Reviews suggest it’s widely purchased and well-reviewed.'}"


    lines = []
    if meta.get("title"):
        lines.append(f"**Product**: {meta['title']}")
    if meta.get("price"):
        lines.append(f"**Price (Amazon)**: {meta['price']}")
    if rag_txt:
        lines.append(_tidy(rag_txt))
    addendum = _short_addendum(rag_txt, agent_txt)
    if addendum:
        lines.append(addendum)
    return "\n\n".join(lines) or "I couldn’t find a clear answer. Try rephrasing."

def _extract_brand_like(text: str) -> str | None:
    if not text:
        return None
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)
    candidates = [t for t in tokens if (t.isupper() or t.istitle())]
    if not candidates:
        return None
    return sorted(candidates, key=len, reverse=True)[0].title()

def _tidy(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()

def _short_addendum(primary: str, extra: str) -> str | None:
    if not extra:
        return None
    if len(extra) > 20 and extra[:60] not in primary:
        return f"_Note_: {extra}"
    return None


@app.route("/upload_and_query", methods=["POST"])
def upload_and_query():
    user_q = (request.form.get("query") or "").strip()

    img_file = request.files.get("image")
    if img_file:
        try:
            image_pipeline.last_image = Image.open(img_file.stream).convert("RGB")
            session["has_image"] = True
            print("[image-route] Stored latest uploaded image.")
        except Exception as e:
            image_pipeline.last_image = None
            session["has_image"] = False
            print(f"[image-route] Failed to load uploaded image: {e}")

    try:
        last_img = image_pipeline.last_image
        signals = _gather_signals(user_q, last_img if not FORCE_AGENT_FOR_IMAGE else None)
        answer = _compose_answer(signals)
    except Exception as e:
        print(f"[route-error] {e}")
        try:
            if image_pipeline.last_image is not None:
                answer = image_pipeline.image_blurb(image_pipeline.last_image, "")
            else:
                answer = "Sorry, I hit an error. Please try again."
        except Exception:
            answer = "Sorry, I hit an error. Please try again."

    return jsonify({
        "ok": True,
        "answer": answer,
        "message": answer,
        "response": answer,
        "text": answer,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=(os.environ.get("FLASK_ENV") != "production"), use_reloader=False)
