from __future__ import annotations
import os
import re
from typing import Optional, Tuple, List
from PIL import Image


last_image: Optional[Image.Image] = None
_caption_pipe = None
_caption_err: Optional[Exception] = None


def _load_captioner() -> None:
    global _caption_pipe, _caption_err
    if _caption_pipe is not None or _caption_err is not None:
        return
    try:
        from transformers import pipeline
        model_id = os.getenv("IMAGE_CAPTION_MODEL", "Salesforce/blip-image-captioning-large")
        _caption_pipe = pipeline("image-to-text", model=model_id)
    except Exception as e:  
        _caption_err = e


def image_blurb(pil_img: Image.Image, prompt: str = "") -> str:
    _load_captioner()
    if _caption_err is not None:
        return "Image captioning is unavailable on this server."
    if _caption_pipe is None:
        return "Image captioner did not initialize."

    try:
        out = _caption_pipe(pil_img)
        if isinstance(out, list) and out:
            cap = (out[0].get("generated_text") or "").strip()
            return cap or "I see a product image."
        return "I see a product image."
    except Exception:
        return "Sorry, I couldn’t analyze the image."


def _closest_css3_name(rgb: Tuple[int, int, int]) -> str:
    try:
        import webcolors  
        try:
            return webcolors.rgb_to_name(rgb, spec="css3")
        except ValueError:
            best_name = None
            best_dist = 10**9
            for name, hexv in webcolors.CSS3_NAMES_TO_HEX.items():
                r2, g2, b2 = webcolors.hex_to_rgb(hexv)
                dist = (rgb[0] - r2) ** 2 + (rgb[1] - g2) ** 2 + (rgb[2] - b2) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
            return best_name or f"rgb{rgb}"
    except Exception:
        return "#{:02x}{:02x}{:02x}".format(*rgb)


def get_dominant_color(pil_img: Image.Image) -> str:
    try:
        img = pil_img.convert("RGB").resize((64, 64))
        pal = img.convert("P", palette=Image.ADAPTIVE, colors=5)
        colors = pal.getcolors()
        if not colors:
            return "Unknown color"
        palette = pal.getpalette()
        colors.sort(reverse=True)  
        idx = colors[0][1]
        r, g, b = palette[idx * 3 : idx * 3 + 3]
        return _closest_css3_name((r, g, b))
    except Exception:
        return "Unknown color"


try:
    import pytesseract  
except Exception:
    pytesseract = None  

_BRANDS: List[str] = [
    "revlon", "dyson", "babyliss", "hot tools", "conair", "philips", "panasonic",
    "remington", "ghd", "chi", "andis", "wahl", "nioxin", "olaplex", "kindle",
    "amazon", "apple", "samsung", "sony", "logitech", "anker", "hp", "dell",
    "lenovo", "asus", "acer", "nintendo",
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 +\-]", " ", text.lower())


def _prep_variants(img: Image.Image):
    
    base = img.convert("RGB")
    w, h = base.size

    crops = [
        ("full", base),
        ("bottom_right", base.crop((int(w * 0.55), int(h * 0.55), w, h))),
        ("bottom_left", base.crop((0, int(h * 0.55), int(w * 0.45), h))),
        ("lower_center", base.crop((int(w * 0.25), int(h * 0.55), int(w * 0.75), h))),
        ("right_center", base.crop((int(w * 0.6), int(h * 0.25), w, int(h * 0.75)))),
    ]

    variants = []
    for name, im in crops:
        im2 = im.resize((max(1, im.width * 2), max(1, im.height * 2)), Image.LANCZOS)
        gray = im2.convert("L")
        variants.append((f"{name}_gray", gray))
        for thr in (140, 160, 200):
            bw = gray.point(lambda p, t=thr: 255 if p > t else 0)
            variants.append((f"{name}_thr{thr}", bw))
    return variants


def _ocr_tesseract(img: Image.Image) -> str:
    if pytesseract is None:
        return ""

    texts: List[str] = []
    configs = [
        "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        "--oem 3 --psm 11",
        "--oem 3 --psm 13",
    ]

    for tag, variant in _prep_variants(img):
        for cfg in configs:
            try:
                txt = pytesseract.image_to_string(variant, lang="eng", config=cfg)
                if txt:
                    texts.append(txt)
                    if os.getenv("OCR_DEBUG") == "1":
                        print(f"[ocr:{tag}] {cfg} -> {repr(txt)}")
            except Exception:
                pass

    return "\n".join(texts)


def detect_brand_via_ocr(pil_img: Image.Image) -> Optional[str]:
    raw = _ocr_tesseract(pil_img)
    if not raw:
        return None

    norm = _normalize(raw)

    for b in sorted(_BRANDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(b)}\b", norm):
            return b.title()

    tokens = [t.strip(".,:;!?()[]{}|/\\\"'") for t in raw.split()]
    uppers = [t for t in tokens if len(t) >= 3 and t.isupper()]
    if uppers:
        return max(uppers, key=len).title()
    caps = [t for t in tokens if len(t) >= 3 and t[0].isupper()]
    if caps:
        return max(caps, key=len).title()

    return None
