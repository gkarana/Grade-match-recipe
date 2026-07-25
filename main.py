"""Gemini image-editing proxy for GradeMatch AI mode.

Keeps the API key server-side (env GEMINI_API_KEY) with an optional
per-request key fallback. Raises GeminiError with user-readable messages.
"""

import base64
import json
import os

import httpx

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = (
    "You are a world-class photo colorist and lighting director.\n"
    "STUDY image 1 (STYLE REFERENCE) and analyze it precisely:\n"
    "(a) LIGHTING - direction and quality of light (soft/hard), the light "
    "ratio, how much light falls on the subject's face versus the background, "
    "catchlights in eyes, depth and direction of shadows, rim/back light;\n"
    "(b) GRADE - palette, white balance, tone curve, contrast, saturation, "
    "highlight and shadow tinting;\n"
    "(c) ATMOSPHERE - haze, glow, vignette, grain.\n"
    "EDIT image 2 (THE PHOTO) with pixel-level fidelity: keep the subject's "
    "face geometry, expression, skin texture and pores, hair strands, fabric "
    "detail, every background object, and the exact composition. Do NOT "
    "regenerate, beautify, slim, smooth, or reposition the subject. Do NOT "
    "copy any object or scenery from image 1.\n"
    "APPLY image 1's style onto image 2: match its lighting mood - including "
    "how much light lands on the face and how shadows fall - its color grade, "
    "its tonal curve, its local contrast pattern (the dodge-and-burn feel), "
    "and its atmosphere. The result must look like the SAME photograph taken "
    "by the SAME camera, then lit and graded by the reference's colorist. "
    "{clarity}Output only the edited photograph."
)
CLARITY_CLAUSE = (
    "Also enhance clarity, texture and fine detail subtly, like a professional "
    "Lightroom clarity adjustment. "
)


class GeminiError(Exception):
    pass


# ---------------------------------------------------------- style analyst (recipe mode)
RECIPE_SCHEMA = """{
  "style_name": "short name, max 4 words",
  "wb_temp": -1.0 to 1.0 (negative=cooler, positive=warmer),
  "exposure": -2.0 to 2.0 (stops),
  "contrast": -1.0 to 1.0,
  "blacks": -1.0 to 1.0 (negative=crushed, positive=lifted/faded),
  "whites": -1.0 to 1.0 (negative=muted, positive=bright),
  "tone_curve": [[x,y],...5-7 points, x ascending, both 0..1, output luma per input luma],
  "saturation": -1.0 to 1.0,
  "vibrance": -1.0 to 1.0,
  "hsl": {"red":[h,s,l], "orange":[h,s,l], "yellow":[h,s,l], "green":[h,s,l],
          "aqua":[h,s,l], "blue":[h,s,l], "purple":[h,s,l], "magenta":[h,s,l]},
  "clarity": 0.0 to 1.0,
  "sharpen": 0.0 to 1.0,
  "vignette": -1.0 to 1.0 (positive=darkened corners),
  "grain": 0.0 to 1.0
}"""

RECIPE_PROMPT = (
    "You are a world-class photo colorist. Analyze the editing style of this "
    "photograph and express it as precise edit instructions that would recreate "
    "the SAME look on a completely different photo.\n"
    "Read the evidence carefully: overall warmth or coolness; exposure level; "
    "contrast; whether blacks are crushed or lifted (matte fade); highlight "
    "brightness; the shape of the tone curve; saturation vs muted palette; "
    "vibrance; per-hue shifts (e.g. teal shadows, orange skin, desaturated "
    "greens, cyan skies); texture/clarity level; vignette; grain.\n"
    "Respond with ONLY valid JSON in exactly this schema (0 = neutral; every "
    "number based on visible evidence; hsl entries are [hue, saturation, "
    "luminance] deltas each -1..1):\n" + RECIPE_SCHEMA
)

_NUM_FIELDS = {"wb_temp": (-1, 1), "exposure": (-2, 2), "contrast": (-1, 1),
               "blacks": (-1, 1), "whites": (-1, 1), "saturation": (-1, 1),
               "vibrance": (-1, 1), "clarity": (0, 1), "sharpen": (0, 1),
               "vignette": (-1, 1), "grain": (0, 1)}


def _validate_recipe(raw):
    """Coerce the model's JSON into a safe, clamped recipe dict."""
    if not isinstance(raw, dict):
        raise GeminiError("Style analysis returned no JSON object.")
    recipe = {"style_name": str(raw.get("style_name", "custom style"))[:60]}
    for k, (lo, hi) in _NUM_FIELDS.items():
        try:
            recipe[k] = max(lo, min(hi, float(raw.get(k, 0) or 0)))
        except (TypeError, ValueError):
            recipe[k] = 0.0
    pts = raw.get("tone_curve")
    if isinstance(pts, list):
        clean = []
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                try:
                    clean.append([float(p[0]), float(p[1])])
                except (TypeError, ValueError):
                    pass
        recipe["tone_curve"] = sorted(clean)[:8]
    hsl = raw.get("hsl")
    recipe["hsl"] = {}
    if isinstance(hsl, dict):
        for name, val in hsl.items():
            if isinstance(val, (list, tuple)) and len(val) == 3:
                try:
                    recipe["hsl"][name] = [max(-1, min(1, float(x))) for x in val]
                except (TypeError, ValueError):
                    pass
    return recipe


async def analyze_style(ref_jpeg: bytes, key: str = "",
                        model: str = "gemini-2.5-flash") -> dict:
    """Vision model -> structured edit recipe. Uses a TEXT model, so it works
    on free-tier keys (no image generation quota needed)."""
    api_key = os.environ.get("GEMINI_API_KEY") or key
    if not api_key:
        raise GeminiError(
            "No API key available. Set GEMINI_API_KEY on the server, "
            "or paste a key in the AI panel."
        )
    body = {
        "contents": [{
            "parts": [
                {"text": RECIPE_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(ref_jpeg).decode()}},
            ]
        }],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
    }
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            res = await client.post(API.format(model=model),
                                    headers={"x-goog-api-key": api_key},
                                    json=body)
    except httpx.HTTPError as e:
        raise GeminiError(f"Could not reach Google: {type(e).__name__} {e}".strip()) from e
    data = res.json() if res.content else {}
    if res.status_code != 200:
        raw = (data.get("error") or {}).get("message", f"HTTP {res.status_code}")
        if any(w in raw.lower() for w in ("quota", "billing", "limit: 0")):
            raise GeminiError(
                "This key has no quota for the analysis model either - check "
                f"your plan at ai.dev/rate-limit. [{raw[:140]}]"
            )
        raise GeminiError(raw)
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise GeminiError("The model returned no analysis. Try a different reference.")
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise GeminiError(f"Could not parse the style recipe: {e}") from e
    return _validate_recipe(parsed)


async def style_match(ref_jpeg: bytes, tgt_jpeg: bytes, model: str,
                      clarity_boost: bool, key: str = "",
                      image_size: str = "1K") -> bytes:
    api_key = os.environ.get("GEMINI_API_KEY") or key
    if not api_key:
        raise GeminiError(
            "No API key available. Set GEMINI_API_KEY on the server, "
            "or paste a key in the AI panel."
        )

    body = {
        "contents": [{
            "parts": [
                {"text": PROMPT.format(clarity=CLARITY_CLAUSE if clarity_boost else "")},
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(ref_jpeg).decode()}},
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(tgt_jpeg).decode()}},
            ]
        }],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    if image_size and image_size != "1K":            # 2K/4K on supported models
        body["generationConfig"]["imageConfig"] = {"imageSize": image_size}

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            res = await client.post(API.format(model=model),
                                    headers={"x-goog-api-key": api_key},
                                    json=body)
    except httpx.HTTPError as e:
        raise GeminiError(f"Could not reach Google: {type(e).__name__} {e}".strip()) from e

    data = res.json() if res.content else {}
    if res.status_code != 200:
        raw = (data.get("error") or {}).get("message", f"HTTP {res.status_code}")
        if any(w in raw.lower() for w in ("quota", "billing", "limit: 0")):
            raise GeminiError(
                "This key has no image quota - Google's free tier does not cover "
                "image models. Enable billing at console.cloud.google.com/billing "
                "(pay-as-you-go, roughly $0.04-0.13/image), or use the free Gemini "
                f"app instead. [{raw[:140]}]"
            )
        raise GeminiError(raw)

    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    for part in parts:
        inline = part.get("inline_data") or part.get("inlineData")
        if inline and inline.get("data"):
            return base64.b64decode(inline["data"])
    raise GeminiError(
        "The model returned no image - it may have refused the content. "
        "Try different photos."
    )
