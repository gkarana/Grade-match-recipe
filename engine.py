"""Style-analyst backends for recipe mode.

Winner of the 5-candidate shootout (120 trials, 8 style archetypes,
3 downscale sizes, judged on dE2000 / SSIM / histogram intersection /
parameter distance vs ground truth):

    1. kimi-latest                      (agent-gw, server-side, no key needed)
    2. kimi-k2.5                        (agent-gw)
    3. kimi-for-coding                  (agent-gw, fastest LLM ~60s)
    4. moonshot-v1-8k-vision-preview    (agent-gw)
    5. PIXELSTAT deterministic baseline (last, but instant)

Gemini BYOK remains as fallback for deployments without agent-gw access.
Downscale finding: 256/448/896 px inputs are statistically indistinguishable,
so ANALYSIS_SIDE stays small - big uploads buy nothing.
"""
import asyncio
import base64
import json
import re

import gemini

GW_MODEL = "kimi-latest"          # measured best overall (composite rank 1.0)
GW_MODEL_FAST = "kimi-for-coding"  # fastest LLM analyst (~60s)


def _parse_recipe(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(json)?|```$", "", t, flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise gemini.GeminiError("Analyst returned no JSON recipe.")
    try:
        return gemini._validate_recipe(json.loads(t[i:j + 1]))
    except json.JSONDecodeError as e:
        raise gemini.GeminiError(f"Could not parse the style recipe: {e}") from e


def _gw_analyze_sync(ref_jpeg: bytes, model: str) -> dict:
    from agent_gw import AgentGwClient  # optional dependency
    client = AgentGwClient()
    b64 = base64.b64encode(ref_jpeg).decode()
    r = client.chat_completion(model=model, messages=[{"role": "user", "content": [
        {"type": "text", "text": gemini.RECIPE_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
        timeout=180)
    text = r["choices"][0]["message"]["content"]
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    recipe = _parse_recipe(text)
    recipe["analyst"] = model
    return recipe


async def analyze(ref_jpeg: bytes, key: str = "") -> dict:
    """Try the server-side agent-gw analyst (measured best, free of user key);
    fall back to Gemini BYOK text models."""
    try:
        return await asyncio.to_thread(_gw_analyze_sync, ref_jpeg, GW_MODEL)
    except Exception as gw_err:                       # noqa: BLE001 - any failure -> BYOK
        if not key:
            raise gemini.GeminiError(
                "Server analyst unavailable "
                f"({type(gw_err).__name__}) and no API key pasted - "
                "add a free Gemini key in the AI panel."
            ) from gw_err
        recipe = await gemini.analyze_style(ref_jpeg, key=key)
        recipe["analyst"] = "gemini-2.5-flash (BYOK)"
        return recipe
