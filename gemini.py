"""GradeMatch engine - deterministic, pixel-preserving photo grade transfer.

Pipeline (mirrors the original JS prototype, vectorized with NumPy):
  1. sRGB -> Lab conversion (D65)
  2. Tone: 1D histogram matching on the L channel (tone curve)
  3. Color: histogram matching on a/b channels, optionally split into
     shadow / midtone / highlight bands (terciles of target luminance)
  4. Intensity: per-pixel lerp between original and matched values
  5. Clarity: midtone-masked local contrast (large-radius box blur)
     Sharpen: unsharp mask (small-radius box blur), luma-ratio scaled

Public API:
    grade_pair(ref_small, tgt_small, tgt_full, ...) -> uint8 RGB array
"""

import numpy as np
from scipy.ndimage import uniform_filter

# Coordinate grids for the light/vibe maps are expensive at full res and are
# identical for every image of the same shape — cache the last few.
_COORD_CACHE = {}
_COORD_CACHE_MAX = 3


def _coords(h, w):
    """Normalized radius map + centered x/y fields, cached per image shape."""
    key = (h, w)
    if key not in _COORD_CACHE:
        yy, xx = np.mgrid[0:h, 0:w]
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)
        r = np.sqrt(((xx - (w - 1) / 2) / (w / 2)) ** 2
                    + ((yy - (h - 1) / 2) / (h / 2)) ** 2) / np.sqrt(2)
        _COORD_CACHE[key] = (r.astype(np.float32), xx / w - 0.5, yy / h - 0.5)
        if len(_COORD_CACHE) > _COORD_CACHE_MAX:
            _COORD_CACHE.pop(next(iter(_COORD_CACHE)))
    return _COORD_CACHE[key]

_XN, _YN, _ZN = 0.95047, 1.0, 1.08883
_D = 6.0 / 29.0


# ---------------------------------------------------------------- color spaces
def _srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)


# sRGB<->linear as lookup tables: input bytes have only 256 possible values,
# and a 4096-entry gamma LUT beats tens of millions of pow() calls on output.
_LIN_LUT = _srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)
_GAM_LUT = (_linear_to_srgb(np.linspace(0, 1, 4096, dtype=np.float32)) * 255 + 0.5).astype(np.uint8)


def rgb_to_lab(rgb_u8):
    """uint8 (H,W,3) -> (L, a, b) float32 arrays. L in 0..100, a/b roughly +-128."""
    rgb = _LIN_LUT[rgb_u8]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    def f(t):
        return np.where(t > _D ** 3, np.cbrt(t), t / (3 * _D * _D) + 4.0 / 29.0)

    fx, fy, fz = f(X / _XN), f(Y / _YN), f(Z / _ZN)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def lab_to_rgb(L, a, b):
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def fi(t):
        t3 = t ** 3
        return np.where(t3 > _D ** 3, t3, 3 * _D * _D * (t - 4.0 / 29.0))

    X, Y, Z = _XN * fi(fx), _YN * fi(fy), _ZN * fi(fz)
    r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    bb = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    rgb = np.stack([r, g, bb], axis=-1)
    idx = np.clip((rgb * 4095).astype(np.int32), 0, 4095)
    return _GAM_LUT[idx]


# ---------------------------------------------------------------- histogram matching
def _hist_lut(src_vals, ref_vals, vmin, vmax, bins=256):
    """Monotone CDF-matching LUT: bin index -> matched value.

    Only bins that actually contain source pixels are matched directly;
    the rest are filled by monotone interpolation, so sparse or narrow
    histograms don't develop cliff artifacts at the edges of the support.
    """
    hs, _ = np.histogram(src_vals, bins=bins, range=(vmin, vmax))
    hr, _ = np.histogram(ref_vals, bins=bins, range=(vmin, vmax))
    cs = np.cumsum(hs).astype(np.float64)
    cr = np.cumsum(hr).astype(np.float64)
    if cs[-1] == 0 or cr[-1] == 0:
        return np.linspace(vmin, vmax, bins, dtype=np.float32)
    cs /= cs[-1]
    cr /= cr[-1]
    occ = np.flatnonzero(hs)                                # occupied source bins
    idx = np.clip(np.searchsorted(cr, cs[occ]), 0, bins - 1)
    mapped = vmin + (idx + 0.5) / bins * (vmax - vmin)
    lut = np.interp(np.arange(bins), occ, mapped).astype(np.float64)
    if len(occ) > 4:                                        # smooth only within support
        lo, hi = occ[0], occ[-1]
        seg = lut[lo:hi + 1]
        for _ in range(2):
            seg = np.convolve(np.pad(seg, (1, 1), mode="edge"), [0.25, 0.5, 0.25], mode="valid")
        lut[lo:hi + 1] = seg
    return lut.astype(np.float32)


def _apply_lut(vals, lut, vmin, vmax):
    bins = lut.shape[0]
    ix = ((vals - vmin) / (vmax - vmin) * bins).astype(np.int32)
    np.clip(ix, 0, bins - 1, out=ix)
    return lut[ix]


# ---------------------------------------------------------------- detail (clarity / sharpen)
def _scale_by_ratio(f, Y, Y_new):
    ratio = np.where(Y > 4, Y_new / np.maximum(Y, 1e-6), 1.0)
    return f * ratio[..., None]


def _luma(f):
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def detail_boost(rgb_u8, clarity=0, sharpen=0):
    """Multi-scale clarity: big-radius midtone contrast (the Lightroom 'clarity'
    look) + a medium-radius 'texture' term for fine local punch + unsharp sharpen."""
    f = rgb_u8.astype(np.float32)
    Y = _luma(f)
    mn = min(Y.shape)
    if clarity > 0:
        midw = 1 - np.abs(Y - 128) / 128
        r_big = max(2, round(mn / 120))
        bl = uniform_filter(Y, size=2 * r_big + 1, mode="nearest")
        d = (Y - bl) * (clarity / 100 * 0.8) * midw
        f = _scale_by_ratio(f, Y, Y + d)
        Y = _luma(f)
        r_mid = max(2, round(mn / 320))                  # texture band
        bl2 = uniform_filter(Y, size=2 * r_mid + 1, mode="nearest")
        d2 = (Y - bl2) * (clarity / 100 * 0.45) * midw
        f = _scale_by_ratio(f, Y, Y + d2)
        Y = _luma(f)
    if sharpen > 0:
        r = max(1, round(mn / 1200))
        bl = uniform_filter(Y, size=2 * r + 1, mode="nearest")
        d = (Y - bl) * (sharpen / 100 * 0.6)
        f = _scale_by_ratio(f, Y, Y + d)
    return np.clip(f + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- grain (granularity)
def estimate_grain(ref_u8):
    """Measure the reference's grain: robust MAD of the high-frequency luma
    residual, per luminance tercile (MAD ignores edges, so no flat-area mask)."""
    f = ref_u8.astype(np.float32)
    Y = _luma(f)
    resid = np.abs(Y - uniform_filter(Y, size=5, mode="nearest"))
    t1, t2 = (float(v) for v in np.percentile(Y, [33.333, 66.667]))
    sigma = []
    for lo, hi in ((-1e9, t1), (t1, t2), (t2, 1e9)):
        m = (Y >= lo) & (Y < hi)
        sigma.append(float(1.4826 * np.median(resid[m])) if m.sum() > 500 else 0.0)
    return {"sigma": sigma, "t1": t1, "t2": t2}


def apply_grain(rgb_u8, params, amount, seed=0):
    """Synthesize filmic grain matched to the reference's measured granularity."""
    if amount <= 0:
        return rgb_u8
    sigma = np.asarray(params["sigma"], np.float32) * (amount / 100.0)
    if sigma.max() <= 0.05:
        return rgb_u8
    f = rgb_u8.astype(np.float32)
    Y = _luma(f)
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(Y.shape).astype(np.float32)
    n = uniform_filter(n, size=3, mode="nearest")        # slight correlation = filmic, not digital
    n /= max(float(n.std()), 1e-6)
    band = np.digitize(Y, [params["t1"], params["t2"]])
    n = n * sigma[band]
    return np.clip(f + n[..., None] + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- vibe / light
def estimate_light(ref_u8, rings=8):
    """Capture the reference's 'light vibe': radial vignette profile + a
    linear brightness gradient (which side of the frame the light comes from)."""
    f = ref_u8.astype(np.float32)
    Y = _luma(f)
    h, w = Y.shape
    r, xn, yn = _coords(h, w)
    prof = []
    for i in range(rings):
        m = (r >= i / rings) & (r < (i + 1) / rings)
        prof.append(float(Y[m].mean()) if m.any() else float(Y.mean()))
    prof = np.asarray(prof, np.float32)
    center = max(float(prof[:2].mean()), 1e-3)
    vig = np.clip(prof / center, 0.55, 1.15)
    ys, xs = np.mgrid[0:h:8, 0:w:8]
    A = np.stack([xs.ravel() / w, ys.ravel() / h, np.ones(xs.size, np.float32)], 1)
    coef, *_ = np.linalg.lstsq(A, Y[::8, ::8].ravel(), rcond=None)
    return {"vig": vig.tolist(), "grad": [float(coef[0]), float(coef[1])],
            "mean": float(Y.mean())}


def _light_map(params, h, w):
    """Render a smooth light-shape map (vignette x gradient) at a given size."""
    r, xn, yn = _coords(h, w)
    vig = np.asarray(params["vig"], np.float32)
    idx = np.clip((r * len(vig)).astype(np.int32), 0, len(vig) - 1)
    light = vig[idx]
    gx, gy = params["grad"]
    plane = (gx * xn + gy * yn) / max(params["mean"], 1e-3)
    return light * (1.0 + np.clip(plane, -0.35, 0.35))


def apply_light(rgb_u8, ref_params, tgt_params, amount):
    """Relight: divide out the target's own light shape, impose the reference's.
    Exposure-neutral; identity when reference and target share the same vibe."""
    if amount <= 0:
        return rgb_u8
    f = rgb_u8.astype(np.float32)
    h, w = f.shape[:2]
    rel = _light_map(ref_params, h, w) / np.maximum(_light_map(tgt_params, h, w), 0.2)
    rel = np.clip(rel, 0.55, 1.8)
    rel /= max(float(rel.mean()), 1e-3)                  # keep overall exposure
    amt = amount / 100.0
    Y = _luma(f)
    out = _scale_by_ratio(f, Y, Y * (1.0 + (rel - 1.0) * amt))
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- warmth / smoother / glow
def apply_warmth(rgb_u8, warmth):
    """Golden-light warmth: strong warm gains + a gold tint weighted into the
    mid/high tones (where 'light' lives). Original luma restored both times."""
    if warmth <= 0:
        return rgb_u8
    w = warmth / 100.0
    f = rgb_u8.astype(np.float32)
    Y = _luma(f)
    gains = np.array([1.0 + 0.20 * w, 1.0 + 0.03 * w, 1.0 - 0.24 * w], np.float32)
    out = _scale_by_ratio(f * gains, _luma(f * gains), Y)
    # golden-light tint, stronger in brights than shadows
    goldw = np.clip((Y - 60.0) / 160.0, 0, 1) ** 0.8
    t = (0.35 * w * goldw)[..., None]
    gold = np.array([1.0, 0.86, 0.58], np.float32)
    f3 = out * (1.0 + (gold - 1.0) * t)
    out = _scale_by_ratio(f3, _luma(f3), Y)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def apply_smoother(rgb_u8, smoother):
    """Edge-preserving softener: blurs flat areas (skin, skies), keeps edges."""
    if smoother <= 0:
        return rgb_u8
    amt = smoother / 100.0
    f = rgb_u8.astype(np.float32)
    Y = _luma(f)
    r = max(2, round(min(Y.shape) / 200))
    smooth = np.stack([uniform_filter(f[..., c], size=2 * r + 1, mode="nearest")
                       for c in range(3)], axis=-1)
    gy, gx = np.gradient(Y)
    edge = np.abs(gx) + np.abs(gy)
    w = np.exp(-((edge / 12.0) ** 2))                    # ~1 flat, ~0 on edges
    blend = (amt * w)[..., None]
    out = f * (1 - blend) + smooth * blend
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def apply_glow(rgb_u8, glow):
    """Golden glow: warm-tinted Orton bloom, weighted into the highlights."""
    if glow <= 0:
        return rgb_u8
    amt = glow / 100.0
    f = rgb_u8.astype(np.float32) / 255.0
    Y = _luma(f * 255.0) / 255.0
    r = max(4, round(min(Y.shape) / 60))
    blur = np.stack([uniform_filter(f[..., c], size=2 * r + 1, mode="nearest")
                     for c in range(3)], axis=-1)
    gold = np.array([1.0, 0.88, 0.62], np.float32)
    glowc = np.clip(blur * gold * 1.15, 0, 1)
    screen = 1 - (1 - f) * (1 - glowc)
    whi = np.clip((Y - 0.25) / 0.75, 0, 1)[..., None]    # reach into the mids
    out = f + (screen - f) * (amt * 0.85) * whi
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- golden-ratio crop
def golden_ratio_crop(shape, energy, phi=1.6180339887):
    """Largest phi-aspect window, positioned so the subject (energy centroid)
    lands on a golden-ratio 'sweet spot' - a phi-grid intersection, as in the
    classic phi-grid composition guide."""
    h, w = shape
    if w >= h:
        cw, ch = w, round(w / phi)
        if ch > h:
            ch, cw = h, round(h * phi)
    else:
        ch, cw = h, round(h / phi)
        if cw > w:
            cw, ch = w, round(w * phi)
    p = 1 - 1 / phi                                # 0.382 — the phi lines
    spots = [(p, p), (p, 1 - p), (1 - p, p), (1 - p, 1 - p)]
    total = energy.sum()
    if total <= 0:
        return 0, 0, cw, ch
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((xs * energy).sum() / total)        # subject centroid
    cy = float((ys * energy).sum() / total)
    sy = max(1, (h - ch) // 24)
    sx = max(1, (w - cw) // 24)
    best, bs = (0, 0), 1e18
    for y0 in range(0, h - ch + 1, sy):
        ny = (cy - y0) / ch
        if not 0 <= ny <= 1:
            continue                               # keep the subject in frame
        for x0 in range(0, w - cw + 1, sx):
            nx = (cx - x0) / cw
            if not 0 <= nx <= 1:
                continue
            d = min((nx - px) ** 2 + (ny - py) ** 2 for px, py in spots)
            if d < bs:
                bs, best = d, (x0, y0)
    x0, y0 = best
    return x0, y0, x0 + cw, y0 + ch


# ---------------------------------------------------------------- AI detail restore
def detail_restore(ai_u8, src_u8, strength=0.65):
    """Diffusion editors regenerate pixels, which washes out micro-texture
    (skin pores, hair, fabric weave, foliage). This injects the ORIGINAL
    photo's high-frequency luma residual back into the AI result, restoring
    its fine detail without disturbing the AI's new grade or lighting.

    ai_u8 / src_u8 must be the same shape (server resizes beforehand)."""
    if strength <= 0:
        return ai_u8
    f_ai = ai_u8.astype(np.float32)
    f_src = src_u8.astype(np.float32)
    Y_src = _luma(f_src)
    r = max(1, round(min(Y_src.shape) / 600))        # very fine detail band
    resid = Y_src - uniform_filter(Y_src, size=2 * r + 1, mode="nearest")
    # don't amplify noise the AI already removed in flat regions: dampen where
    # the SOURCE residual is tiny relative to its global scale
    scale = max(float(np.median(np.abs(resid)) * 1.4826), 1e-3)
    damp = np.clip(np.abs(resid) / (3 * scale), 0.25, 1.0)
    Y_ai = _luma(f_ai)
    out = _scale_by_ratio(f_ai, Y_ai, Y_ai + resid * damp * strength)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- HSV helpers (recipe mode)
def _rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, -1)
    mn = np.min(rgb, -1)
    diff = mx - mn
    h = np.zeros_like(mx)
    nz = diff > 1e-6
    mr = nz & (mx == r)
    mg = nz & (mx == g)
    mb = nz & ~mr & ~mg
    h[mr] = ((g - b)[mr] / diff[mr]) % 6
    h[mg] = (b - r)[mg] / diff[mg] + 2
    h[mb] = (r - g)[mb] / diff[mb] + 4
    h = h / 6.0
    s = diff / np.maximum(mx, 1e-6)
    return h, s, mx


def _hsv_to_rgb(h, s, v):
    i = np.floor(h * 6).astype(np.int32) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    out = np.zeros((*h.shape, 3), np.float32)
    for k, (rr, gg, bb) in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                      (p, q, v), (t, p, v), (v, p, q)]):
        m = i == k
        out[..., 0][m] = rr[m]
        out[..., 1][m] = gg[m]
        out[..., 2][m] = bb[m]
    return out


# ---------------------------------------------------------------- recipe mode
_HSL_SECTORS = {"red": (0, 22), "orange": (30, 18), "yellow": (60, 18),
                "green": (120, 45), "aqua": (180, 22), "blue": (225, 28),
                "purple": (275, 25), "magenta": (320, 22)}


def _hue_weight(h_deg, center, width):
    d = np.abs((h_deg - center + 180) % 360 - 180)
    return np.clip(1 - d / width, 0, 1)


def apply_recipe(rgb_u8, recipe, intensity=100):
    """Execute a structured edit recipe (from the vision style-analyst)
    parametrically - the photo's pixels are adjusted, never regenerated."""
    r = recipe
    # 1) white balance + exposure in linear light
    f = _LIN_LUT[rgb_u8]
    wt = float(r.get("wb_temp", 0))
    f = f * np.array([1 + 0.18 * wt, 1 + 0.03 * wt, 1 - 0.20 * wt], np.float32)
    f = f * float(2.0 ** np.clip(r.get("exposure", 0), -2, 2))
    g = _linear_to_srgb(f)                                     # gamma floats 0..1
    # 2) contrast around a mid pivot
    c = np.clip(r.get("contrast", 0), -1, 1) * 0.9
    g = np.clip(0.5 + (g - 0.5) * (1 + c), 0, 1)
    # 3) black / white points on luma
    Y = _luma(g * 255) / 255
    bl, wh = np.clip(r.get("blacks", 0), -1, 1), np.clip(r.get("whites", 0), -1, 1)
    Y2 = np.clip(Y + bl * 0.22 * (1 - Y) * (1 - Y) + wh * 0.22 * Y * Y, 0, 1)
    g = g * (Y2 / np.maximum(Y, 0.02))[..., None]
    # 4) tone curve on luma
    pts = r.get("tone_curve") or []
    if len(pts) >= 2:
        pts = sorted((float(np.clip(a, 0, 1)), float(np.clip(b, 0, 1))) for a, b in pts)
        xp, fp = zip(*pts)
        Y2 = _luma(g * 255) / 255
        Y3 = np.interp(Y2, xp, fp).astype(np.float32)
        g = g * (Y3 / np.maximum(Y2, 0.02))[..., None]
    # 5) saturation / vibrance / HSL sector shifts in HSV
    h, s, v = _rgb_to_hsv(np.clip(g, 0, 1))
    s = np.clip(s * (1 + np.clip(r.get("saturation", 0), -1, 1) * 0.6), 0, 1)
    s = np.clip(s * (1 + np.clip(r.get("vibrance", 0), -1, 1) * 0.6 * (1 - s)), 0, 1)
    hsl = r.get("hsl") or {}
    hd = h * 360.0
    dh = np.zeros_like(hd)
    ds = np.zeros_like(hd)
    dl = np.zeros_like(hd)
    for name, (cen, wid) in _HSL_SECTORS.items():
        if name in hsl and isinstance(hsl[name], (list, tuple)) and len(hsl[name]) == 3:
            w = _hue_weight(hd, cen, wid)
            dh += float(np.clip(hsl[name][0], -1, 1)) * (15 / 360) * w
            ds += float(np.clip(hsl[name][1], -1, 1)) * 0.5 * w
            dl += float(np.clip(hsl[name][2], -1, 1)) * 0.25 * w
    h = (h + dh) % 1.0
    s = np.clip(s * (1 + ds), 0, 1)
    v = np.clip(v * (1 + dl), 0, 1)
    g = _hsv_to_rgb(h, s, v)
    # 6) vignette (positive = darker corners)
    vig = float(np.clip(r.get("vignette", 0), -1, 1))
    if vig:
        Yg = _luma(g * 255)
        rr, _, _ = _coords(*Yg.shape)
        factor = 1 - np.clip(vig, 0, 1) * 0.5 * rr ** 1.5 + np.clip(-vig, 0, 1) * 0.35 * rr ** 1.5
        g = g * (Yg * factor / np.maximum(Yg, 0.02))[..., None]
    out = (np.clip(g, 0, 1) * 255 + 0.5).astype(np.uint8)
    # 7) clarity / sharpen / grain via the existing deterministic passes
    clar = float(np.clip(r.get("clarity", 0), 0, 1)) * 100
    shp = float(np.clip(r.get("sharpen", 0), 0, 1)) * 100
    if clar or shp:
        out = detail_boost(out, clar, shp)
    gr = float(np.clip(r.get("grain", 0), 0, 1))
    if gr > 0:
        Yo = _luma(out.astype(np.float32))
        t1, t2 = (float(x) for x in np.percentile(Yo, [33.333, 66.667]))
        out = apply_grain(out, {"sigma": [gr * 8, gr * 10, gr * 8], "t1": t1, "t2": t2}, 100)
    # 8) intensity blend against the untouched original
    t = np.clip(intensity, 0, 100) / 100.0
    if t < 1:
        out = np.clip(rgb_u8.astype(np.float32) * (1 - t)
                      + out.astype(np.float32) * t + 0.5, 0, 255).astype(np.uint8)
    return out


def phi_crop(rgb_u8):
    """Apply the golden-ratio composition crop to a finished image."""
    small = rgb_u8[::4, ::4]
    gy2, gx2 = np.gradient(_luma(small.astype(np.float32)))
    energy = np.abs(gx2) + np.abs(gy2)
    x0, y0, x1, y1 = golden_ratio_crop(small.shape[:2], energy)
    return rgb_u8[y0 * 4:y1 * 4, x0 * 4:x1 * 4]


# ---------------------------------------------------------------- main entry
def grade_pair(ref_small, tgt_small, tgt_full,
               intensity=85, tone=True, color=True, bands=True,
               clarity=0, sharpen=0, grain=0, vibe=0,
               warmth=0, smoother=0, glow=0, crop="",
               ref_grain_src=None):
    """
    ref_small / tgt_small : uint8 RGB arrays used for statistics (analysis size)
    tgt_full              : uint8 RGB array the grade is applied to (full size)
    ref_grain_src         : optional larger reference copy for accurate grain
                            estimation (pixel-level grain vanishes when downscaled)
    returns               : uint8 RGB array, same shape as tgt_full
    """
    rL, ra, rb = rgb_to_lab(ref_small)
    sL, sa, sb = rgb_to_lab(tgt_small)
    L, a, b = rgb_to_lab(tgt_full)

    t = intensity / 100.0
    L2, A2, B2 = L.copy(), a.copy(), b.copy()

    if tone:
        lutL = _hist_lut(sL.ravel(), rL.ravel(), 0, 100)
        L2 = L + (_apply_lut(L, lutL, 0, 100) - L) * t

    if color:
        if bands:
            t1, t2 = np.percentile(sL, [33.333, 66.667])
            band_of = np.digitize(L, [t1, t2])            # full-res band map
            sband_of = np.digitize(sL, [t1, t2])          # analysis band map
            rband_of = np.digitize(rL, [t1, t2])
            for bi in range(3):
                sm = sband_of == bi
                rm = rband_of == bi
                if sm.sum() < 200 or rm.sum() < 200:      # sparse band -> global fallback
                    sm = np.ones_like(sband_of, bool)
                    rm = np.ones_like(rband_of, bool)
                lutA = _hist_lut(sa[sm].ravel(), ra[rm].ravel(), -128, 128)
                lutB = _hist_lut(sb[sm].ravel(), rb[rm].ravel(), -128, 128)
                m = band_of == bi
                A2[m] = a[m] + (_apply_lut(a[m], lutA, -128, 128) - a[m]) * t
                B2[m] = b[m] + (_apply_lut(b[m], lutB, -128, 128) - b[m]) * t
        else:
            lutA = _hist_lut(sa.ravel(), ra.ravel(), -128, 128)
            lutB = _hist_lut(sb.ravel(), rb.ravel(), -128, 128)
            A2 = a + (_apply_lut(a, lutA, -128, 128) - a) * t
            B2 = b + (_apply_lut(b, lutB, -128, 128) - b) * t

    out = lab_to_rgb(L2, A2, B2)
    if warmth > 0:                                     # golden temperature shift
        out = apply_warmth(out, warmth)
    if clarity > 0 or sharpen > 0:
        out = detail_boost(out, clarity, sharpen)
    if vibe > 0:                                       # relight with reference's vibe
        out = apply_light(out, estimate_light(ref_small), estimate_light(tgt_small), vibe)
    if smoother > 0:                                   # edge-preserving softening
        out = apply_smoother(out, smoother)
    if glow > 0:                                       # golden highlight bloom
        out = apply_glow(out, glow)
    if grain > 0:                                      # reference's granularity, on top
        src = ref_grain_src if ref_grain_src is not None else ref_small
        out = apply_grain(out, estimate_grain(src), grain)
    if crop == "phi":                                  # aesthetic golden-ratio crop
        out = phi_crop(out)
    return out
