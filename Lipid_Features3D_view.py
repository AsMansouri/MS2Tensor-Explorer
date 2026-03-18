# scripts/dash_feature_3d_viewer.py
# Multi-tab Dash app:
#   - Home: 3D viewer (samples + condition means) + dataset type selector
#   - Comparison:
#        * Absent/Present
#        * DE: gapfill (median-per-feature-per-condition) + log2FC + test + p + FDR
#        * 2D/3D visualization toggle
#        * Kendrick plot shown for lipidomics
#        * Ion-pattern interpretation for lipidomics:
#            - candidate homologous CH2 series
#            - candidate unsaturation-like pairs
#            - candidate adduct groups
#            - RT coherence scoring
#            - priority ranking
#            - optional highlighting of selected interpretation group on both plots
#
# Run:
#   pip install dash plotly pandas numpy openpyxl scipy
#   python scripts/dash_feature_3d_viewer.py
# Open:
#   http://127.0.0.1:8050/

import base64
import io
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update

# -----------------------------
# Optional stats backends
# -----------------------------
try:
    from scipy.stats import ttest_ind
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Optional limma (requires rpy2 + R + limma installed). Best-effort optional.
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    _HAVE_RPY2 = True
except Exception:
    _HAVE_RPY2 = False


# -----------------------------
# Constants
# -----------------------------
KENDRICK_BASE_EXACT = 14.01565
KENDRICK_BASE_NOMINAL = 14.00000
UNSATURATION_DELTA = 2.01565

COMMON_ADDUCT_OFFSETS_POS = [
    ("H↔NH4", 17.02655),
    ("H↔Na", 21.98194),
    ("NH4↔Na", 4.95539),
    ("H↔K", 37.95588),
    ("Na↔K", 15.97394),
]

COMMON_ADDUCT_OFFSETS_NEG = [
    ("H↔Cl", 35.97668),
    ("H↔Formate", 44.99820),
    ("H↔Acetate", 59.01385),
]

DEFAULT_KMD_TOL = 0.01
DEFAULT_PPM_TOL = 10.0
DEFAULT_RT_WINDOW = 0.30


# -----------------------------
# Helpers
# -----------------------------
def safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")


def safe_numeric_frame(df_block: pd.DataFrame) -> pd.DataFrame:
    out = df_block.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ""), errors="coerce")
    return out


def parse_uploaded(contents, filename) -> pd.DataFrame:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    name = (filename or "").lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(decoded))

    text = decoded.decode("utf-8", errors="replace")
    if name.endswith(".tsv") or name.endswith(".tab"):
        return pd.read_csv(io.StringIO(text), sep="\t")
    if name.endswith(".csv"):
        return pd.read_csv(io.StringIO(text))
    try:
        return pd.read_csv(io.StringIO(text), sep="\t")
    except Exception:
        return pd.read_csv(io.StringIO(text))


def detect_axes(df: pd.DataFrame):
    mz = "row m/z" if "row m/z" in df.columns else None
    rt = "row retention time" if "row retention time" in df.columns else None

    if "row ion mobility" in df.columns:
        z = "row ion mobility"
    elif "row CCS" in df.columns:
        z = "row CCS"
    else:
        z = None

    if mz is None or rt is None:
        raise ValueError("Expected MZmine columns: 'row m/z' and 'row retention time'.")

    if z is None:
        z = "__z_dummy__"
        df[z] = 0.0

    return mz, rt, z


def get_sample_cols(df: pd.DataFrame):
    return [c for c in df.columns if c.endswith(" Peak area")]


def sample_name_from_peak_area_col(col: str) -> str:
    return col.replace(" Peak area", "").strip()


def pick_annotation_sample_id_col(ann: pd.DataFrame):
    candidates = ["Files", "File", "Sample_name", "Sample", "SampleName", "Filename", "RawFile"]
    for c in candidates:
        if c in ann.columns:
            return c
    return None


def extract_row_index(clickData):
    try:
        pt = clickData["points"][0]
    except Exception:
        return None

    cd = pt.get("customdata", None)
    if cd is not None:
        try:
            return int(cd)
        except Exception:
            pass

    txt = pt.get("text", "")
    if isinstance(txt, str) and "row_index=" in txt:
        try:
            return int(txt.split("row_index=")[-1].strip())
        except Exception:
            pass

    ht = pt.get("hovertext", "")
    if isinstance(ht, str) and "row_index=" in ht:
        try:
            return int(ht.split("row_index=")[-1].strip())
        except Exception:
            return None

    return None


def sanitize_filename_part(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in str(s))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    q = np.full(n, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q

    p_ok = p[ok]
    order = np.argsort(p_ok)
    ranked = p_ok[order]
    m = ranked.size
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)

    q_ok = np.empty_like(p_ok)
    q_ok[order] = q_ranked
    q[ok] = q_ok
    return q


def median_gapfill_by_feature(mat: np.ndarray) -> np.ndarray:
    out = mat.copy()
    med = np.nanmedian(out, axis=1)
    inds = np.where(np.isnan(out))
    rows = inds[0]
    out[inds] = med[rows]
    return out


def welch_ttest_fallback(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mean_a = np.nanmean(a, axis=1)
    mean_b = np.nanmean(b, axis=1)
    var_a = np.nanvar(a, axis=1, ddof=1)
    var_b = np.nanvar(b, axis=1, ddof=1)
    na = np.sum(np.isfinite(a), axis=1)
    nb = np.sum(np.isfinite(b), axis=1)

    se = np.sqrt(var_a / na + var_b / nb)
    t = (mean_a - mean_b) / se

    abs_t = np.abs(t)
    p = 2 * (1 - 0.5 * (1 + erf(abs_t / np.sqrt(2))))
    p = np.clip(p, 0, 1)
    p[~np.isfinite(se)] = np.nan
    return p


def erf(x):
    x = np.asarray(x, dtype=float)
    return np.vectorize(math.erf)(x)


def ppm_window(mass, ppm):
    return abs(float(mass)) * float(ppm) / 1e6


def best_ion_str(x):
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    return str(x).strip()


def infer_ion_mode_from_string(s):
    s = best_ion_str(s)
    if "-" in s:
        return "neg"
    if "+" in s:
        return "pos"
    return "unknown"


def infer_global_ion_mode(series):
    modes = pd.Series(series).map(infer_ion_mode_from_string)
    counts = modes.value_counts()
    if counts.empty:
        return "unknown"
    return counts.index[0]


def to_native(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def serialize_records(records):
    out = []
    for rec in records:
        row = {}
        for k, v in rec.items():
            if isinstance(v, list):
                row[k] = [to_native(x) for x in v]
            else:
                row[k] = to_native(v)
        out.append(row)
    return out


def parse_interp_store(store):
    if not store:
        return {"analysis_rows": [], "homolog_groups": [], "adduct_groups": [], "unsat_pairs": [], "summary": ""}
    return {
        "analysis_rows": store.get("analysis_rows", []),
        "homolog_groups": store.get("homolog_groups", []),
        "adduct_groups": store.get("adduct_groups", []),
        "unsat_pairs": store.get("unsat_pairs", []),
        "summary": store.get("summary", ""),
    }


def get_selected_group_members(interp_store, selected_group_id):
    parsed = parse_interp_store(interp_store)
    if not selected_group_id:
        return []

    for rec in parsed["homolog_groups"]:
        if rec.get("group_id") == selected_group_id:
            return rec.get("members", [])

    for rec in parsed["adduct_groups"]:
        if rec.get("group_id") == selected_group_id:
            return rec.get("members", [])

    return []


# -----------------------------
# Kendrick helpers
# -----------------------------
def compute_kendrick_from_mz(mz_values):
    mz = pd.to_numeric(pd.Series(mz_values), errors="coerce").to_numpy(dtype=float)
    km = mz * (KENDRICK_BASE_NOMINAL / KENDRICK_BASE_EXACT)
    nominal_km = np.round(km)
    kmd = nominal_km - km
    return km, nominal_km, kmd


def build_empty_kendrick_figure(title="Kendrick plot"):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Kendrick Mass (CH₂ base)",
        yaxis_title="Kendrick Mass Defect (KMD)",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_kendrick_analysis_df(cmp_result, de_json, df_json, meta, dataset_type):
    if dataset_type != "lipidomics":
        return pd.DataFrame()
    if not cmp_result or not df_json or not meta:
        return pd.DataFrame()

    df = pd.read_json(df_json, orient="split")
    mz_col = meta["mz_col"]
    rt_col = meta["rt_col"]
    mode = cmp_result.get("mode", "present_absent")

    keep_cols = ["row ID", mz_col, rt_col]
    if "best ion" in df.columns:
        keep_cols.append("best ion")

    if mode == "de":
        if not de_json:
            return pd.DataFrame()
        de_df = pd.read_json(de_json, orient="split")
        if "significant" not in de_df.columns:
            return pd.DataFrame()
        sub = de_df[de_df["significant"] == True].copy()
        if sub.empty:
            return pd.DataFrame()
        if "_global_index_" not in sub.columns:
            sub["_global_index_"] = sub.index
        out = sub.copy()
        out["mode"] = "de"
        out["category"] = "significant_de"
    else:
        idx = cmp_result.get("row_indices", [])
        cats = cmp_result.get("row_categories", [])
        if not idx:
            return pd.DataFrame()
        out = df.iloc[idx][keep_cols].copy()
        out["_global_index_"] = idx
        out["mode"] = "present_absent"
        out["category"] = cats
        out["log2FC"] = np.nan
        out["p_value"] = np.nan
        out["FDR"] = np.nan
        out["significant"] = False

    if "best ion" not in out.columns:
        out["best ion"] = ""

    out[mz_col] = safe_numeric_series(out[mz_col])
    out[rt_col] = safe_numeric_series(out[rt_col])
    out = out.dropna(subset=[mz_col, rt_col]).copy()

    km, nominal_km, kmd = compute_kendrick_from_mz(out[mz_col])
    out["KM"] = km
    out["NominalKM"] = nominal_km
    out["KMD"] = kmd
    out["ion_mode"] = out["best ion"].map(infer_ion_mode_from_string)

    return out


def rt_coherence_metrics(group_df, mz_col, rt_col):
    if len(group_df) < 2:
        return {"rho": None, "positive_fraction": None, "label": "single"}

    s = group_df.sort_values(mz_col)
    mz_rank = s[mz_col].rank(method="average")
    rt_rank = s[rt_col].rank(method="average")
    rho = mz_rank.corr(rt_rank, method="pearson")
    rt_diff = np.diff(s[rt_col].to_numpy(dtype=float))
    pos_frac = float(np.mean(rt_diff >= 0)) if len(rt_diff) > 0 else None

    if rho is None or np.isnan(rho):
        label = "weak"
    elif rho >= 0.70:
        label = "strong"
    elif rho >= 0.30:
        label = "moderate"
    else:
        label = "weak"

    return {"rho": rho, "positive_fraction": pos_frac, "label": label}


def connected_components_from_edges(node_ids, edges):
    adjacency = {n: set() for n in node_ids}
    for a, b, _ in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    seen = set()
    comps = []
    for n in node_ids:
        if n in seen:
            continue
        stack = [n]
        comp = []
        seen.add(n)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nei in adjacency[cur]:
                if nei not in seen:
                    seen.add(nei)
                    stack.append(nei)
        comps.append(sorted(comp))
    return comps


def detect_homologous_series(dfk, mz_col, rt_col, ppm_tol=DEFAULT_PPM_TOL, kmd_tol=DEFAULT_KMD_TOL, require_same_ion=False, require_rt=False):
    if dfk.empty or len(dfk) < 2:
        return []

    work = dfk.sort_values(mz_col).reset_index(drop=True).copy()
    node_ids = work["_global_index_"].astype(int).tolist()
    mz_map = dict(zip(work["_global_index_"], work[mz_col]))
    rt_map = dict(zip(work["_global_index_"], work[rt_col]))
    kmd_map = dict(zip(work["_global_index_"], work["KMD"]))
    ion_map = dict(zip(work["_global_index_"], work["best ion"]))
    cat_map = dict(zip(work["_global_index_"], work["category"]))

    edges = []

    ids = work["_global_index_"].astype(int).tolist()
    for i in range(len(ids)):
        id_i = ids[i]
        mz_i = float(mz_map[id_i])
        for j in range(i + 1, len(ids)):
            id_j = ids[j]
            mz_j = float(mz_map[id_j])
            delta = abs(mz_j - mz_i)

            step_hit = None
            for n_steps in (1, 2, 3):
                target = n_steps * KENDRICK_BASE_EXACT
                tol = max(ppm_window((mz_i + mz_j) / 2.0, ppm_tol), 0.005)
                if abs(delta - target) <= tol:
                    step_hit = n_steps
                    break
            if step_hit is None:
                continue

            if abs(float(kmd_map[id_i]) - float(kmd_map[id_j])) > float(kmd_tol):
                continue

            if require_same_ion:
                ion_i = best_ion_str(ion_map[id_i])
                ion_j = best_ion_str(ion_map[id_j])
                if ion_i and ion_j and ion_i != ion_j:
                    continue

            edges.append((id_i, id_j, step_hit))

    if not edges:
        return []

    comps = connected_components_from_edges(node_ids, edges)
    comps = [c for c in comps if len(c) >= 2]
    if not comps:
        return []

    groups = []
    group_num = 1
    for comp in comps:
        g = work[work["_global_index_"].isin(comp)].copy()
        rt_stats = rt_coherence_metrics(g, mz_col, rt_col)
        if require_rt and rt_stats["label"] == "weak":
            continue

        comp_edges = [e for e in edges if e[0] in comp and e[1] in comp]
        step_counts = {}
        for _, _, st in comp_edges:
            step_counts[st] = step_counts.get(st, 0) + 1

        ion_counts = g["best ion"].map(best_ion_str).replace("", np.nan).dropna().value_counts()
        dominant_ion = ion_counts.index[0] if not ion_counts.empty else ""

        vals = pd.to_numeric(g.get("log2FC", pd.Series(np.nan, index=g.index)), errors="coerce")
        mean_abs_log2fc = float(vals.abs().mean()) if vals.notna().any() else None

        cat_counts = g["category"].astype(str).value_counts()
        dominant_category = cat_counts.index[0] if not cat_counts.empty else ""

        score = float(len(g) * 2.0)
        if rt_stats["label"] == "strong":
            score += 2.0
        elif rt_stats["label"] == "moderate":
            score += 1.0
        if mean_abs_log2fc is not None:
            score += min(mean_abs_log2fc, 5.0)

        groups.append(
            {
                "group_id": f"H{group_num:02d}",
                "group_type": "homologous",
                "members": sorted([int(x) for x in comp]),
                "n_members": int(len(g)),
                "mz_min": float(g[mz_col].min()),
                "mz_max": float(g[mz_col].max()),
                "rt_min": float(g[rt_col].min()),
                "rt_max": float(g[rt_col].max()),
                "kmd_mean": float(g["KMD"].mean()),
                "kmd_spread": float(g["KMD"].max() - g["KMD"].min()),
                "dominant_best_ion": dominant_ion,
                "dominant_category": dominant_category,
                "rt_rho": None if rt_stats["rho"] is None or pd.isna(rt_stats["rho"]) else float(rt_stats["rho"]),
                "rt_positive_fraction": None if rt_stats["positive_fraction"] is None else float(rt_stats["positive_fraction"]),
                "rt_coherence": rt_stats["label"],
                "step_edges": int(len(comp_edges)),
                "step_counts": step_counts,
                "mean_abs_log2FC": mean_abs_log2fc,
                "priority_score": round(score, 3),
            }
        )
        group_num += 1

    return sorted(groups, key=lambda x: x["priority_score"], reverse=True)


def detect_unsaturation_pairs(dfk, mz_col, rt_col, ppm_tol=DEFAULT_PPM_TOL, rt_window=DEFAULT_RT_WINDOW):
    if dfk.empty or len(dfk) < 2:
        return []

    pairs = []
    ids = dfk["_global_index_"].astype(int).tolist()
    mz_map = dict(zip(dfk["_global_index_"], dfk[mz_col]))
    rt_map = dict(zip(dfk["_global_index_"], dfk[rt_col]))
    ion_map = dict(zip(dfk["_global_index_"], dfk["best ion"]))

    for i in range(len(ids)):
        id_i = ids[i]
        mz_i = float(mz_map[id_i])
        rt_i = float(rt_map[id_i])
        ion_i = best_ion_str(ion_map[id_i])

        for j in range(i + 1, len(ids)):
            id_j = ids[j]
            mz_j = float(mz_map[id_j])
            rt_j = float(rt_map[id_j])
            ion_j = best_ion_str(ion_map[id_j])

            delta = abs(mz_j - mz_i)
            tol = max(ppm_window((mz_i + mz_j) / 2.0, ppm_tol), 0.003)
            if abs(delta - UNSATURATION_DELTA) > tol:
                continue
            if abs(rt_j - rt_i) > float(rt_window):
                continue
            if ion_i and ion_j and ion_i != ion_j:
                continue

            pairs.append(
                {
                    "member_a": int(id_i),
                    "member_b": int(id_j),
                    "mz_delta": float(delta),
                    "rt_delta": float(abs(rt_j - rt_i)),
                    "best_ion": ion_i if ion_i == ion_j else "",
                }
            )
    return pairs


def detect_adduct_groups(dfk, mz_col, rt_col, ppm_tol=DEFAULT_PPM_TOL, rt_window=DEFAULT_RT_WINDOW):
    if dfk.empty or len(dfk) < 2:
        return []

    global_mode = infer_global_ion_mode(dfk["best ion"])
    if global_mode == "neg":
        offsets = COMMON_ADDUCT_OFFSETS_NEG
    elif global_mode == "pos":
        offsets = COMMON_ADDUCT_OFFSETS_POS
    else:
        offsets = COMMON_ADDUCT_OFFSETS_POS + COMMON_ADDUCT_OFFSETS_NEG

    ids = dfk["_global_index_"].astype(int).tolist()
    mz_map = dict(zip(dfk["_global_index_"], dfk[mz_col]))
    rt_map = dict(zip(dfk["_global_index_"], dfk[rt_col]))
    ion_map = dict(zip(dfk["_global_index_"], dfk["best ion"]))

    edges = []
    for i in range(len(ids)):
        id_i = ids[i]
        mz_i = float(mz_map[id_i])
        rt_i = float(rt_map[id_i])
        ion_i = best_ion_str(ion_map[id_i])

        for j in range(i + 1, len(ids)):
            id_j = ids[j]
            mz_j = float(mz_map[id_j])
            rt_j = float(rt_map[id_j])
            ion_j = best_ion_str(ion_map[id_j])

            rt_diff = abs(rt_j - rt_i)
            if rt_diff > float(rt_window):
                continue

            delta = abs(mz_j - mz_i)
            matched = None
            for label, offset in offsets:
                tol = max(ppm_window((mz_i + mz_j) / 2.0, ppm_tol), 0.003)
                if abs(delta - offset) <= tol:
                    matched = (label, offset)
                    break
            if matched is None:
                continue

            if ion_i and ion_j and ion_i == ion_j:
                continue

            edges.append((id_i, id_j, matched[0]))

    if not edges:
        return []

    comps = connected_components_from_edges(ids, edges)
    comps = [c for c in comps if len(c) >= 2]
    if not comps:
        return []

    groups = []
    group_num = 1
    for comp in comps:
        g = dfk[dfk["_global_index_"].isin(comp)].copy()
        comp_edges = [e for e in edges if e[0] in comp and e[1] in comp]
        label_counts = {}
        for _, _, lbl in comp_edges:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        dominant_label = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[0][0] if label_counts else ""
        ion_labels = sorted({best_ion_str(x) for x in g["best ion"] if best_ion_str(x)})

        vals = pd.to_numeric(g.get("log2FC", pd.Series(np.nan, index=g.index)), errors="coerce")
        mean_abs_log2fc = float(vals.abs().mean()) if vals.notna().any() else None

        score = float(len(g) * 1.5 + len(comp_edges) * 0.5)
        if mean_abs_log2fc is not None:
            score += min(mean_abs_log2fc, 5.0)

        groups.append(
            {
                "group_id": f"A{group_num:02d}",
                "group_type": "adduct",
                "members": sorted([int(x) for x in comp]),
                "n_members": int(len(g)),
                "mz_min": float(g[mz_col].min()),
                "mz_max": float(g[mz_col].max()),
                "rt_min": float(g[rt_col].min()),
                "rt_max": float(g[rt_col].max()),
                "dominant_offset": dominant_label,
                "edge_count": int(len(comp_edges)),
                "ion_labels": ", ".join(ion_labels),
                "mean_abs_log2FC": mean_abs_log2fc,
                "priority_score": round(score, 3),
            }
        )
        group_num += 1

    return sorted(groups, key=lambda x: x["priority_score"], reverse=True)


def build_interpretation_summary(dfk, homolog_groups, adduct_groups, unsat_pairs, cmp_result):
    if dfk.empty:
        return "No lipid features available for Kendrick interpretation."

    mode = cmp_result.get("mode", "present_absent")
    c1 = cmp_result.get("cond1", "")
    c2 = cmp_result.get("cond2", "")

    lines = [
        "Ion-pattern interpretation",
        f"Mode: {mode}",
        f"Comparison: {c1} vs {c2}",
        f"Features analyzed: {len(dfk)}",
        f"Candidate homologous groups: {len(homolog_groups)}",
        f"Candidate adduct groups: {len(adduct_groups)}",
        f"Unsaturation-like pairs (~2.01565 Da): {len(unsat_pairs)}",
        "",
        "Priority groups:",
    ]

    combined = []
    for g in homolog_groups:
        combined.append((g["priority_score"], g["group_id"], "homologous", g["n_members"]))
    for g in adduct_groups:
        combined.append((g["priority_score"], g["group_id"], "adduct", g["n_members"]))
    combined = sorted(combined, reverse=True)[:5]

    if not combined:
        lines.append("  None detected.")
    else:
        for score, gid, gtype, n_members in combined:
            lines.append(f"  {gid}: {gtype}, n={n_members}, score={score:.2f}")

    if homolog_groups:
        top = homolog_groups[0]
        lines.extend(
            [
                "",
                "Top homologous series:",
                f"  {top['group_id']} | n={top['n_members']} | KMD spread={top['kmd_spread']:.5f} | RT coherence={top['rt_coherence']}",
            ]
        )

    if adduct_groups:
        top = adduct_groups[0]
        lines.extend(
            [
                "",
                "Top adduct group:",
                f"  {top['group_id']} | n={top['n_members']} | dominant offset={top['dominant_offset']}",
            ]
        )

    return "\n".join(lines)


def homolog_table_records(groups):
    rows = []
    for g in groups:
        rows.append(
            {
                "group_id": g["group_id"],
                "n_members": g["n_members"],
                "mz_range": f"{g['mz_min']:.4f} – {g['mz_max']:.4f}",
                "rt_range": f"{g['rt_min']:.3f} – {g['rt_max']:.3f}",
                "kmd_spread": round(g["kmd_spread"], 6),
                "dominant_best_ion": g["dominant_best_ion"],
                "dominant_category": g["dominant_category"],
                "rt_coherence": g["rt_coherence"],
                "mean_abs_log2FC": None if g["mean_abs_log2FC"] is None else round(g["mean_abs_log2FC"], 3),
                "priority_score": round(g["priority_score"], 3),
            }
        )
    return rows


def adduct_table_records(groups):
    rows = []
    for g in groups:
        rows.append(
            {
                "group_id": g["group_id"],
                "n_members": g["n_members"],
                "mz_range": f"{g['mz_min']:.4f} – {g['mz_max']:.4f}",
                "rt_range": f"{g['rt_min']:.3f} – {g['rt_max']:.3f}",
                "dominant_offset": g["dominant_offset"],
                "ion_labels": g["ion_labels"],
                "mean_abs_log2FC": None if g["mean_abs_log2FC"] is None else round(g["mean_abs_log2FC"], 3),
                "priority_score": round(g["priority_score"], 3),
            }
        )
    return rows


# -----------------------------
# Plot builders
# -----------------------------
def build_figure_3d(
    df: pd.DataFrame,
    mz_col: str,
    rt_col: str,
    z_col: str,
    color_values: np.ndarray,
    colorbar_title: str,
    max_points: int = 30000,
    title: str = "",
    point_size: int = 4,
    colorscale: str = "Plasma",
):
    n = len(df)
    if n == 0:
        return go.Figure().update_layout(template="plotly_dark", title=title)

    idx = np.arange(n)
    if n > max_points:
        w = np.abs(color_values)
        w = np.nan_to_num(w, nan=0.0)
        if w.sum() <= 0:
            take = np.random.choice(idx, size=max_points, replace=False)
        else:
            w = w / w.sum()
            take = np.random.choice(idx, size=max_points, replace=False, p=w)
    else:
        take = idx

    x = safe_numeric_series(df.iloc[take][rt_col]).to_numpy()
    y = safe_numeric_series(df.iloc[take][mz_col]).to_numpy()
    z = safe_numeric_series(df.iloc[take][z_col]).to_numpy()
    c = color_values[take]

    cd = df.iloc[take]["_global_index_"].to_numpy() if "_global_index_" in df.columns else df.index.values[take]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=c,
                    colorscale=colorscale,
                    opacity=0.85,
                    colorbar=dict(title=colorbar_title),
                ),
                customdata=cd,
                hovertext=[f"row_index={i}" for i in cd],
                hovertemplate="%{hovertext}<br>color=%{marker.color:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        scene=dict(xaxis_title="RT", yaxis_title="m/z", zaxis_title="IM/CCS", aspectmode="cube"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_figure_2d(
    df: pd.DataFrame,
    mz_col: str,
    rt_col: str,
    color_values: np.ndarray,
    colorbar_title: str,
    max_points: int = 30000,
    title: str = "",
    point_size: int = 10,
    colorscale: str = "Plasma",
):
    n = len(df)
    if n == 0:
        return go.Figure().update_layout(template="plotly_dark", title=title)

    idx = np.arange(n)
    if n > max_points:
        w = np.abs(color_values)
        w = np.nan_to_num(w, nan=0.0)
        if w.sum() <= 0:
            take = np.random.choice(idx, size=max_points, replace=False)
        else:
            w = w / w.sum()
            take = np.random.choice(idx, size=max_points, replace=False, p=w)
    else:
        take = idx

    x = safe_numeric_series(df.iloc[take][rt_col]).to_numpy()
    y = safe_numeric_series(df.iloc[take][mz_col]).to_numpy()
    c = color_values[take]

    cd = df.iloc[take]["_global_index_"].to_numpy() if "_global_index_" in df.columns else df.index.values[take]

    fig = go.Figure(
        data=[
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=c,
                    colorscale=colorscale,
                    opacity=0.85,
                    showscale=True,
                    colorbar=dict(title=colorbar_title),
                ),
                customdata=cd,
                text=[f"row_index={i}" for i in cd],
                hovertemplate="%{text}<br>color=%{marker.color:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="RT",
        yaxis_title="m/z",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def add_highlight_trace_to_cmp(fig, selected_rows_df, mz_col, rt_col, z_col, plot_dim):
    if selected_rows_df.empty:
        return fig

    if plot_dim == "2d":
        fig.add_trace(
            go.Scattergl(
                x=selected_rows_df[rt_col],
                y=selected_rows_df[mz_col],
                mode="markers",
                marker=dict(size=16, color="yellow", opacity=0.95, symbol="circle-open"),
                name="Selected interpretation group",
                customdata=selected_rows_df["_global_index_"],
                text=[f"row_index={i}" for i in selected_rows_df["_global_index_"]],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=selected_rows_df[rt_col],
                y=selected_rows_df[mz_col],
                z=selected_rows_df[z_col],
                mode="markers",
                marker=dict(size=8, color="yellow", opacity=0.95),
                name="Selected interpretation group",
                customdata=selected_rows_df["_global_index_"],
                hovertext=[f"row_index={i}" for i in selected_rows_df["_global_index_"]],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )
    return fig


def add_highlight_trace_to_kendrick(fig, selected_rows_df):
    if selected_rows_df.empty:
        return fig
    fig.add_trace(
        go.Scattergl(
            x=selected_rows_df["KM"],
            y=selected_rows_df["KMD"],
            mode="markers",
            marker=dict(size=14, color="yellow", opacity=0.95, symbol="circle-open"),
            name="Selected interpretation group",
            customdata=selected_rows_df["_global_index_"],
            text=[f"row_index={i}" for i in selected_rows_df["_global_index_"]],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    return fig


# -----------------------------
# App + Layout
# -----------------------------
app = Dash(__name__)
app.title = "Feature 3D Viewer"

LABEL_STYLE = {"color": "#f0f0f0", "fontWeight": 600}
INPUT_STYLE = {
    "width": "100%",
    "backgroundColor": "#111",
    "color": "#f0f0f0",
    "caretColor": "#f0f0f0",
    "border": "1px solid #333",
    "fontSize": "16px",
    "lineHeight": "22px",
    "padding": "12px 12px",
    "paddingRight": "3.8em",
    "minHeight": "52px",
    "boxSizing": "border-box",
}
PANEL_STYLE = {
    "background": "#000",
    "color": "#f0f0f0",
    "padding": "10px",
    "minHeight": 0,
    "display": "flex",
    "flexDirection": "column",
    "gap": "10px",
}
CENTER_STYLE = {
    "flex": 1,
    "padding": "10px",
    "minHeight": 0,
    "display": "flex",
    "flexDirection": "column",
    "background": "#000",
}
RADIO_LABEL_STYLE = {"color": "#f0f0f0"}
RADIO_INPUT_STYLE = {"marginRight": "6px"}

TABLE_STYLE = {
    "backgroundColor": "#111",
    "color": "#f0f0f0",
    "border": "1px solid #333",
    "fontSize": "12px",
}
TABLE_CELL_STYLE = {
    "backgroundColor": "#111",
    "color": "#f0f0f0",
    "border": "1px solid #333",
    "whiteSpace": "normal",
    "height": "auto",
    "textAlign": "left",
    "minWidth": "80px",
    "maxWidth": "180px",
}


def home_view():
    return html.Div(
        style={"height": "calc(100vh - 140px)", "display": "flex", "minHeight": 0},
        children=[
            html.Div(
                style={**PANEL_STYLE, "width": "24%", "borderRight": "1px solid #333"},
                children=[
                    html.Div("Options", style={"fontWeight": 800, "color": "#fff"}),

                    html.Label("Dataset type", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="dataset_type_dropdown",
                        options=[
                            {"label": "Metabolomics", "value": "metabolomics"},
                            {"label": "Lipidomics", "value": "lipidomics"},
                            {"label": "Proteomics", "value": "proteomics"},
                        ],
                        value="metabolomics",
                        clearable=False,
                        style={"backgroundColor": "#111", "color": "#000"},
                    ),

                    html.Label("View (sample or condition mean)", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="intensity_dropdown",
                        options=[],
                        value=None,
                        clearable=False,
                        style={"backgroundColor": "#111", "color": "#000"},
                    ),

                    html.Label("Min intensity", style=LABEL_STYLE),
                    dcc.Input(id="min_intensity", type="number", value=0, step=1000, style=INPUT_STYLE),

                    html.Label("Max points (downsample)", style=LABEL_STYLE),
                    dcc.Input(id="max_points", type="number", value=30000, step=5000, style=INPUT_STYLE),

                    dcc.Checklist(
                        id="equalize_axes",
                        options=[{"label": "Scale axis ranges (better visualization)", "value": "yes"}],
                        value=[],
                        style={"color": "#f0f0f0"},
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"color": "#f0f0f0"},
                    ),
                ],
            ),

            html.Div(
                style=CENTER_STYLE,
                children=[
                    dcc.Graph(
                        id="home_graph",
                        style={"flex": 1, "minHeight": 0},
                        config={"displayModeBar": True},
                        figure=go.Figure().update_layout(
                            template="plotly_dark",
                            title="Load a feature file to begin",
                            margin=dict(l=0, r=0, t=40, b=0),
                        ),
                    ),
                    html.Div(id="home_click_info", style={"fontFamily": "monospace", "fontSize": 12, "opacity": 0.9}),
                ],
            ),

            html.Div(
                style={**PANEL_STYLE, "width": "24%", "borderLeft": "1px solid #333"},
                children=[
                    html.Div("Selected feature", style={"fontWeight": 800, "color": "#fff"}),
                    html.Pre(
                        id="home_selected_feature_box",
                        style={"whiteSpace": "pre-wrap", "fontSize": 12, "color": "#ddd"},
                        children="Click a point in the Home view to select a feature.",
                    ),
                ],
            ),
        ],
    )


def comparison_view():
    return html.Div(
        style={"height": "calc(100vh - 140px)", "display": "flex", "minHeight": 0},
        children=[
            html.Div(
                style={**PANEL_STYLE, "width": "20%", "borderRight": "1px solid #333", "overflowY": "auto", "height": "100%"},
                children=[
                    html.Div("Comparison", style={"fontWeight": 800, "color": "#fff"}),

                    html.Label("Condition 1", style=LABEL_STYLE),
                    dcc.Dropdown(id="cmp_cond1", options=[], value=None, clearable=False, style={"backgroundColor": "#111", "color": "#000"}),

                    html.Label("Condition 2", style=LABEL_STYLE),
                    dcc.Dropdown(id="cmp_cond2", options=[], value=None, clearable=False, style={"backgroundColor": "#111", "color": "#000"}),

                    html.Label("Mode", style=LABEL_STYLE),
                    dcc.RadioItems(
                        id="cmp_mode",
                        options=[
                            {"label": "Absent / present", "value": "present_absent"},
                            {"label": "Differentially expressed", "value": "de"},
                        ],
                        value="present_absent",
                        inputStyle=RADIO_INPUT_STYLE,
                        labelStyle=RADIO_LABEL_STYLE,
                    ),

                    html.Label("Plot type", style=LABEL_STYLE),
                    dcc.RadioItems(
                        id="cmp_plot_dim",
                        options=[
                            {"label": "3D (RT, m/z, IM/CCS)", "value": "3d"},
                            {"label": "2D (RT vs m/z)", "value": "2d"},
                        ],
                        value="3d",
                        inputStyle=RADIO_INPUT_STYLE,
                        labelStyle=RADIO_LABEL_STYLE,
                    ),

                    html.Hr(style={"borderColor": "#333"}),

                    html.Div("Detection thresholds (used by Absent/Present AND as filters for DE)", style={"fontWeight": 700, "color": "#fff"}),

                    html.Label("% missing to call 'undetected' (per condition)", style=LABEL_STYLE),
                    dcc.Input(id="cmp_undetected_pct", type="number", value=50, min=0, max=100, step=5, style=INPUT_STYLE),

                    html.Label("% detected to call 'detected/expressed' (per condition)", style=LABEL_STYLE),
                    dcc.Input(id="cmp_detected_pct", type="number", value=50, min=0, max=100, step=5, style=INPUT_STYLE),

                    html.Hr(style={"borderColor": "#333"}),

                    html.Div("Gapfilling (DE only)", style={"fontWeight": 700, "color": "#fff"}),
                    dcc.Checklist(
                        id="de_gapfill_enable",
                        options=[{"label": "Enable median gapfilling (per condition)", "value": "yes"}],
                        value=["yes"],
                        style={"color": "#f0f0f0"},
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"color": "#f0f0f0"},
                    ),
                    dcc.Checklist(
                        id="de_zero_as_missing",
                        options=[{"label": "Treat zeros as missing (recommended)", "value": "yes"}],
                        value=["yes"],
                        style={"color": "#f0f0f0"},
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"color": "#f0f0f0"},
                    ),

                    html.Hr(style={"borderColor": "#333"}),

                    html.Div("DE thresholds (DE only)", style={"fontWeight": 700, "color": "#fff"}),
                    html.Label("p-value threshold", style=LABEL_STYLE),
                    dcc.Input(id="de_p_thresh", type="number", value=0.05, min=0, max=1, step=0.005, style=INPUT_STYLE),

                    html.Label("abs(log2FC) threshold", style=LABEL_STYLE),
                    dcc.Input(id="de_fc_thresh", type="number", value=1.0, min=0, step=0.005, style=INPUT_STYLE),

                    html.Label("Test method", style=LABEL_STYLE),
                    dcc.RadioItems(
                        id="de_test_method",
                        options=[
                            {"label": "Welch t-test (Python)", "value": "ttest"},
                            {"label": "limma (R) (optional)", "value": "limma"},
                        ],
                        value="ttest",
                        inputStyle=RADIO_INPUT_STYLE,
                        labelStyle=RADIO_LABEL_STYLE,
                    ),

                    html.Label("DE display", style=LABEL_STYLE),
                    dcc.RadioItems(
                        id="de_show_sig",
                        options=[
                            {"label": "Show all", "value": "all"},
                            {"label": "Only significant", "value": "sig"},
                        ],
                        value="all",
                        inputStyle=RADIO_INPUT_STYLE,
                        labelStyle=RADIO_LABEL_STYLE,
                    ),

                    html.Button(
                        "Analyze",
                        id="cmp_analyze_btn",
                        n_clicks=0,
                        style={"marginTop": "6px", "padding": "8px 10px", "background": "#1f6feb", "color": "white", "border": "0", "borderRadius": "6px", "cursor": "pointer"},
                    ),

                    html.Button(
                        "Export features",
                        id="cmp_export_btn",
                        n_clicks=0,
                        style={"padding": "8px 10px", "background": "#238636", "color": "white", "border": "0", "borderRadius": "6px", "cursor": "pointer"},
                    ),

                    html.Div(id="cmp_run_status", style={"fontSize": 12, "opacity": 0.9}),
                ],
            ),

            html.Div(
                style=CENTER_STYLE,
                children=[
                    dcc.Graph(
                        id="cmp_graph",
                        style={"flex": 1, "minHeight": 0},
                        config={"displayModeBar": True},
                        figure=go.Figure().update_layout(
                            template="plotly_dark",
                            title="Click Analyze to view results",
                            margin=dict(l=0, r=0, t=40, b=0),
                        ),
                    ),
                    html.Div(id="cmp_plot_note", style={"fontFamily": "monospace", "fontSize": 12, "opacity": 0.9}),
                ],
            ),

            html.Div(
                style={**PANEL_STYLE, "width": "30%", "borderLeft": "1px solid #333", "overflowY": "auto"},
                children=[
                    html.Div("Results summary", style={"fontWeight": 800, "color": "#fff"}),
                    html.Div(
                        id="cmp_summary",
                        style={"fontFamily": "monospace", "whiteSpace": "pre-wrap", "marginTop": "6px"},
                        children="Load features + annotations, choose conditions, then click Analyze.",
                    ),

                    html.Hr(style={"borderColor": "#333"}),

                    html.Div("Selected feature (Comparison)", style={"fontWeight": 800, "color": "#fff"}),
                    html.Pre(
                        id="cmp_selected_feature_box",
                        style={"whiteSpace": "pre-wrap", "fontSize": 12, "color": "#ddd"},
                        children="Click a point in the Comparison view to see details here.",
                    ),

                    html.Hr(style={"borderColor": "#333"}),

                    html.Div(
                        id="kendrick_wrap",
                        style={"display": "none"},
                        children=[
                            html.Div("Kendrick plot", style={"fontWeight": 800, "color": "#fff"}),

                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
                                children=[
                                    html.Div(
                                        children=[
                                            html.Label("Mass tolerance (ppm)", style=LABEL_STYLE),
                                            dcc.Input(id="kendrick_ppm_tol", type="number", value=10, min=0, step=1, style=INPUT_STYLE),
                                        ]
                                    ),
                                    html.Div(
                                        children=[
                                            html.Label("RT window (min)", style=LABEL_STYLE),
                                            dcc.Input(id="kendrick_rt_window", type="number", value=0.30, min=0, step=0.01, style=INPUT_STYLE),
                                        ]
                                    ),
                                ],
                            ),

                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
                                children=[
                                    html.Div(
                                        children=[
                                            html.Label("KMD tolerance", style=LABEL_STYLE),
                                            dcc.Input(id="kendrick_kmd_tol", type="number", value=0.01, min=0, step=0.0001, style=INPUT_STYLE),
                                        ]
                                    ),
                                    html.Div(
                                        children=[
                                            html.Label("Highlight group", style=LABEL_STYLE),
                                            dcc.Dropdown(
                                                id="kendrick_group_select",
                                                options=[],
                                                value=None,
                                                clearable=True,
                                                placeholder="Select detected group",
                                                style={"backgroundColor": "#111", "color": "#000"},
                                            ),
                                        ]
                                    ),
                                ],
                            ),

                            dcc.Checklist(
                                id="kendrick_options",
                                options=[
                                    {"label": "Require RT coherence for homolog groups", "value": "require_rt"},
                                    {"label": "Require same best ion for homolog groups", "value": "same_ion"},
                                    {"label": "Use adduct grouping", "value": "use_adduct"},
                                ],
                                value=["use_adduct"],
                                style={"color": "#f0f0f0"},
                                inputStyle={"marginRight": "6px"},
                                labelStyle={"display": "block", "color": "#f0f0f0"},
                            ),

                            dcc.Graph(
                                id="kendrick_graph",
                                style={"height": "320px"},
                                config={"displayModeBar": True},
                                figure=build_empty_kendrick_figure("Kendrick plot"),
                            ),
                            html.Div(id="kendrick_note", style={"fontFamily": "monospace", "fontSize": 12, "opacity": 0.9}),

                            html.Div("Ion-pattern summary", style={"fontWeight": 800, "color": "#fff", "marginTop": "8px"}),
                            html.Pre(
                                id="kendrick_summary_box",
                                style={"whiteSpace": "pre-wrap", "fontSize": 12, "color": "#ddd", "maxHeight": "220px", "overflowY": "auto"},
                            ),

                            html.Div("Homologous groups", style={"fontWeight": 800, "color": "#fff"}),
                            dash_table.DataTable(
                                id="kendrick_homolog_table",
                                columns=[],
                                data=[],
                                style_table={"overflowX": "auto", "maxHeight": "240px", "overflowY": "auto"},
                                style_header=TABLE_STYLE,
                                style_cell=TABLE_CELL_STYLE,
                                sort_action="native",
                            ),

                            html.Div("Adduct groups", style={"fontWeight": 800, "color": "#fff"}),
                            dash_table.DataTable(
                                id="kendrick_adduct_table",
                                columns=[],
                                data=[],
                                style_table={"overflowX": "auto", "maxHeight": "220px", "overflowY": "auto"},
                                style_header=TABLE_STYLE,
                                style_cell=TABLE_CELL_STYLE,
                                sort_action="native",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


app.layout = html.Div(
    style={"height": "100vh", "display": "flex", "flexDirection": "column"},
    children=[
        html.Div("✅ Dash UI loaded", style={"padding": "6px 10px", "background": "#222", "color": "white"}),

        html.Div(
            style={"padding": "8px 12px", "display": "flex", "gap": "12px", "alignItems": "center", "borderBottom": "1px solid #333", "background": "#111", "color": "#f0f0f0"},
            children=[
                html.Div("Feature App", style={"fontWeight": 700}),
                html.Div("Features:", style={"opacity": 0.9}),
                dcc.Upload(id="upload_features", children=html.Button("Browse…"), multiple=False),
                html.Div(id="features_name", style={"flex": 1, "opacity": 0.9}),
                html.Div("Annotations:", style={"opacity": 0.9}),
                dcc.Upload(id="upload_ann", children=html.Button("Browse…"), multiple=False),
                html.Div(id="ann_name", style={"flex": 1, "opacity": 0.9}),
                html.Div(id="load_status", style={"opacity": 0.9}),
            ],
        ),

        dcc.Tabs(
            id="tabs",
            value="tab-home",
            children=[dcc.Tab(label="Home", value="tab-home"), dcc.Tab(label="Comparison", value="tab-comparison")],
        ),

        html.Div(
            style={"flex": 1, "minHeight": 0, "background": "#000"},
            children=[
                html.Div(id="home_wrap", children=[home_view()], style={"display": "block", "height": "100%"}),
                html.Div(id="cmp_wrap", children=[comparison_view()], style={"display": "none", "height": "100%"}),
            ],
        ),

        dcc.Store(id="df_store"),
        dcc.Store(id="meta_store"),
        dcc.Store(id="cond_map_store"),
        dcc.Store(id="sample_meta_store"),

        dcc.Store(id="cmp_result_store"),
        dcc.Store(id="de_store"),
        dcc.Store(id="de_gapfill_store"),
        dcc.Store(id="de_summary_store"),
        dcc.Store(id="lipid_interp_store"),

        dcc.Download(id="cmp_download_features"),
        dcc.Download(id="cmp_download_summary"),
        dcc.Download(id="de_download_gapfill_matrix"),
        dcc.Download(id="de_download_gapfill_summary"),
        dcc.Download(id="de_download_results"),
        dcc.Download(id="de_download_summary"),
    ],
)


# -----------------------------
# Tab toggle
# -----------------------------
@app.callback(
    Output("home_wrap", "style"),
    Output("cmp_wrap", "style"),
    Input("tabs", "value"),
)
def toggle_tabs(tab):
    if tab == "tab-comparison":
        return {"display": "none", "height": "100%"}, {"display": "block", "height": "100%"}
    return {"display": "block", "height": "100%"}, {"display": "none", "height": "100%"}


# -----------------------------
# Load files + compute mappings
# -----------------------------
@app.callback(
    Output("df_store", "data"),
    Output("meta_store", "data"),
    Output("cond_map_store", "data"),
    Output("sample_meta_store", "data"),
    Output("load_status", "children"),
    Output("intensity_dropdown", "options"),
    Output("intensity_dropdown", "value"),
    Output("features_name", "children"),
    Output("ann_name", "children"),
    Input("upload_features", "contents"),
    State("upload_features", "filename"),
    Input("upload_ann", "contents"),
    State("upload_ann", "filename"),
    prevent_initial_call=True,
)
def load_files(features_contents, features_filename, ann_contents, ann_filename):
    if not features_contents:
        return None, None, None, None, "Select a feature file.", [], None, "", ""

    df = parse_uploaded(features_contents, features_filename)
    mz_col, rt_col, z_col = detect_axes(df)

    for c in [mz_col, rt_col, z_col]:
        df[c] = safe_numeric_series(df[c])
    df = df.dropna(subset=[mz_col, rt_col, z_col]).copy()

    sample_cols = get_sample_cols(df)
    if not sample_cols:
        return None, None, None, None, "No '* Peak area' columns found in feature file.", [], None, f"{features_filename}", f"{ann_filename or ''}"

    cond_map = {}
    sample_meta_map = {}
    condition_mean_cols = []
    ann_status = "no annotations"

    if ann_contents:
        ann = parse_uploaded(ann_contents, ann_filename)
        sid_col = pick_annotation_sample_id_col(ann)
        if sid_col is None:
            ann_status = "annotation: could not find sample-id column (expected Files/Sample_name/etc.)"
        elif "Condition" not in ann.columns:
            ann_status = "annotation: missing 'Condition' column"
        else:
            ann = ann.copy()
            ann[sid_col] = ann[sid_col].astype(str).str.strip()
            ann["Condition"] = ann["Condition"].astype(str).str.strip()

            if "Replicate" not in ann.columns:
                ann["Replicate"] = np.nan
            ann["Replicate"] = pd.to_numeric(ann["Replicate"], errors="coerce")

            sample_to_cond = dict(zip(ann[sid_col], ann["Condition"]))
            sample_to_rep = dict(zip(ann[sid_col], ann["Replicate"]))

            ann_status = f"annotations loaded ({len(ann)} rows; sample id='{sid_col}')"

            peak_to_sample = {col: sample_name_from_peak_area_col(col) for col in sample_cols}
            for peak_col, sample_name in peak_to_sample.items():
                cond = sample_to_cond.get(sample_name, None)
                if cond is None:
                    continue
                cond_map.setdefault(cond, []).append(peak_col)

                rep = sample_to_rep.get(sample_name, np.nan)
                rep_str = f"rep{int(rep)}" if pd.notna(rep) else "repNA"
                sample_meta_map[peak_col] = {"condition": cond, "replicate": rep, "new_name": f"{cond}_{rep_str}"}

            for cond, cols in cond_map.items():
                if not cols:
                    continue
                new_col = f"COND__{cond} (mean Peak area)"
                df[new_col] = safe_numeric_frame(df[cols]).mean(axis=1, skipna=True)
                condition_mean_cols.append(new_col)

    sample_options = [{"label": f"[Sample] {sample_name_from_peak_area_col(c)}", "value": c} for c in sample_cols]
    mean_options = [{"label": f"[Condition mean] {c.replace('COND__','')}", "value": c} for c in condition_mean_cols]
    options = sample_options + mean_options
    default_value = sample_cols[0] if sample_cols else None

    meta = {
        "mz_col": mz_col,
        "rt_col": rt_col,
        "z_col": z_col,
        "nrows": int(len(df)),
        "features_filename": features_filename,
        "ann_filename": ann_filename,
        "conditions": sorted(list(cond_map.keys())),
        "sample_cols": sample_cols,
    }

    store_json = df.to_json(date_format="iso", orient="split")
    status = f"Loaded features ({len(df)} rows); {len(sample_cols)} samples; {ann_status}; {len(condition_mean_cols)} condition-means."
    return store_json, meta, cond_map, sample_meta_map, status, options, default_value, f"{features_filename}", f"{ann_filename or '(none)'}"


# -----------------------------
# Home plot
# -----------------------------
@app.callback(
    Output("home_graph", "figure"),
    Input("df_store", "data"),
    Input("meta_store", "data"),
    Input("intensity_dropdown", "value"),
    Input("min_intensity", "value"),
    Input("max_points", "value"),
    prevent_initial_call=False,
)
def update_home_plot(df_json, meta, intensity_col, min_intensity, max_points):
    if not df_json or not meta or not intensity_col:
        return go.Figure().update_layout(template="plotly_dark", title="Load a feature file to begin")

    df = pd.read_json(df_json, orient="split")
    mz_col, rt_col, z_col = meta["mz_col"], meta["rt_col"], meta["z_col"]

    inten = safe_numeric_series(df[intensity_col]).fillna(0.0).to_numpy()
    color = np.log1p(inten)
    return build_figure_3d(
        df=df,
        mz_col=mz_col,
        rt_col=rt_col,
        z_col=z_col,
        color_values=color,
        colorbar_title="log1p(Intensity)",
        max_points=int(max_points or 30000),
        title=f"Home: {intensity_col}",
        point_size=4,
        colorscale="Plasma",
    )


@app.callback(
    Output("home_selected_feature_box", "children"),
    Output("home_click_info", "children"),
    Input("home_graph", "clickData"),
    State("df_store", "data"),
    State("meta_store", "data"),
    State("intensity_dropdown", "value"),
    State("dataset_type_dropdown", "value"),
    prevent_initial_call=True,
)
def on_home_click(clickData, df_json, meta, intensity_col, dataset_type):
    if not clickData or not df_json or not meta or not intensity_col:
        return no_update, ""

    df = pd.read_json(df_json, orient="split")
    mz_col, rt_col, z_col = meta["mz_col"], meta["rt_col"], meta["z_col"]

    row_idx = extract_row_index(clickData)
    if row_idx is None or row_idx < 0 or row_idx >= len(df):
        return no_update, "Click captured but no valid row index extracted."

    row = df.iloc[row_idx]
    inten = float(safe_numeric_series(pd.Series([row[intensity_col]])).fillna(0).iloc[0])

    txt = (
        f"dataset_type: {dataset_type}\n"
        f"row_index: {row_idx}\n"
        f"row ID: {row.get('row ID','')}\n"
        f"RT: {row[rt_col]:.4f}\n"
        f"m/z: {row[mz_col]:.6f}\n"
        f"IM/CCS: {row[z_col]:.4f}\n"
        f"view: {intensity_col}\n"
        f"Intensity: {inten:.3g}\n"
        f"best ion: {row.get('best ion','')}\n"
    )
    return txt, f"Selected row_index={row_idx}"


# -----------------------------
# Populate comparison condition dropdowns
# -----------------------------
@app.callback(
    Output("cmp_cond1", "options"),
    Output("cmp_cond2", "options"),
    Output("cmp_cond1", "value"),
    Output("cmp_cond2", "value"),
    Input("meta_store", "data"),
)
def populate_cmp_conditions(meta):
    if not meta or not meta.get("conditions"):
        return [], [], None, None
    conds = meta["conditions"]
    opts = [{"label": c, "value": c} for c in conds]
    v1 = conds[0] if len(conds) > 0 else None
    v2 = conds[1] if len(conds) > 1 else v1
    return opts, opts, v1, v2


# -----------------------------
# Analyze comparison
# -----------------------------
@app.callback(
    Output("cmp_result_store", "data"),
    Output("de_store", "data"),
    Output("de_gapfill_store", "data"),
    Output("de_summary_store", "data"),
    Output("cmp_summary", "children"),
    Output("cmp_run_status", "children"),
    Input("cmp_analyze_btn", "n_clicks"),
    State("df_store", "data"),
    State("meta_store", "data"),
    State("cond_map_store", "data"),
    State("sample_meta_store", "data"),
    State("cmp_cond1", "value"),
    State("cmp_cond2", "value"),
    State("cmp_mode", "value"),
    State("cmp_undetected_pct", "value"),
    State("cmp_detected_pct", "value"),
    State("de_gapfill_enable", "value"),
    State("de_zero_as_missing", "value"),
    State("de_p_thresh", "value"),
    State("de_fc_thresh", "value"),
    State("de_test_method", "value"),
    State("dataset_type_dropdown", "value"),
    prevent_initial_call=True,
)
def run_comparison(
    n_clicks,
    df_json,
    meta,
    cond_map,
    sample_meta_map,
    c1,
    c2,
    mode,
    undetected_pct,
    detected_pct,
    gapfill_enable,
    zero_as_missing,
    p_thresh,
    fc_thresh,
    test_method,
    dataset_type,
):
    if not df_json or not meta or not cond_map or not c1 or not c2:
        return None, None, None, None, "Load features + annotations, choose conditions, then click Analyze.", ""

    df = pd.read_json(df_json, orient="split")
    mz_col, rt_col, z_col = meta["mz_col"], meta["rt_col"], meta["z_col"]

    cols1 = cond_map.get(c1, [])
    cols2 = cond_map.get(c2, [])
    if not cols1 or not cols2:
        msg = f"Missing sample columns for one/both conditions. c1={len(cols1)} cols, c2={len(cols2)} cols."
        return None, None, None, None, msg, ""

    undetected_pct = float(undetected_pct or 0)
    detected_pct = float(detected_pct or 0)

    X1 = safe_numeric_frame(df[cols1]).to_numpy()
    X2 = safe_numeric_frame(df[cols2]).to_numpy()

    det1 = (np.nan_to_num(X1, nan=0.0) > 0).mean(axis=1) * 100.0
    det2 = (np.nan_to_num(X2, nan=0.0) > 0).mean(axis=1) * 100.0
    miss1 = 100.0 - det1
    miss2 = 100.0 - det2

    und1 = miss1 >= undetected_pct
    und2 = miss2 >= undetected_pct
    det_call1 = det1 >= detected_pct
    det_call2 = det2 >= detected_pct

    if mode == "present_absent":
        only_c1 = det_call1 & und2
        only_c2 = det_call2 & und1
        both = det_call1 & det_call2
        neither = und1 & und2

        viz_mask = only_c1 | only_c2
        cats = np.full(len(df), "other", dtype=object)
        cats[only_c1] = f"present_in_{c1}_only"
        cats[only_c2] = f"present_in_{c2}_only"

        label = f"Absent/Present: {c1} vs {c2} (only-in-either)"
        summary = (
            f"Dataset type: {dataset_type}\n"
            f"Comparison: {c1} vs {c2}\n"
            f"Mode: Absent/Present\n"
            f"Undetected threshold: missing ≥ {undetected_pct:.1f}%\n"
            f"Detected threshold: detected ≥ {detected_pct:.1f}%\n\n"
            f"Counts (features):\n"
            f"  Present in {c1} / Undetected in {c2}: {int(only_c1.sum())}\n"
            f"  Present in {c2} / Undetected in {c1}: {int(only_c2.sum())}\n"
            f"  Present in both: {int(both.sum())}\n"
            f"  Undetected in both: {int(neither.sum())}\n\n"
            f"Replicates per condition:\n"
            f"  {c1}: {len(cols1)} samples\n"
            f"  {c2}: {len(cols2)} samples\n"
        )

        viz_idx = np.where(viz_mask)[0].tolist()
        viz_cat = [cats[i] for i in viz_idx]

        return (
            {
                "mode": mode,
                "label": label,
                "row_indices": viz_idx,
                "row_categories": viz_cat,
                "cond1": c1,
                "cond2": c2,
                "dataset_type": dataset_type,
            },
            None,
            None,
            None,
            summary,
            f"Analyze complete (n={n_clicks}). Visualizing {int(viz_mask.sum())} features.",
        )

    p_thresh = float(p_thresh or 0.05)
    fc_thresh = float(fc_thresh or 0.0)

    detected_both = det_call1 & det_call2
    idx_keep = np.where(detected_both)[0]

    if idx_keep.size == 0:
        summary = (
            f"Dataset type: {dataset_type}\n"
            f"Comparison: {c1} vs {c2}\n"
            f"Mode: DE\n"
            f"No features passed 'detected in both' filter (detected ≥ {detected_pct:.1f}% in each condition).\n"
        )
        return (
            {"mode": "de", "label": "DE: none", "row_indices": [], "cond1": c1, "cond2": c2, "dataset_type": dataset_type},
            None, None, summary, summary, "No DE features to analyze.",
        )

    A = X1[idx_keep, :].copy()
    B = X2[idx_keep, :].copy()

    if "yes" in (zero_as_missing or []):
        A[A == 0] = np.nan
        B[B == 0] = np.nan

    gapfill_on = "yes" in (gapfill_enable or [])
    if gapfill_on:
        A_imp = median_gapfill_by_feature(A)
        B_imp = median_gapfill_by_feature(B)
    else:
        A_imp = A
        B_imp = B

    sample_meta_map = sample_meta_map or {}

    def rename_cols(cols, cond_fallback):
        names = []
        for c in cols:
            info = sample_meta_map.get(c)
            if info and info.get("new_name"):
                names.append(info["new_name"])
            else:
                names.append(sample_name_from_peak_area_col(c) if c.endswith(" Peak area") else f"{cond_fallback}_{c}")
        return names

    cols1_names = rename_cols(cols1, c1)
    cols2_names = rename_cols(cols2, c2)

    base_cols = [c for c in ["row ID", mz_col, rt_col, z_col, "best ion"] if c in df.columns]
    gapfill_df = df.iloc[idx_keep][base_cols].copy()
    for j, name in enumerate(cols1_names):
        gapfill_df[name] = A_imp[:, j]
    for j, name in enumerate(cols2_names):
        gapfill_df[name] = B_imp[:, j]

    mean1 = np.nanmean(A_imp, axis=1)
    mean2 = np.nanmean(B_imp, axis=1)
    eps = 1e-12
    log2fc = np.log2((mean1 + eps) / (mean2 + eps))

    if test_method == "limma" and _HAVE_RPY2:
        test_method_used = "ttest"
    else:
        test_method_used = "ttest"

    if test_method_used == "ttest":
        if _HAVE_SCIPY:
            pvals = np.full(len(idx_keep), np.nan, dtype=float)
            for i in range(len(idx_keep)):
                a = A_imp[i, :]
                b = B_imp[i, :]
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if a.size < 2 or b.size < 2:
                    continue
                pvals[i] = ttest_ind(a, b, equal_var=False).pvalue
        else:
            pvals = welch_ttest_fallback(A_imp, B_imp)
    else:
        pvals = np.full(len(idx_keep), np.nan, dtype=float)

    fdr = bh_fdr(pvals)
    sig = (np.isfinite(pvals) & (pvals <= p_thresh) & (np.abs(log2fc) >= fc_thresh))

    de_df = df.iloc[idx_keep][base_cols].copy()
    de_df["_global_index_"] = idx_keep
    de_df[f"mean_{c1}"] = mean1
    de_df[f"mean_{c2}"] = mean2
    de_df["log2FC"] = log2fc
    de_df["p_value"] = pvals
    de_df["FDR"] = fdr
    de_df["significant"] = sig

    de_summary = (
        f"Dataset type: {dataset_type}\n"
        f"Comparison: {c1} vs {c2}\n"
        f"Mode: DE\n"
        f"Detected-in-both filter: detected ≥ {detected_pct:.1f}% in each condition\n"
        f"Zero-as-missing: {'yes' if 'yes' in (zero_as_missing or []) else 'no'}\n"
        f"Gapfill: {'median per feature per condition' if gapfill_on else 'OFF'}\n"
        f"Test: {('Welch t-test (scipy)' if _HAVE_SCIPY else 'Welch t-test (fallback)')}\n"
        f"Thresholds: p ≤ {p_thresh}, abs(log2FC) ≥ {fc_thresh}\n"
        f"Features tested (detected in both): {len(idx_keep)}\n"
        f"Significant: {int(sig.sum())}\n"
    )

    gapfill_json = gapfill_df.to_json(date_format="iso", orient="split")
    de_json = de_df.to_json(date_format="iso", orient="split")

    cmp_result = {
        "mode": "de",
        "label": f"DE: {c1} vs {c2}",
        "cond1": c1,
        "cond2": c2,
        "row_indices": idx_keep.tolist(),
        "dataset_type": dataset_type,
    }

    return (
        cmp_result,
        de_json,
        gapfill_json,
        de_summary,
        de_summary,
        f"Analyze complete (n={n_clicks}). Tested {len(idx_keep)} features; significant {int(sig.sum())}.",
    )


# -----------------------------
# Lipid interpretation panel
# -----------------------------
@app.callback(
    Output("kendrick_wrap", "style"),
    Output("lipid_interp_store", "data"),
    Output("kendrick_summary_box", "children"),
    Output("kendrick_homolog_table", "columns"),
    Output("kendrick_homolog_table", "data"),
    Output("kendrick_adduct_table", "columns"),
    Output("kendrick_adduct_table", "data"),
    Output("kendrick_group_select", "options"),
    Output("kendrick_group_select", "value"),
    Input("cmp_result_store", "data"),
    Input("dataset_type_dropdown", "value"),
    Input("kendrick_ppm_tol", "value"),
    Input("kendrick_kmd_tol", "value"),
    Input("kendrick_rt_window", "value"),
    Input("kendrick_options", "value"),
    State("de_store", "data"),
    State("df_store", "data"),
    State("meta_store", "data"),
    State("kendrick_group_select", "value"),
)
def update_lipid_interpretation_panel(
    cmp_result,
    dataset_type,
    ppm_tol,
    kmd_tol,
    rt_window,
    kendrick_options,
    de_json,
    df_json,
    meta,
    current_group_value,
):
    empty_cols = []
    empty_data = []
    empty_store = {"analysis_rows": [], "homolog_groups": [], "adduct_groups": [], "unsat_pairs": [], "summary": ""}

    if dataset_type != "lipidomics" or not cmp_result or not df_json or not meta:
        return {"display": "none"}, empty_store, "", empty_cols, empty_data, empty_cols, empty_data, [], None

    ppm_tol = float(ppm_tol or DEFAULT_PPM_TOL)
    kmd_tol = float(kmd_tol or DEFAULT_KMD_TOL)
    rt_window = float(rt_window or DEFAULT_RT_WINDOW)
    require_rt = "require_rt" in (kendrick_options or [])
    require_same_ion = "same_ion" in (kendrick_options or [])
    use_adduct = "use_adduct" in (kendrick_options or [])

    dfk = build_kendrick_analysis_df(cmp_result, de_json, df_json, meta, dataset_type)
    if dfk.empty:
        return {"display": "block"}, empty_store, "No lipid features passed for Kendrick interpretation.", empty_cols, empty_data, empty_cols, empty_data, [], None

    mz_col = meta["mz_col"]
    rt_col = meta["rt_col"]

    homolog_groups = detect_homologous_series(
        dfk=dfk,
        mz_col=mz_col,
        rt_col=rt_col,
        ppm_tol=ppm_tol,
        kmd_tol=kmd_tol,
        require_same_ion=require_same_ion,
        require_rt=require_rt,
    )
    unsat_pairs = detect_unsaturation_pairs(dfk=dfk, mz_col=mz_col, rt_col=rt_col, ppm_tol=ppm_tol, rt_window=rt_window)
    adduct_groups = detect_adduct_groups(dfk=dfk, mz_col=mz_col, rt_col=rt_col, ppm_tol=ppm_tol, rt_window=rt_window) if use_adduct else []

    summary = build_interpretation_summary(dfk, homolog_groups, adduct_groups, unsat_pairs, cmp_result)

    homolog_rows = homolog_table_records(homolog_groups)
    adduct_rows = adduct_table_records(adduct_groups)

    homolog_cols = [{"name": c, "id": c} for c in (list(homolog_rows[0].keys()) if homolog_rows else [])]
    adduct_cols = [{"name": c, "id": c} for c in (list(adduct_rows[0].keys()) if adduct_rows else [])]

    group_options = []
    for g in homolog_groups:
        group_options.append({"label": f"{g['group_id']} | homolog | n={g['n_members']} | RT={g['rt_coherence']}", "value": g["group_id"]})
    for g in adduct_groups:
        group_options.append({"label": f"{g['group_id']} | adduct | n={g['n_members']} | {g['dominant_offset']}", "value": g["group_id"]})

    group_values = [x["value"] for x in group_options]
    selected_value = current_group_value if current_group_value in group_values else None

    store = {
        "analysis_rows": serialize_records(dfk.replace({np.nan: None}).to_dict("records")),
        "homolog_groups": serialize_records(homolog_groups),
        "adduct_groups": serialize_records(adduct_groups),
        "unsat_pairs": serialize_records(unsat_pairs),
        "summary": summary,
    }

    return {"display": "block"}, store, summary, homolog_cols, homolog_rows, adduct_cols, adduct_rows, group_options, selected_value


# -----------------------------
# Build comparison plot
# -----------------------------
@app.callback(
    Output("cmp_graph", "figure"),
    Output("cmp_plot_note", "children"),
    Input("cmp_result_store", "data"),
    Input("kendrick_group_select", "value"),
    State("de_store", "data"),
    State("df_store", "data"),
    State("meta_store", "data"),
    State("cmp_plot_dim", "value"),
    State("de_show_sig", "value"),
    State("lipid_interp_store", "data"),
)
def update_cmp_plot(result, selected_group_id, de_json, df_json, meta, cmp_plot_dim, de_show_sig, lipid_interp_store):
    if not result or not df_json or not meta:
        return go.Figure().update_layout(template="plotly_dark", title="Click Analyze to view results"), ""

    df = pd.read_json(df_json, orient="split")
    mz_col, rt_col, z_col = meta["mz_col"], meta["rt_col"], meta["z_col"]

    selected_members = set(get_selected_group_members(lipid_interp_store, selected_group_id))
    mode = result.get("mode", "present_absent")

    if mode == "de":
        if not de_json:
            return go.Figure().update_layout(template="plotly_dark", title="DE: no results yet"), ""

        de_df = pd.read_json(de_json, orient="split")

        if de_show_sig == "sig" and "significant" in de_df.columns:
            de_df = de_df[de_df["significant"] == True].copy()

        if de_df.empty:
            return (
                go.Figure().update_layout(template="plotly_dark", title="DE: no points to display (after filtering)"),
                "No DE points to display (after filtering).",
            )

        if "_global_index_" not in de_df.columns:
            return (
                go.Figure().update_layout(template="plotly_dark", title="DE: missing _global_index_"),
                "DE results missing _global_index_.",
            )

        color_vals = pd.to_numeric(de_df["log2FC"], errors="coerce").fillna(0.0).to_numpy()

        title = f"DE: {result.get('cond1')} vs {result.get('cond2')} (color=log2FC)"
        note = f"Mode=DE | n={len(de_df)} | view={'sig only' if de_show_sig == 'sig' else 'all'} | color=log2FC"

        if cmp_plot_dim == "2d":
            fig = build_figure_2d(
                df=de_df,
                mz_col=mz_col,
                rt_col=rt_col,
                color_values=color_vals,
                colorbar_title="log2FC",
                max_points=30000,
                title=title,
                point_size=10,
                colorscale="RdBu",
            )
        else:
            fig = build_figure_3d(
                df=de_df,
                mz_col=mz_col,
                rt_col=rt_col,
                z_col=z_col,
                color_values=color_vals,
                colorbar_title="log2FC",
                max_points=30000,
                title=title,
                point_size=4,
                colorscale="RdBu",
            )

        if selected_members:
            sel = de_df[de_df["_global_index_"].isin(selected_members)].copy()
            fig = add_highlight_trace_to_cmp(fig, sel, mz_col, rt_col, z_col, cmp_plot_dim)
            note += f" | highlighted={selected_group_id}"

        return fig, note

    idx = result.get("row_indices", [])
    cats = result.get("row_categories", [])
    c1 = result.get("cond1")
    c2 = result.get("cond2")

    if not idx:
        return go.Figure().update_layout(template="plotly_dark", title="No features to plot"), "No features under thresholds."

    sub = df.iloc[idx].reset_index(drop=True).copy()
    sub["_global_index_"] = np.array(idx)
    sub["_cat_"] = cats

    title = result.get("label", "Absent/Present")
    note = f"Mode=Absent/Present | n={len(sub)} | categorical (legend)"

    fig = go.Figure()
    for cat_label, trace_name, fixed_color in [
        (f"present_in_{c1}_only", f"Present in {c1} only", "#1f77b4"),
        (f"present_in_{c2}_only", f"Present in {c2} only", "#ff7f0e"),
    ]:
        s = sub[sub["_cat_"] == cat_label]
        if s.empty:
            continue

        if cmp_plot_dim == "2d":
            fig.add_trace(
                go.Scattergl(
                    x=s[rt_col],
                    y=s[mz_col],
                    mode="markers",
                    marker=dict(size=10, color=fixed_color, opacity=0.9),
                    name=trace_name,
                    customdata=s["_global_index_"],
                    text=[f"row_index={i}" for i in s["_global_index_"]],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=s[rt_col],
                    y=s[mz_col],
                    z=s[z_col],
                    mode="markers",
                    marker=dict(size=4, color=fixed_color, opacity=0.9),
                    name=trace_name,
                    customdata=s["_global_index_"],
                    hovertext=[f"row_index={i}" for i in s["_global_index_"]],
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )

    if selected_members:
        sel = sub[sub["_global_index_"].isin(selected_members)].copy()
        fig = add_highlight_trace_to_cmp(fig, sel, mz_col, rt_col, z_col, cmp_plot_dim)
        note += f" | highlighted={selected_group_id}"

    if cmp_plot_dim == "2d":
        fig.update_layout(
            template="plotly_dark",
            title=title,
            xaxis_title="RT",
            yaxis_title="m/z",
            legend=dict(x=1.02, y=1, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=120, t=40, b=0),
        )
    else:
        fig.update_layout(
            template="plotly_dark",
            title=title,
            scene=dict(xaxis_title="RT", yaxis_title="m/z", zaxis_title="IM/CCS", aspectmode="cube"),
            legend=dict(x=1.02, y=1, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=120, t=40, b=0),
        )

    return fig, note


# -----------------------------
# Kendrick plot callback
# -----------------------------
@app.callback(
    Output("kendrick_graph", "figure"),
    Output("kendrick_note", "children"),
    Input("lipid_interp_store", "data"),
    Input("kendrick_group_select", "value"),
    State("meta_store", "data"),
    State("cmp_result_store", "data"),
    State("dataset_type_dropdown", "value"),
)
def update_kendrick_plot(lipid_interp_store, selected_group_id, meta, cmp_result, dataset_type):
    empty_fig = build_empty_kendrick_figure("Kendrick plot")
    if dataset_type != "lipidomics" or not lipid_interp_store or not meta or not cmp_result:
        return empty_fig, ""

    parsed = parse_interp_store(lipid_interp_store)
    analysis_rows = parsed["analysis_rows"]
    if not analysis_rows:
        return build_empty_kendrick_figure("Kendrick plot: no lipid features"), "No lipid features available."

    dfk = pd.DataFrame(analysis_rows)
    if dfk.empty:
        return build_empty_kendrick_figure("Kendrick plot: no lipid features"), "No lipid features available."

    mode = cmp_result.get("mode", "present_absent")
    c1 = cmp_result.get("cond1", "")
    c2 = cmp_result.get("cond2", "")
    mz_col = meta["mz_col"]

    if mode == "de":
        hover_text = []
        for _, r in dfk.iterrows():
            txt = (
                f"row_index={int(r['_global_index_'])}"
                f"<br>row ID={r.get('row ID', '')}"
                f"<br>m/z={float(r[mz_col]):.6f}"
                f"<br>KM={float(r['KM']):.6f}"
                f"<br>KMD={float(r['KMD']):.6f}"
            )
            if pd.notna(r.get("log2FC")):
                txt += f"<br>log2FC={float(r['log2FC']):.3f}"
            hover_text.append(txt)

        fig = go.Figure(
            data=[
                go.Scattergl(
                    x=dfk["KM"],
                    y=dfk["KMD"],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=pd.to_numeric(dfk.get("log2FC", pd.Series(np.zeros(len(dfk)))), errors="coerce").fillna(0.0),
                        colorscale="RdBu",
                        opacity=0.9,
                        showscale=True,
                        colorbar=dict(title="log2FC"),
                    ),
                    customdata=dfk["_global_index_"],
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=hover_text,
                    name="Significant lipid features",
                )
            ]
        )
        title = f"Kendrick plot (significant DE lipids: {c1} vs {c2})"
        note = (
            f"Dataset type=lipidomics | source=significant DE features only | n={len(dfk)} | "
            f"base unit=CH₂ ({KENDRICK_BASE_NOMINAL:.4f}/{KENDRICK_BASE_EXACT:.5f})"
        )
    else:
        fig = go.Figure()
        for cat_value, trace_name, trace_color in [
            (f"present_in_{c1}_only", f"Present in {c1} only", "#1f77b4"),
            (f"present_in_{c2}_only", f"Present in {c2} only", "#ff7f0e"),
        ]:
            s = dfk[dfk["category"] == cat_value].copy()
            if s.empty:
                continue

            hover_text = []
            for _, r in s.iterrows():
                hover_text.append(
                    f"row_index={int(r['_global_index_'])}"
                    f"<br>row ID={r.get('row ID', '')}"
                    f"<br>m/z={float(r[mz_col]):.6f}"
                    f"<br>KM={float(r['KM']):.6f}"
                    f"<br>KMD={float(r['KMD']):.6f}"
                    f"<br>best ion={r.get('best ion', '')}"
                )

            fig.add_trace(
                go.Scattergl(
                    x=s["KM"],
                    y=s["KMD"],
                    mode="markers",
                    marker=dict(size=8, color=trace_color, opacity=0.9),
                    name=trace_name,
                    customdata=s["_global_index_"],
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=hover_text,
                )
            )

        title = f"Kendrick plot ({c1} vs {c2})"
        note = (
            f"Dataset type=lipidomics | source=present/absent features only | n={len(dfk)} | "
            f"base unit=CH₂ ({KENDRICK_BASE_NOMINAL:.4f}/{KENDRICK_BASE_EXACT:.5f})"
        )

    selected_members = set(get_selected_group_members(lipid_interp_store, selected_group_id))
    if selected_members:
        sel = dfk[dfk["_global_index_"].isin(selected_members)].copy()
        fig = add_highlight_trace_to_kendrick(fig, sel)
        note += f" | highlighted={selected_group_id}"

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Kendrick Mass (CH₂ base)",
        yaxis_title="Kendrick Mass Defect (KMD)",
        legend=dict(x=1.02, y=1, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=100, t=40, b=40),
    )
    return fig, note


# -----------------------------
# Comparison click info
# -----------------------------
@app.callback(
    Output("cmp_selected_feature_box", "children"),
    Input("cmp_graph", "clickData"),
    State("df_store", "data"),
    State("meta_store", "data"),
    State("cmp_result_store", "data"),
    State("de_store", "data"),
    State("dataset_type_dropdown", "value"),
    prevent_initial_call=True,
)
def on_cmp_click(clickData, df_json, meta, cmp_result, de_json, dataset_type):
    if not clickData or not df_json or not meta:
        return no_update

    df = pd.read_json(df_json, orient="split")
    mz_col, rt_col, z_col = meta["mz_col"], meta["rt_col"], meta["z_col"]

    row_idx = extract_row_index(clickData)
    if row_idx is None or row_idx < 0 or row_idx >= len(df):
        return "Click captured but no valid row index extracted."

    row = df.iloc[row_idx]

    base_txt = (
        f"dataset_type: {dataset_type}\n"
        f"row_index: {row_idx}\n"
        f"row ID: {row.get('row ID','')}\n"
        f"RT: {row[rt_col]:.4f}\n"
        f"m/z: {row[mz_col]:.6f}\n"
        f"IM/CCS: {row[z_col]:.4f}\n"
        f"best ion: {row.get('best ion','')}\n"
    )

    if dataset_type == "lipidomics":
        km, nominal_km, kmd = compute_kendrick_from_mz([row[mz_col]])
        if np.isfinite(km[0]):
            base_txt += (
                "\n--- Kendrick ---\n"
                f"KM: {float(km[0]):.6f}\n"
                f"Nominal KM: {float(nominal_km[0]):.0f}\n"
                f"KMD: {float(kmd[0]):.6f}\n"
            )

    if cmp_result and cmp_result.get("mode") == "de" and de_json:
        de_df = pd.read_json(de_json, orient="split")
        if "_global_index_" in de_df.columns:
            hit = de_df[de_df["_global_index_"] == row_idx]
            if not hit.empty:
                h = hit.iloc[0]
                base_txt += (
                    "\n--- DE stats ---\n"
                    f"log2FC (Cond1/Cond2): {float(h['log2FC']):.3f}\n"
                    f"p-value: {float(h['p_value']):.3g}\n"
                    f"FDR: {float(h['FDR']):.3g}\n"
                    f"significant: {bool(h['significant'])}\n"
                )

    return base_txt


# -----------------------------
# Export features
# -----------------------------
@app.callback(
    Output("cmp_download_features", "data"),
    Output("cmp_download_summary", "data"),
    Output("de_download_gapfill_matrix", "data"),
    Output("de_download_gapfill_summary", "data"),
    Output("de_download_results", "data"),
    Output("de_download_summary", "data"),
    Input("cmp_export_btn", "n_clicks"),
    State("cmp_result_store", "data"),
    State("df_store", "data"),
    State("meta_store", "data"),
    State("cond_map_store", "data"),
    State("sample_meta_store", "data"),
    State("cmp_summary", "children"),
    State("de_store", "data"),
    State("de_gapfill_store", "data"),
    State("de_summary_store", "data"),
    prevent_initial_call=True,
)
def export_cmp(
    n_clicks,
    result,
    df_json,
    meta,
    cond_map,
    sample_meta_map,
    summary_text,
    de_json,
    gapfill_json,
    de_summary_text,
):
    if not n_clicks or not result or not df_json or not meta:
        return no_update, no_update, no_update, no_update, no_update, no_update

    base = meta.get("features_filename", "features.csv")
    base_no_ext = os.path.splitext(base)[0]

    mode = result.get("mode", "present_absent")
    c1 = result.get("cond1")
    c2 = result.get("cond2")
    if not c1 or not c2:
        return no_update, no_update, no_update, no_update, no_update, no_update

    suffix = f"{sanitize_filename_part(c1)}_vs_{sanitize_filename_part(c2)}"

    if mode == "de":
        if not de_json or not gapfill_json:
            return no_update, no_update, no_update, no_update, no_update, no_update

        de_df = pd.read_json(de_json, orient="split")
        gapfill_df = pd.read_json(gapfill_json, orient="split")

        mode_tag = "DE"
        gapfill_csv_name = f"{base_no_ext}_{suffix}_{mode_tag}_GapFillMatrix.csv"
        gapfill_txt_name = f"{base_no_ext}_{suffix}_{mode_tag}_GapFillSummary.txt"
        de_csv_name = f"{base_no_ext}_{suffix}_{mode_tag}.csv"
        de_txt_name = f"{base_no_ext}_{suffix}_{mode_tag}_Summary.txt"

        if isinstance(de_summary_text, (list, tuple)):
            de_summary_text = "\n".join(str(x) for x in de_summary_text)
        if not de_summary_text:
            de_summary_text = summary_text
        de_summary_text = str(de_summary_text)

        return (
            no_update, no_update,
            dcc.send_data_frame(gapfill_df.to_csv, gapfill_csv_name, index=False),
            dict(content=de_summary_text, filename=gapfill_txt_name),
            dcc.send_data_frame(de_df.to_csv, de_csv_name, index=False),
            dict(content=de_summary_text, filename=de_txt_name),
        )

    if not cond_map:
        return no_update, no_update, no_update, no_update, no_update, no_update

    idx = result.get("row_indices", [])
    if not idx:
        return no_update, no_update, no_update, no_update, no_update, no_update

    df = pd.read_json(df_json, orient="split")
    subset = df.iloc[idx].copy()

    cols1 = cond_map.get(c1, [])
    cols2 = cond_map.get(c2, [])
    keep_intensity_cols = [c for c in (cols1 + cols2) if c in subset.columns]

    mz_col, rt_col, z_col = meta["mz_col"], meta["rt_col"], meta["z_col"]
    core_candidates = ["row ID", mz_col, rt_col, z_col, "best ion"]
    base_cols = [c for c in core_candidates if c in subset.columns]

    export_df = subset[base_cols + keep_intensity_cols].copy()

    if result.get("mode") == "present_absent":
        cat_map = dict(zip(result.get("row_indices", []), result.get("row_categories", [])))
        export_df["present_group"] = [cat_map.get(i, "") for i in idx]

    sample_meta_map = sample_meta_map or {}
    rename_map = {}
    for peak_col in keep_intensity_cols:
        info = sample_meta_map.get(peak_col)
        if info and info.get("new_name"):
            rename_map[peak_col] = info["new_name"]
        else:
            rename_map[peak_col] = sample_name_from_peak_area_col(peak_col)
    export_df = export_df.rename(columns=rename_map)

    mode_tag = "AbsentPresent"
    csv_name = f"{base_no_ext}_{suffix}_{mode_tag}.csv"
    summary_name = f"{base_no_ext}_{suffix}_{mode_tag}_Summary.txt"

    if isinstance(summary_text, (list, tuple)):
        summary_text = "\n".join(str(x) for x in summary_text)
    summary_text = str(summary_text)

    return (
        dcc.send_data_frame(export_df.to_csv, csv_name, index=False),
        dict(content=summary_text, filename=summary_name),
        no_update, no_update, no_update, no_update,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)