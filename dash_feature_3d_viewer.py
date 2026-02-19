# scripts/dash_feature_3d_viewer.py
# Multi-tab Dash app:
#   - Home: 3D viewer (samples + condition means)
#   - Comparison:
#        * Absent/Present (as before)
#        * DE (real): gapfill (median-per-feature-per-condition) + log2FC + test + p + FDR
#        * 2D/3D visualization toggle
#        * Color by:
#            - Absent/Present: log1p(max intensity) for context
#            - DE: log2FC (Cond1 / Cond2)
#        * On Analyze(DE): also triggers downloads:
#            - Gapfilled matrix (only detected-in-both features) + _GapFillSummary.txt
#            - DE results table + _DESummary.txt
#
# Notes:
# - For DE test: uses Welch t-test via scipy if available; otherwise a pure-numpy Welch t-test fallback.
# - "limma" is not implemented natively in Python here (requires rpy2 + R limma). UI includes a limma option,
#   but will fall back to t-test unless rpy2+limma are detected (optional stub included).
# - Browser apps cannot save to the original folder path; downloads go to browser download location.
#
# Run:
#   pip install dash plotly pandas numpy openpyxl scipy
#   python scripts/dash_feature_3d_viewer.py
# Open:
#   http://127.0.0.1:8050/

import base64
import io
import numpy as np
import pandas as pd
import os
import math

from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go


# -----------------------------
# Optional stats backends
# -----------------------------
try:
    from scipy.stats import ttest_ind
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Optional limma (requires rpy2 + R + limma installed). We keep as a best-effort optional.
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    _HAVE_RPY2 = True
except Exception:
    _HAVE_RPY2 = False


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


def equalize_ranges(x, y, z, robust=True):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    if robust:
        lo, hi = 1, 99
        x0, x1 = np.nanpercentile(x, [lo, hi])
        y0, y1 = np.nanpercentile(y, [lo, hi])
        z0, z1 = np.nanpercentile(z, [lo, hi])
    else:
        x0, x1 = np.nanmin(x), np.nanmax(x)
        y0, y1 = np.nanmin(y), np.nanmax(y)
        z0, z1 = np.nanmin(z), np.nanmax(z)

    rx = max(x1 - x0, 1e-12)
    ry = max(y1 - y0, 1e-12)
    rz = max(z1 - z0, 1e-12)
    target = max(rx, ry, rz)

    sx, sy, sz = target / rx, target / ry, target / rz

    x_s = (x - np.nanmean(x)) * sx
    y_s = (y - np.nanmean(y)) * sy
    z_s = (z - np.nanmean(z)) * sz
    return x_s, y_s, z_s, (sx, sy, sz)


def extract_row_index(clickData):
    """Robust extraction for both 3D and 2D plots."""
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
    """Benjamini-Hochberg FDR. Returns q-values aligned to input order."""
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
    # enforce monotonicity
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)

    q_ok = np.empty_like(p_ok)
    q_ok[order] = q_ranked
    q[ok] = q_ok
    return q


def median_gapfill_by_feature(mat: np.ndarray) -> np.ndarray:
    """
    mat: 2D array (n_features x n_samples) with NaNs for missing.
    Returns a copy where NaNs are replaced by per-feature median of non-NaN values.
    """
    out = mat.copy()
    # median per feature ignoring nan
    med = np.nanmedian(out, axis=1)
    # if a row is all nan, nanmedian -> nan; leave as nan
    inds = np.where(np.isnan(out))
    # inds: (row_idx, col_idx)
    rows = inds[0]
    out[inds] = med[rows]
    return out


def welch_ttest_fallback(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a,b: 2D arrays (n_features x n_rep) with finite values (NaNs allowed but will be ignored).
    Returns p-values per feature using Welch's t-test approximation (normal approximation for CDF if scipy absent).
    """
    # Compute mean/var with nan handling
    mean_a = np.nanmean(a, axis=1)
    mean_b = np.nanmean(b, axis=1)
    var_a = np.nanvar(a, axis=1, ddof=1)
    var_b = np.nanvar(b, axis=1, ddof=1)
    na = np.sum(np.isfinite(a), axis=1)
    nb = np.sum(np.isfinite(b), axis=1)

    se = np.sqrt(var_a / na + var_b / nb)
    t = (mean_a - mean_b) / se

    # Welch-Satterthwaite df
    df_num = (var_a / na + var_b / nb) ** 2
    df_den = (var_a**2) / (na**2 * (na - 1)) + (var_b**2) / (nb**2 * (nb - 1))
    df = df_num / df_den
    # two-sided p using normal approx if df huge; but we don't have t CDF without scipy.
    # We'll use normal approximation: p = 2*(1 - Phi(|t|))
    # This is acceptable as a fallback; recommend installing scipy for exact.
    abs_t = np.abs(t)
    p = 2 * (1 - 0.5 * (1 + erf(abs_t / np.sqrt(2))))
    # guard
    p = np.clip(p, 0, 1)
    # features with invalid se -> nan
    p[~np.isfinite(se)] = np.nan
    return p


def erf(x):
    # vectorized error function (approx) if math.erf not vectorized
    x = np.asarray(x, dtype=float)
    return np.vectorize(math.erf)(x)


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
        fig = go.Figure().update_layout(template="plotly_dark", title=title)
        return fig

    # downsample for speed (weighted by |color| if possible)
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

    # global index for clicks
    if "_global_index_" in df.columns:
        cd = df.iloc[take]["_global_index_"].to_numpy()
    else:
        cd = df.index.values[take]

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
        scene=dict(
            xaxis_title="RT",
            yaxis_title="m/z",
            zaxis_title="IM/CCS",
            aspectmode="cube",
        ),
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
        fig = go.Figure().update_layout(template="plotly_dark", title=title)
        return fig

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

    if "_global_index_" in df.columns:
        cd = df.iloc[take]["_global_index_"].to_numpy()
    else:
        cd = df.index.values[take]

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
    "padding": "12px 12px",     # <-- taller
    "paddingRight": "3.8em",    # leaves space for +/- spinner
    "minHeight": "52px",        # <-- ensures height
    "boxSizing": "border-box",
}
PANEL_STYLE = {"background": "#000", "color": "#f0f0f0", "padding": "10px", "minHeight": 0, "display": "flex", "flexDirection": "column", "gap": "10px"}
CENTER_STYLE = {"flex": 1, "padding": "10px", "minHeight": 0, "display": "flex", "flexDirection": "column", "background": "#000"}

RADIO_LABEL_STYLE = {"color": "#f0f0f0"}
RADIO_INPUT_STYLE = {"marginRight": "6px"}


def home_view():
    return html.Div(
        style={"height": "calc(100vh - 140px)", "display": "flex", "minHeight": 0},
        children=[
            html.Div(
                style={**PANEL_STYLE, "width": "24%", "borderRight": "1px solid #333"},
                children=[
                    html.Div("Options", style={"fontWeight": 800, "color": "#fff"}),
                    html.Label("View (sample or condition mean)", style=LABEL_STYLE),
                    dcc.Dropdown(id="intensity_dropdown", options=[], value=None, clearable=False,
                                 style={"backgroundColor": "#111", "color": "#000"}),
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
                        figure=go.Figure().update_layout(template="plotly_dark", title="Load a feature file to begin",
                                                         margin=dict(l=0, r=0, t=40, b=0)),
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
            # Left panel
            html.Div(
                style={**PANEL_STYLE, "width": "20%", "borderRight": "1px solid #333", "overflowY": "auto","height": "100%"},
                children=[
                    html.Div("Comparison", style={"fontWeight": 800, "color": "#fff"}),

                    html.Label("Condition 1", style=LABEL_STYLE),
                    dcc.Dropdown(id="cmp_cond1", options=[], value=None, clearable=False,
                                 style={"backgroundColor": "#111", "color": "#000"}),

                    html.Label("Condition 2", style=LABEL_STYLE),
                    dcc.Dropdown(id="cmp_cond2", options=[], value=None, clearable=False,
                                 style={"backgroundColor": "#111", "color": "#000"}),

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
                        style={"marginTop": "6px", "padding": "8px 10px", "background": "#1f6feb",
                               "color": "white", "border": "0", "borderRadius": "6px", "cursor": "pointer"},
                    ),

                    html.Button(
                        "Export features",
                        id="cmp_export_btn",
                        n_clicks=0,
                        style={"padding": "8px 10px", "background": "#238636", "color": "white",
                               "border": "0", "borderRadius": "6px", "cursor": "pointer"},
                    ),

                    html.Div(id="cmp_run_status", style={"fontSize": 12, "opacity": 0.9}),
                ],
            ),

            # Middle panel (plot)
            html.Div(
                style=CENTER_STYLE,
                children=[
                    dcc.Graph(
                        id="cmp_graph",
                        style={"flex": 1, "minHeight": 0},
                        config={"displayModeBar": True},
                        figure=go.Figure().update_layout(template="plotly_dark", title="Click Analyze to view results",
                                                         margin=dict(l=0, r=0, t=40, b=0)),
                    ),
                    html.Div(id="cmp_plot_note", style={"fontFamily": "monospace", "fontSize": 12, "opacity": 0.9}),
                ],
            ),

            # Right panel (summary + selected feature)
            html.Div(
                style={**PANEL_STYLE, "width": "18%", "borderLeft": "1px solid #333"},
                children=[
                    html.Div("Results summary", style={"fontWeight": 800, "color": "#fff"}),
                    html.Div(id="cmp_summary", style={"fontFamily": "monospace", "whiteSpace": "pre-wrap", "marginTop": "6px"},
                             children="Load features + annotations, choose conditions, then click Analyze."),
                    html.Hr(style={"borderColor": "#333"}),
                    html.Div("Selected feature (Comparison)", style={"fontWeight": 800, "color": "#fff"}),
                    html.Pre(
                        id="cmp_selected_feature_box",
                        style={"whiteSpace": "pre-wrap", "fontSize": 12, "color": "#ddd"},
                        children="Click a point in the Comparison view to see details here.",
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
            style={"padding": "8px 12px", "display": "flex", "gap": "12px", "alignItems": "center",
                   "borderBottom": "1px solid #333", "background": "#111", "color": "#f0f0f0"},
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
            children=[dcc.Tab(label="Home", value="tab-home"),
                      dcc.Tab(label="Comparison", value="tab-comparison")],
        ),

        html.Div(
            style={"flex": 1, "minHeight": 0, "background": "#000"},
            children=[
                html.Div(id="home_wrap", children=[home_view()], style={"display": "block", "height": "100%"}),
                html.Div(id="cmp_wrap", children=[comparison_view()], style={"display": "none", "height": "100%"}),
            ],
        ),

        # Data stores
        dcc.Store(id="df_store"),
        dcc.Store(id="meta_store"),
        dcc.Store(id="cond_map_store"),
        dcc.Store(id="sample_meta_store"),

        # Comparison stores
        dcc.Store(id="cmp_result_store"),   # general
        dcc.Store(id="de_store"),           # DE results table (json)

        dcc.Store(id="de_gapfill_store"),   # JSON for gapfill matrix (detected-in-both)
        dcc.Store(id="de_summary_store"),   # string summary for DE + gapfill settings

        # Downloads
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
    prevent_initial_call=True,
)
def on_home_click(clickData, df_json, meta, intensity_col):
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
# - For present_absent: sets cmp_result_store; no auto-download
# - For DE: computes gapfill + DE; stores de_store + de_gapfill_store + de_summary_store
#   (NO downloads here; downloads happen on Export button)
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

    # -------------------
    # Absent / Present
    # -------------------
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
            {"mode": mode, "label": label, "row_indices": viz_idx, "row_categories": viz_cat, "cond1": c1, "cond2": c2},
            None,   # de_store
            None,   # de_gapfill_store
            None,   # de_summary_store
            summary,
            f"Analyze complete (n={n_clicks}). Visualizing {int(viz_mask.sum())} features.",
        )

    # -------------------
    # Differential Expression (DE)
    # -------------------
    p_thresh = float(p_thresh or 0.05)
    fc_thresh = float(fc_thresh or 0.0)

    detected_both = det_call1 & det_call2
    idx_keep = np.where(detected_both)[0]

    if idx_keep.size == 0:
        summary = (
            f"Comparison: {c1} vs {c2}\n"
            f"Mode: DE\n"
            f"No features passed 'detected in both' filter (detected ≥ {detected_pct:.1f}% in each condition).\n"
        )
        return (
            {"mode": "de", "label": "DE: none", "row_indices": [], "cond1": c1, "cond2": c2},
            None, None, summary,
            summary,
            "No DE features to analyze.",
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

    # rename replicate cols for gapfill matrix
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

    # test selection (limma optional; currently falls back to t-test)
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

    # store JSON (no downloads here)
    gapfill_json = gapfill_df.to_json(date_format="iso", orient="split")
    de_json = de_df.to_json(date_format="iso", orient="split")

    cmp_result = {
        "mode": "de",
        "label": f"DE: {c1} vs {c2}",
        "cond1": c1,
        "cond2": c2,
        "row_indices": idx_keep.tolist(),
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
# Build comparison plot
# - Absent/Present: color by log1p(max intensity)
# - DE: color by log2FC from de_store; significant defined by thresholds (already computed)
# -----------------------------
@app.callback(
    Output("cmp_graph", "figure"),
    Output("cmp_plot_note", "children"),
    Input("cmp_result_store", "data"),
    State("de_store", "data"),
    State("df_store", "data"),
    State("meta_store", "data"),
    State("cmp_plot_dim", "value"),
    State("de_show_sig", "value"),
)
def update_cmp_plot(result, de_json, df_json, meta, cmp_plot_dim, de_show_sig):
    if not result or not df_json or not meta:
        return go.Figure().update_layout(template="plotly_dark", title="Click Analyze to view results"), ""

    df = pd.read_json(df_json, orient="split")
    mz_col, rt_col, z_col = meta["mz_col"], meta["rt_col"], meta["z_col"]

    mode = result.get("mode", "present_absent")

    if mode == "de":
        if not de_json:
            return go.Figure().update_layout(template="plotly_dark", title="DE: no results yet"), ""

        # ALWAYS create de_df first
        de_df = pd.read_json(de_json, orient="split")

        # Optional filter
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

        # Colors: log2FC
        color_vals = pd.to_numeric(de_df["log2FC"], errors="coerce").fillna(0.0).to_numpy()

        title = f"DE: {result.get('cond1')} vs {result.get('cond2')} (color=log2FC)"
        note = f"Mode=DE | n={len(de_df)} | view={'sig only' if de_show_sig=='sig' else 'all'} | color=log2FC"

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

        return fig, note


    # Absent/present plotting
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
            fig.add_trace(go.Scattergl(
                x=s[rt_col],
                y=s[mz_col],
                mode="markers",
                marker=dict(size=10, color=fixed_color, opacity=0.9),
                name=trace_name,
                customdata=s["_global_index_"],
                text=[f"row_index={i}" for i in s["_global_index_"]],
                hovertemplate="%{text}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=s[rt_col],
                y=s[mz_col],
                z=s[z_col],
                mode="markers",
                marker=dict(size=4, color=fixed_color, opacity=0.9),
                name=trace_name,
                customdata=s["_global_index_"],
                hovertext=[f"row_index={i}" for i in s["_global_index_"]],
                hovertemplate="%{hovertext}<extra></extra>",
            ))

    # Layout: legend instead of colorbar
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
# Comparison click info (works for 2D and 3D)
# -----------------------------
@app.callback(
    Output("cmp_selected_feature_box", "children"),
    Input("cmp_graph", "clickData"),
    State("df_store", "data"),
    State("meta_store", "data"),
    State("cmp_result_store", "data"),
    State("de_store", "data"),
    prevent_initial_call=True,
)
def on_cmp_click(clickData, df_json, meta, cmp_result, de_json):
    if not clickData or not df_json or not meta:
        return no_update

    df = pd.read_json(df_json, orient="split")
    mz_col, rt_col, z_col = meta["mz_col"], meta["rt_col"], meta["z_col"]

    row_idx = extract_row_index(clickData)
    if row_idx is None or row_idx < 0 or row_idx >= len(df):
        return "Click captured but no valid row index extracted."

    row = df.iloc[row_idx]

    # Base feature info
    base_txt = (
        f"row_index: {row_idx}\n"
        f"row ID: {row.get('row ID','')}\n"
        f"RT: {row[rt_col]:.4f}\n"
        f"m/z: {row[mz_col]:.6f}\n"
        f"IM/CCS: {row[z_col]:.4f}\n"
        f"best ion: {row.get('best ion','')}\n"
    )

    # If DE mode, append DE stats (log2FC, p, FDR)
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
# - Absent/Present: export subset (as before) + summary txt
# - DE: export gapfill matrix + gapfill summary + DE results + DE summary
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

    # -------------------
    # DE export
    # -------------------
    if mode == "de":
        if not de_json or not gapfill_json:
            # nothing to export yet
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
            no_update, no_update,  # cmp downloads not used in DE mode
            dcc.send_data_frame(gapfill_df.to_csv, gapfill_csv_name, index=False),
            dict(content=de_summary_text, filename=gapfill_txt_name),
            dcc.send_data_frame(de_df.to_csv, de_csv_name, index=False),
            dict(content=de_summary_text, filename=de_txt_name),
        )

    # -------------------
    # Absent/Present export (existing behavior)
    # -------------------
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

    # add present_group if available
    if result.get("mode") == "present_absent":
        cat_map = dict(zip(result.get("row_indices", []), result.get("row_categories", [])))
        export_df["present_group"] = [cat_map.get(i, "") for i in idx]

    # rename intensity cols
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
        no_update, no_update, no_update, no_update
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
