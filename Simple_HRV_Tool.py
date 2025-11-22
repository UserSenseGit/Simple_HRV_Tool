
import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import pyhrv.time_domain as td
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Simple HRV Tool", page_icon="🫀")
st.title("Simple HRV Tool")

st.markdown(
    "Upload a **.csv** with R-R intervals. "
)

uploaded_file = st.file_uploader("Choose a CSV-file", type=["csv"])

# ----------------- helpers -----------------
def _normalize(name: str) -> str:
    if name is None:
        return ""
    # lower, strip, collapse spaces, remove non-alphanum except underscore
    name = str(name).lower().strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name

CANDIDATES = {
    "rr", "rrms", "rr_ms", "rrinterval", "rrintervals", "r_r", "r_r_ms", "r_rinterval",
    "rri", "rri_ms", "nn", "nn_ms", "ibi", "ibi_ms", "interbeat", "interbeat_interval",
}

def _try_read_csv(file_obj):
    """Try multiple ways to read a CSV with common European/edge cases."""
    # Keep a copy since Streamlit's UploadedFile is a BytesIO-like object
    raw = file_obj.read()
    # Try sep=None (sniff) first
    for attempt in (
        dict(sep=None, engine="python"),
        dict(sep=";"),
        dict(sep=","),
        dict(delim_whitespace=True),
    ):
        for enc in ("utf-8", "utf-8-sig", "latin1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, **attempt)
                return df
            except Exception:
                continue
    # Last resort: read without header and force single column
    try:
        df = pd.read_csv(io.BytesIO(raw), header=None)
        return df
    except Exception:
        return None

def _coerce_numeric(series: pd.Series) -> pd.Series:
    # Handle decimal commas and strip non-numeric
    s = series.astype(str).str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-eE]", "", regex=True)
    s = pd.to_numeric(s, errors="coerce")
    return s

def _find_rr_column(df: pd.DataFrame):
    # If no header, create generic names
    if df.columns.dtype == "int64":
        # try first numeric-looking column with plausible RR values
        for col in df.columns:
            s = _coerce_numeric(df[col])
            valid = s[(s > 250) & (s < 3000)]
            if valid.count() >= max(2, int(0.05 * len(s))):
                return col, s
        return None, None

    # Normalize column names
    colmap = {c: _normalize(c) for c in df.columns}
    # Try exact candidate match
    for orig, norm in colmap.items():
        if norm in CANDIDATES or re.fullmatch(r"(rr|rri|nn)(|_ms)$", norm or ""):
            s = _coerce_numeric(df[orig])
            if s.notna().sum() >= 2:
                return orig, s

    # If not found, pick the most plausible numeric column by value range
    best_col, best_score, best_series = None, -1, None
    for orig in df.columns:
        s = _coerce_numeric(df[orig])
        if s.notna().sum() < 2:
            continue
        # score by fraction within plausible RR range (250–3000 ms)
        within = s[(s > 250) & (s < 3000)].count()
        score = within / max(1, s.notna().sum())
        if score > best_score:
            best_col, best_score, best_series = orig, score, s
    if best_col is not None and best_score > 0.3:  # at least 30% plausible
        return best_col, best_series

    return None, None

# ----------------- main flow -----------------
if uploaded_file:
    df = _try_read_csv(uploaded_file)
    if df is None or len(df) == 0:
        st.error("Could not read the cvs-file. Check the format.")
        st.stop()

    # Show quick preview
    st.markdown("#### example of the read in data")
    st.dataframe(df.head(), use_container_width=True)

    # Try to detect RR column
    rr_col, rr_series = _find_rr_column(df)

    # Allow manual override if detection failed or user wants to change
    cols = list(df.columns)
    manual = False
    if rr_col is None:
        st.warning("Couldn't find a clear R-R column automatically. Select the correct column below.")
        manual = True
    else:
        st.success(f"Gedetecteerde R-R kolom: **{rr_col}**")
        manual = st.checkbox("Select column manually?", value=False)

    if manual:
        rr_choice = st.selectbox("Select the column with R-R intervals (in ms)", options=cols)
        rr_series = _coerce_numeric(df[rr_choice])
        rr_col = rr_choice

    # Validate
    if rr_series is None or rr_series.notna().sum() < 2:
        st.error("File does not contain a usable R-R column (at least 2 numeric values ​​required).")
        st.stop()

    # Clean final series
    rr_ms = rr_series.dropna().astype(float).values
    # Remove non-positive entries
    rr_ms = rr_ms[rr_ms > 0]

    if rr_ms.size < 2:
        st.error("Insufficient valid R-R values ​​after cleaning.")
        st.stop()

    # Compute HR in BPM
    full_hr = 60000.0 / rr_ms

    # Robust slider bounds using percentiles and clamp to physiological range
    p1 = float(np.nanpercentile(full_hr, 1))
    p99 = float(np.nanpercentile(full_hr, 99))
    min_hr = int(np.clip(min(p1, p99), 30, 220))
    max_hr = int(np.clip(max(p1, p99), 30, 220))
    if min_hr >= max_hr:
        min_hr = max(min_hr - 1, 30)
        max_hr = min(max_hr + 1, 220)

    bpm_min, bpm_max = st.slider(
        "Select BPM range (values ​​outside this range will be ignored)",
        min_value=30, max_value=220, value=(min_hr, max_hr), step=1
    )

    mask = (full_hr >= bpm_min) & (full_hr <= bpm_max)
    rr_ms = rr_ms[mask]
    full_hr = full_hr[mask]

    if rr_ms.size < 2:
        st.error("Too few data points within the selected BPM range.")
        st.stop()

    st.markdown("### Measured heart rate (BPM) over filtered range")

    num_regions = st.number_input("Number of regions", min_value=1, max_value=10, value=1, step=1)

    selections = []
    rmssd_per_regio = []
    slider_ranges = []

    with st.expander("🛠️ select regions"):
        for i in range(num_regions):
            st.markdown(f"#### Region {i+1}")
            slider = st.slider(
                f"Select region {i+1}",
                min_value=0,
                max_value=len(rr_ms) - 1,
                value=(0, min(50, len(rr_ms) - 1)),
                step=1,
                key=f"slider_{i}"
            )
            start, end = slider
            region = rr_ms[start:end]
            slider_ranges.append((start, end))

            if len(region) < 2:
                st.warning(f"Region {i+1} has too little data (<2).")
            else:
                gebruiken = st.checkbox(f"Region {i+1} use in analysis?", value=True, key=f"use_{i}")
                if gebruiken:
                    selections.append(region)
                    rmssd = td.rmssd(region)['rmssd']
                    rmssd_per_regio.append({
                        "Region": f"{i+1}",
                        "Start": start,
                        "End": end,
                        "Values": len(region),
                        "RMSSD (ms)": round(rmssd, 2)
                    })

    # Plot with regions
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=full_hr, mode='lines', name='HR (BPM)', line=dict(color='blue')))

    kleuren = [
        'rgba(255,0,0,0.2)', 'rgba(0,255,0,0.2)', 'rgba(0,0,255,0.2)',
        'rgba(255,165,0,0.2)', 'rgba(128,0,128,0.2)', 'rgba(0,206,209,0.2)',
        'rgba(255,20,147,0.2)', 'rgba(0,100,0,0.2)'
    ]
    for i, (start, end) in enumerate(slider_ranges):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=kleuren[i % len(kleuren)], opacity=0.3,
            layer="below", line_width=0,
            annotation_text=f"Regio {i+1}", annotation_position="top left"
        )

    st.markdown("### Y-axis settings (BPM range)")
    y_min = int(max(30, np.floor(np.min(full_hr)) - 5))
    y_max = int(min(220, np.ceil(np.max(full_hr)) + 5))
    y_min, y_max = st.slider(
        "Set BPM range for vertical zoom",
        min_value=30, max_value=220, value=(y_min, y_max), step=1
    )

    fig.update_layout(
        title="Measured heart rate (BPM) over filtered range",
        xaxis_title="Index",
        yaxis_title="BPM",
        height=350,
        margin=dict(l=10, r=10, t=40, b=20),
        hovermode='x unified',
        yaxis=dict(range=[y_min, y_max]),
    )
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    if rmssd_per_regio:
        st.markdown("### RMSSD by region")
        st.dataframe(pd.DataFrame(rmssd_per_regio), use_container_width=True)

    if st.button("Calculate combined values"):
        if len(selections) == 0:
            st.warning("No valid regions selected.")
        else:
            combined = np.concatenate(selections)
            rmssd_combined = td.rmssd(combined)['rmssd']
            mean_rr = float(np.mean(combined))
            mean_bpm = 60000.0 / mean_rr

            st.markdown("### Combined analysis of all regions")
            st.success(f"**Average heart rate:** {mean_bpm:.1f} bpm")
            st.success(f"**RMSSD combined:** {rmssd_combined:.2f} ms")
            

            combined_hr = 60000.0 / combined
            st.line_chart(combined_hr, height=250, use_container_width=True)

