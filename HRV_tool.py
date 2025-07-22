import streamlit as st
import pandas as pd
import numpy as np
import pyhrv.time_domain as td
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="HRV Tool", page_icon="🫀")
st.title("Kubios-achtige HRV Analyse")

uploaded_file = st.file_uploader("Upload een R-R interval bestand (txt of csv, 1 kolom, ms)", type=["txt", "csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, header=None, delim_whitespace=True)
    except:
        df = pd.read_csv(uploaded_file, header=None)

    try:
        rr_intervals = df[0].astype(float).values
    except:
        st.error("Kon de gegevens niet omzetten naar numerieke R-R intervallen.")
        st.stop()

    full_hr = 60000 / rr_intervals

    st.markdown("### Gemeten hartslag (BPM) over volledige reeks")

    # Aantal regio’s selecteren
    num_regions = st.number_input("Aantal regio’s", min_value=1, max_value=10, value=2, step=1)

    # Definieer sliders in een expander
    selections = []
    rmssd_per_regio = []
    slider_ranges = []

    with st.expander("🛠️ Selecteer regio’s"):
        for i in range(num_regions):
            st.markdown(f"#### Regio {i+1}")
            slider = st.slider(
                f"Selecteer regio {i+1}",
                min_value=0,
                max_value=len(rr_intervals) - 1,
                value=(0, min(50, len(rr_intervals) - 1)),
                step=1,
                key=f"slider_{i}"
            )
            start, end = slider
            region = rr_intervals[start:end]
            slider_ranges.append((start, end))

            if len(region) < 2:
                st.warning(f"Regio {i+1} heeft te weinig data (<2).")
            else:
                selections.append(region)
                rmssd = td.rmssd(region)['rmssd']
                rmssd_per_regio.append({
                    "Regio": f"{i+1}",
                    "Start": start,
                    "Einde": end,
                    "Waarden": len(region),
                    "RMSSD (ms)": round(rmssd, 2)
                })

    # Maak interactieve plotly-grafiek
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=full_hr, mode='lines', name='HR (BPM)', line=dict(color='blue')))

    kleuren = ['rgba(255,0,0,0.2)', 'rgba(0,255,0,0.2)', 'rgba(0,0,255,0.2)', 'rgba(255,165,0,0.2)', 
               'rgba(128,0,128,0.2)', 'rgba(0,206,209,0.2)', 'rgba(255,20,147,0.2)', 'rgba(0,100,0,0.2)']

    for i, (start, end) in enumerate(slider_ranges):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=kleuren[i % len(kleuren)],
            opacity=0.3,
            layer="below",
            line_width=0,
            annotation_text=f"Regio {i+1}",
            annotation_position="top left"
        )

    fig.update_layout(
        title="Gemeten hartslag (BPM) over volledige reeks",
        xaxis_title="Index",
        yaxis_title="BPM",
        height=350,
        margin=dict(l=10, r=10, t=40, b=20),
        hovermode='x unified'
    )

    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    # Toon RMSSD per regio
    if rmssd_per_regio:
        st.markdown("### RMSSD per regio")
        st.dataframe(pd.DataFrame(rmssd_per_regio), use_container_width=True)

    # Gecombineerde analyse
    if st.button("Bereken gecombineerde waarden"):
        if len(selections) == 0:
            st.warning("Geen geldige regio's geselecteerd.")
        else:
            combined = np.concatenate(selections)
            rmssd_combined = td.rmssd(combined)['rmssd']
            mean_rr = np.mean(combined)
            mean_bpm = 60000 / mean_rr

            st.markdown("### Gecombineerde analyse van alle regio’s")
            st.success(f"**RMSSD gecombineerd:** {rmssd_combined:.2f} ms")
            st.success(f"**Gemiddelde hartslag:** {mean_bpm:.1f} bpm")

            combined_hr = 60000 / combined
            st.line_chart(combined_hr, height=250, use_container_width=True)

