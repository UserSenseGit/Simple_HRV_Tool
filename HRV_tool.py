st.set_page_config(layout="centered", page_title="HRV Tool", page_icon="🫀")
import streamlit as st
import pandas as pd
import numpy as np
import pyhrv.time_domain as td

st.set_page_config(layout="wide")
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
    st.line_chart(full_hr, height=300)

    st.markdown("### Selecteer regio’s")
    selections = []
    rmssd_per_regio = []
    num_regions = st.number_input("Aantal regio’s", min_value=1, max_value=10, value=2, step=1)

    for i in range(num_regions):
        st.markdown(f"#### Regio {i+1}")
        slider = st.slider(
            f"Selecteer regio {i+1}",
            min_value=0,
            max_value=len(rr_intervals) - 1,
            value=(0, min(50, len(rr_intervals)-1)),
            step=1,
            key=f"slider_{i}"
        )
        start, end = slider
        region = rr_intervals[start:end]

        if len(region) < 2:
            st.warning(f"Regio {i+1} heeft te weinig data (<2).")
        else:
            selections.append(region)
            rmssd = td.rmssd(region)['rmssd']
            rmssd_per_regio.append({"Regio": f"{i+1}", "Start": start, "Einde": end, "Waarden": len(region), "RMSSD (ms)": round(rmssd, 2)})

    if rmssd_per_regio:
        st.markdown("### RMSSD per regio")
        st.dataframe(pd.DataFrame(rmssd_per_regio), use_container_width=True)

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
