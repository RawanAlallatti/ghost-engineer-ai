"""
Ghost Engineer AI - Transformer DGA diagnosis (research prototype)
Run on Streamlit Community Cloud. Files needed in the same folder:
app.py, ghost_rf.joblib, requirements.txt
"""
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# 1) Settings (copied from the capstone notebook)
# ---------------------------------------------------------------
GAS_COLUMNS = ["Hydrogen", "Methane", "Ethane", "Ethylene",
               "Acetylene", "Carbon Monoxide", "Carbon Dioxide"]
GASES_5 = ["Hydrogen", "Methane", "Ethane", "Ethylene", "Acetylene"]   # model inputs
HYDROCARBONS = ["Methane", "Ethane", "Ethylene", "Acetylene"]
PAPER_GASES = ["Carbon Monoxide", "Carbon Dioxide"]

FORMULA = {"Hydrogen": "H₂", "Methane": "CH₄", "Ethane": "C₂H₆", "Ethylene": "C₂H₄",
           "Acetylene": "C₂H₂", "Carbon Monoxide": "CO", "Carbon Dioxide": "CO₂"}

# IEEE C57.104-2019 limits, unknown age (Bustamante et al., Energies 2020, Tables 1 and 2)
LIMITS = {
    "Sealed": {
        "p90": {"Hydrogen": 80, "Methane": 90, "Ethane": 90, "Ethylene": 50,
                "Acetylene": 1, "Carbon Monoxide": 900, "Carbon Dioxide": 9000},
        "p95": {"Hydrogen": 200, "Methane": 150, "Ethane": 175, "Ethylene": 100,
                "Acetylene": 2, "Carbon Monoxide": 1100, "Carbon Dioxide": 12500},
    },
    "Free-breathing": {
        "p90": {"Hydrogen": 40, "Methane": 20, "Ethane": 15, "Ethylene": 50,
                "Acetylene": 2, "Carbon Monoxide": 500, "Carbon Dioxide": 5000},
        "p95": {"Hydrogen": 90, "Methane": 50, "Ethane": 40, "Ethylene": 100,
                "Acetylene": 7, "Carbon Monoxide": 600, "Carbon Dioxide": 7000},
    },
}


# ---------------------------------------------------------------
# 2) Logic (same rules as the notebook, no Streamlit inside)
# ---------------------------------------------------------------
def oil_type_from_ratio(oxygen, nitrogen):
    """O2/N2 <= 0.2 -> Sealed, otherwise Free-breathing."""
    ratio = oxygen / nitrogen
    return ("Sealed" if ratio <= 0.2 else "Free-breathing"), ratio


def gas_status(gases, oil_type):
    """Normal / Elevated / High for each gas, against the IEEE limits."""
    status = {}
    for gas in GAS_COLUMNS:
        p90 = LIMITS[oil_type]["p90"][gas]
        p95 = LIMITS[oil_type]["p95"][gas]
        if gases[gas] > p95:
            status[gas] = "High"
        elif gases[gas] > p90:
            status[gas] = "Elevated"
        else:
            status[gas] = "Normal"
    return status


def diagnose(gases, status, model):
    """The 4 rules of the decision function."""
    elevated = [g for g in GAS_COLUMNS if status[g] != "Normal"]

    # Rule 1: nothing elevated -> do not call the model
    if len(elevated) == 0:
        return {"App Result": "No fault indication", "ML Type": None, "Confidence": None}

    # Rule 2: only CO / CO2 elevated
    if all(g in PAPER_GASES for g in elevated):
        return {"App Result": "Possible paper insulation degradation", "ML Type": None, "Confidence": None}

    # Rule 3: H2 elevated, all 4 hydrocarbons Normal
    if "Hydrogen" in elevated and not any(g in elevated for g in HYDROCARBONS):
        return {"App Result": "Possible partial discharge (H2 elevated)", "ML Type": None, "Confidence": None}

    # Rule 4: call the ML model
    x = pd.DataFrame([{g: float(gases[g]) for g in GASES_5}])
    ml_type = model.predict(x)[0]
    confidence = model.predict_proba(x).max()
    return {"App Result": "ML diagnosis", "ML Type": ml_type, "Confidence": round(float(confidence), 2)}


# ---------------------------------------------------------------
# 3) Page
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("ghost_rf.joblib")


def main():
    st.set_page_config(page_title="Ghost Engineer AI", page_icon="⚡", layout="centered")
    st.title("⚡ Ghost Engineer AI")
    st.caption("Transformer fault diagnosis from Dissolved Gas Analysis (DGA). "
               "Research prototype for a bootcamp capstone. Not a replacement for engineering judgement.")

    model = load_model()

    st.subheader("1. Oil preservation type")
    c1, c2 = st.columns(2)
    oxygen = c1.number_input("Oxygen O₂ (ppm)", min_value=0.0, value=0.0, step=100.0)
    nitrogen = c2.number_input("Nitrogen N₂ (ppm)", min_value=0.0, value=0.0, step=1000.0)

    if nitrogen > 0:
        oil_type, ratio = oil_type_from_ratio(oxygen, nitrogen)
        st.info(f"O₂/N₂ = {ratio:.2f} → **{oil_type}** (threshold 0.2)")
    else:
        oil_type = st.radio("N₂ not entered. Choose the type:", ["Sealed", "Free-breathing"], horizontal=True)

    st.subheader("2. Fault gases (ppm)")
    gases = {}
    cols = st.columns(2)
    for i, gas in enumerate(GAS_COLUMNS):
        gases[gas] = cols[i % 2].number_input(f"{gas} {FORMULA[gas]}", min_value=0.0, value=0.0, step=1.0)

    if not st.button("Diagnose", type="primary", use_container_width=True):
        return

    status = gas_status(gases, oil_type)
    result = diagnose(gases, status, model)

    # Per-gas table
    st.subheader("3. Per-gas assessment (IEEE C57.104-2019)")
    table = pd.DataFrame({
        "Gas": [f"{g} ({FORMULA[g]})" for g in GAS_COLUMNS],
        "Value (ppm)": [gases[g] for g in GAS_COLUMNS],
        "90th pct limit": [LIMITS[oil_type]["p90"][g] for g in GAS_COLUMNS],
        "95th pct limit": [LIMITS[oil_type]["p95"][g] for g in GAS_COLUMNS],
        "Status": [status[g] for g in GAS_COLUMNS],
    })
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Limits are statistical (90th / 95th percentile of many transformers), unknown age. "
               "'Elevated' means a higher probability of a problem, not a confirmed fault.")

    # Result
    st.subheader("4. Result")
    app_result = result["App Result"]
    if app_result == "No fault indication":
        st.success("✅ No fault indication: all gases are within the 90th percentile limits.")
    elif app_result == "Possible paper insulation degradation":
        st.warning("🟠 Possible paper insulation degradation: only CO/CO₂ are elevated. "
                   "The ML model is not used because it does not see CO/CO₂.")
    elif app_result == "Possible partial discharge (H2 elevated)":
        st.warning("🟠 Possible partial discharge: H₂ is elevated while all hydrocarbons are normal. "
                   "The ML model is not used because it gives H₂ little weight and was trained on only one PD case.")
    else:
        st.error(f"🔴 Possible **{result['ML Type']}** fault "
                 f"(model confidence {result['Confidence']:.0%})")
        st.caption("Confidence = agreement between the trees of the Random Forest, not the real probability "
                   "that the fault exists. The model was trained on Duval Triangle labels (85% thermal), "
                   "so it reproduces Duval-style diagnosis and tends to prefer 'Thermal'.")

    elevated = [f"{g} ({FORMULA[g]})" for g in GAS_COLUMNS if status[g] != "Normal"]
    if elevated:
        st.write("**Elevated gases:** " + ", ".join(elevated))

    with st.expander("About this prototype"):
        st.markdown(
            "- Data: public DGA dataset, Instituto Tecnológico de Costa Rica (Mendeley, CC BY 4.0).\n"
            "- Model: Random Forest on 5 gases (H₂, CH₄, C₂H₆, C₂H₄, C₂H₂), grouped CV macro-F1 ≈ 0.96.\n"
            "- Labels come from the Duval Triangle, not from confirmed inspections.\n"
            "- The system diagnoses the fault type; it does not predict future failures.")


if __name__ == "__main__":
    main()
