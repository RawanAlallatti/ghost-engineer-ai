"""
Ghost Engineer AI - Transformer DGA diagnosis + equipment pages (research prototype)
Files needed in the repo: app_capstone.py, ghost_rf.joblib, requirements.txt

Open an equipment page with a link like:  https://<your-app>.streamlit.app/?id=TR-01
All equipment and test records below are FICTIONAL DEMO DATA.
"""
import sqlite3

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

DB_PATH = "ghost_demo.db"

# ---------------------------------------------------------------
# 2) Demo database (FICTIONAL data, realistic Kuwait 11 kV / 433 V, 50 Hz system)
# ---------------------------------------------------------------
EQUIPMENT = [
    # equipment_id, equipment_type, manufacturer, manufacture_year, location, phases
    ("TR-01", "Distribution transformer 11/0.433 kV, 1000 kVA, ONAN", "Demo Manufacturer A", 2012, "Substation S1 (demo)", 3),
    ("TR-02", "Distribution transformer 11/0.433 kV, 1500 kVA, ONAN", "Demo Manufacturer B", 2008, "Substation S2 (demo)", 3),
    ("SWG-01", "11 kV switchgear (VCB panel)", "Demo Manufacturer C", 2012, "Substation S1 (demo)", 3),
    ("LVP-01", "LV distribution panel 433 V", "Demo Manufacturer D", 2015, "Substation S1 (demo)", 3),
]

TEST_RESULTS = [
    # equipment_id, test_date, test_type, result
    ("TR-01", "2026-03-10", "Insulation resistance (Megger), HV-LV", "Pass"),
    ("TR-01", "2026-03-10", "Turns ratio (TTR), expected ≈ 25.4", "Pass"),
    ("TR-02", "2026-02-18", "Insulation resistance (Megger), HV-LV", "Pass"),
    ("SWG-01", "2026-04-02", "VCB trip test (operating time)", "Pass"),
    ("LVP-01", "2026-05-20", "Insulation resistance (Megger)", "Pass"),
]

DGA_SAMPLES = [
    # equipment_id, sample_date, H2, CH4, C2H6, C2H4, C2H2, CO, CO2, O2, N2  (ppm, demo values)
    ("TR-01", "2026-09-15", 30, 10, 5, 8, 20, 300, 2500, 15000, 50000),
    ("TR-02", "2026-08-20", 25, 10, 8, 5, 0, 950, 6000, 1000, 40000),
]


def build_database():
    """Create the SQLite tables and fill them with the demo data (safe to run every time)."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS equipment (
        equipment_id TEXT PRIMARY KEY, equipment_type TEXT, manufacturer TEXT,
        manufacture_year INTEGER, location TEXT, phases INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS test_results (
        test_id INTEGER PRIMARY KEY AUTOINCREMENT, equipment_id TEXT, test_date TEXT,
        test_type TEXT, result TEXT,
        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS dga_samples (
        sample_id INTEGER PRIMARY KEY AUTOINCREMENT, equipment_id TEXT, sample_date TEXT,
        Hydrogen REAL, Methane REAL, Ethane REAL, Ethylene REAL, Acetylene REAL,
        "Carbon Monoxide" REAL, "Carbon Dioxide" REAL, Oxygen REAL, Nitrogen REAL,
        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id))""")

    # Start clean each time so the demo data is never duplicated
    cur.execute("DELETE FROM equipment")
    cur.execute("DELETE FROM test_results")
    cur.execute("DELETE FROM dga_samples")
    cur.executemany("INSERT INTO equipment VALUES (?, ?, ?, ?, ?, ?)", EQUIPMENT)
    cur.executemany("INSERT INTO test_results (equipment_id, test_date, test_type, result) "
                    "VALUES (?, ?, ?, ?)", TEST_RESULTS)
    cur.executemany("""INSERT INTO dga_samples (equipment_id, sample_date, Hydrogen, Methane, Ethane,
        Ethylene, Acetylene, "Carbon Monoxide", "Carbon Dioxide", Oxygen, Nitrogen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", DGA_SAMPLES)
    con.commit()
    con.close()


def query(sql, params=()):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, con, params=params)
    con.close()
    return df


# ---------------------------------------------------------------
# 3) Diagnosis logic (same rules as the notebook, no Streamlit inside)
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
    return {"App Result": "ML diagnosis", "ML Type": str(ml_type), "Confidence": round(float(confidence), 2)}


# ---------------------------------------------------------------
# 4) Page parts
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("ghost_rf.joblib")


def show_diagnosis(gases, oil_type, model):
    """Per-gas table + result box."""
    status = gas_status(gases, oil_type)
    result = diagnose(gases, status, model)

    st.markdown("**Per-gas assessment (IEEE C57.104-2019)**")
    table = pd.DataFrame({
        "Gas": [f"{g} ({FORMULA[g]})" for g in GAS_COLUMNS],
        "Value (ppm)": [gases[g] for g in GAS_COLUMNS],
        "90th pct limit": [LIMITS[oil_type]["p90"][g] for g in GAS_COLUMNS],
        "95th pct limit": [LIMITS[oil_type]["p95"][g] for g in GAS_COLUMNS],
        "Status": [status[g] for g in GAS_COLUMNS],
    })
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(f"Oil type: {oil_type}. Limits are statistical (90th / 95th percentile, unknown age). "
               "'Elevated' means a higher probability of a problem, not a confirmed fault.")

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


def equipment_page(equipment_id, model):
    info = query("SELECT * FROM equipment WHERE equipment_id = ?", (equipment_id,))
    if info.empty:
        st.error(f"Equipment '{equipment_id}' not found.")
        return

    row = info.iloc[0]
    st.header(f"🔌 {row['equipment_id']}")
    st.caption("⚠️ Fictional demo data, not real airport equipment.")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Type:** {row['equipment_type']}  \n**Manufacturer:** {row['manufacturer']}")
    c2.markdown(f"**Manufacture year:** {row['manufacture_year']}  \n**Location:** {row['location']}  \n"
                f"**Phases:** {row['phases']}")

    st.subheader("Test history")
    tests = query("SELECT test_date AS Date, test_type AS Test, result AS Result "
                  "FROM test_results WHERE equipment_id = ? ORDER BY test_date DESC", (equipment_id,))
    if tests.empty:
        st.write("No tests recorded.")
    else:
        st.dataframe(tests, hide_index=True, use_container_width=True)

    dga = query("SELECT * FROM dga_samples WHERE equipment_id = ? ORDER BY sample_date DESC",
                (equipment_id,))
    if not dga.empty:
        latest = dga.iloc[0]
        st.subheader(f"Latest DGA sample ({latest['sample_date']})")
        gases = {g: float(latest[g]) for g in GAS_COLUMNS}
        oil_type, ratio = oil_type_from_ratio(latest["Oxygen"], latest["Nitrogen"])
        st.caption(f"O₂/N₂ = {ratio:.2f} → {oil_type}")
        show_diagnosis(gases, oil_type, model)


def manual_page(model):
    st.subheader("Manual DGA input")
    c1, c2 = st.columns(2)
    oxygen = c1.number_input("Oxygen O₂ (ppm)", min_value=0.0, value=0.0, step=100.0)
    nitrogen = c2.number_input("Nitrogen N₂ (ppm)", min_value=0.0, value=0.0, step=1000.0)

    if nitrogen > 0:
        oil_type, ratio = oil_type_from_ratio(oxygen, nitrogen)
        st.info(f"O₂/N₂ = {ratio:.2f} → **{oil_type}** (threshold 0.2)")
    else:
        oil_type = st.radio("N₂ not entered. Choose the type:", ["Sealed", "Free-breathing"], horizontal=True)

    gases = {}
    cols = st.columns(2)
    for i, gas in enumerate(GAS_COLUMNS):
        gases[gas] = cols[i % 2].number_input(f"{gas} {FORMULA[gas]} (ppm)", min_value=0.0, value=0.0, step=1.0)

    if st.button("Diagnose", type="primary", use_container_width=True):
        show_diagnosis(gases, oil_type, model)


def main():
    st.set_page_config(page_title="Ghost Engineer AI", page_icon="⚡", layout="centered")
    st.title("⚡ Ghost Engineer AI")
    st.caption("Transformer fault diagnosis from Dissolved Gas Analysis (DGA). "
               "Research prototype for a bootcamp capstone. Not a replacement for engineering judgement.")

    build_database()
    model = load_model()

    # QR code link: ?id=TR-01 opens that equipment page directly
    equipment_id = st.query_params.get("id")

    if equipment_id:
        equipment_page(equipment_id, model)
        st.divider()
        st.link_button("⬅ All equipment / manual input", "?")
    else:
        ids = query("SELECT equipment_id FROM equipment")["equipment_id"].tolist()
        choice = st.selectbox("Open an equipment page (demo)", ["Manual DGA input"] + ids)
        if choice == "Manual DGA input":
            manual_page(model)
        else:
            equipment_page(choice, model)

    with st.expander("About this prototype"):
        st.markdown(
            "- Data: public DGA dataset, Instituto Tecnológico de Costa Rica (Mendeley, CC BY 4.0).\n"
            "- Model: Random Forest on 5 gases (H₂, CH₄, C₂H₆, C₂H₄, C₂H₂), grouped CV macro-F1 ≈ 0.96.\n"
            "- Labels come from the Duval Triangle, not from confirmed inspections.\n"
            "- Equipment and test records are fictional demo data.\n"
            "- The system diagnoses the fault type; it does not predict future failures.")


if __name__ == "__main__":
    main()
