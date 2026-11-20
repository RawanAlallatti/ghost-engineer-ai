# ⚡ Ghost Engineer AI

**AI-assisted transformer fault diagnosis from Dissolved Gas Analysis (DGA), with equipment pages for substation maintenance.**

Capstone project, CODED AI & Data Science bootcamp (graduation 10 December 2026).

🔗 **Live app:** https://ghost-engineer-ai-tvlqzlkfyxvsfsx4e6vubz.streamlit.app/

> Research prototype. It supports the engineer; it does not replace site procedures, the manufacturer's instructions or engineering judgement.

---

## The problem

Oil-filled power transformers produce gases when their insulation is stressed by heat or electrical discharges. Laboratories measure these gases (DGA), and engineers interpret them with classic methods such as the **Duval Triangle**. These methods have two weaknesses:

1. They assign a fault type from gas **ratios**, even when the gas **levels** are normal.
2. Many samples cannot be diagnosed at all (in our data, 44% of samples were "Undefined" or "Unidentifiable" by Duval).

## Research question

> Can a system that combines per-gas assessment against IEEE C57.104 with a machine-learning model identify the type of transformer fault, explain its probable cause, and give an answer for cases that the Duval Triangle cannot diagnose?

## How the system works

```
DGA sample (7 gases + O₂/N₂)
        │
        ▼
1. Oil type from O₂/N₂ (Sealed ≤ 0.2 < Free-breathing)
        │
        ▼
2. IEEE C57.104-2019 per-gas check → Normal / Elevated / High
        │
        ├── nothing elevated            → "No fault indication"
        ├── only CO / CO₂ elevated      → "Possible paper insulation degradation"
        ├── only H₂ elevated            → "Possible partial discharge"
        └── hydrocarbons elevated       → 3. Random Forest: Thermal or Electrical + confidence
        │
        ▼
4. Engineer guide: what to check next (sourced)
```

The IEEE layer works as a **gate**: the ML model has no "healthy" class and would answer "Thermal" with 100% confidence even for an all-zero sample, so it is only used when its gases are elevated.

## Data

- **Source:** public DGA dataset, Instituto Tecnológico de Costa Rica, Mendeley Data, DOI [10.17632/98f4z3f8tx.2](https://data.mendeley.com/datasets/98f4z3f8tx/2), licence **CC BY 4.0**.
- **Cleaning:** 1,866 rows → the second half was an exact copy without transformer IDs (dropped) → 933 rows → 296 repeated measurements of the same transformer removed → **637 samples, 105 transformers**.
- **Labels:** Duval Triangle diagnosis grouped into **Thermal** (T1, T2, T3) and **Electrical** (PD, D1, D2) → 345 labelled samples (305 Thermal, 40 Electrical). DT (mixed) and undiagnosable samples were kept for analysis, not for training.
- **IEEE limits:** 90th / 95th percentile tables of IEEE C57.104-2019 as published in Bustamante et al., *Energies* 2020, 13, 5891 (unknown-age column, since sample dates are not available).

## Method

- **Grouped evaluation:** split and 5-fold cross-validation **by transformer** (`StratifiedGroupKFold`), so the same transformer never appears in both training and test.
- **Metric:** macro-F1 (an "always Thermal" model already gets 89% accuracy).
- **Feature selection:** error analysis showed that CO and CO₂ (paper-insulation gases) made the model miss electrical faults, so the final model uses 5 gases: H₂, CH₄, C₂H₆, C₂H₄, C₂H₂.

## Results (5-fold grouped cross-validation, macro-F1)

| Model | Features | Macro-F1 |
|---|---|---|
| Dummy (always Thermal) | – | 0.469 |
| Random Forest | 7 gases | 0.779 ± 0.201 |
| Random Forest | 5 gases | 0.946 ± 0.066 |
| **Random Forest, tuned (max_depth=5) — final** | **5 gases** | **0.961** |
| XGBoost | 5 gases | 0.907 ± 0.120 |
| KNN (scaled) | 5 gases | 0.885 ± 0.052 |
| Soft voting (RF + XGBoost + KNN) | 5 gases | 0.933 ± 0.123 |
| MLP neural network (16, 8) | 5 gases | 0.941 ± 0.050 |

- Adding Duval percentages gave 0.996, but these are exactly Duval's inputs, so that model only reproduces Duval's rule (not used).
- Deep learning gave no advantage with 345 labelled samples.
- **Feature importance:** Acetylene 0.57, Ethylene 0.24, Methane 0.09 (the three Duval gases, 0.90 in total), Ethane 0.06, Hydrogen 0.04.
- **Cases Duval cannot diagnose:** all 80 flagged samples without a Duval diagnosis received an answer (49 ML fault type, 19 paper degradation, 12 partial discharge). These are suggestions; the dataset has no confirmed faults to verify them.

## The app

- **Manual DGA input:** enter gas values and get the per-gas table, the result and the engineer guide.
- **Equipment pages:** 7 demo items (5 distribution transformers 11/0.433 kV, an 11 kV switchgear, a 433 V LV panel) with test history (Megger, TTR, trip test), open problems and recommended actions. Each page has a **QR code** link (`?id=TR-01`).
- Built with Streamlit, SQLite and scikit-learn.

⚠️ **All equipment and test records in the app are fictional demo data.** No real facility data is used.

## Limitations

- Labels come from the Duval Triangle, not from confirmed internal inspections, so the model learns Duval-style diagnosis.
- Only 40 electrical samples, from 15 transformers; the data is 85% thermal, and the model tends to prefer "Thermal".
- Model confidence is agreement between trees, not the probability that a fault exists.
- No sample dates or failure records → the system **diagnoses** fault type; it does **not predict** future failures.
- IEEE limits are statistical and the age-specific limits could not be used.

## Future work

- Fault and gas-trend forecasting with time-stamped DGA data.
- External test on DGA data from other utilities.
- Review of the engineer guide by senior maintenance engineers.

## Files

| File | Content |
|---|---|
| `app_capstone.py` | Streamlit app (decision rules, SQLite demo database, engineer guide) |
| `ghost_rf.joblib` | Final Random Forest model (scikit-learn 1.6.1) |
| `requirements.txt` | Python packages |

## Author

Rawan Alallatti — Electrical Engineer
