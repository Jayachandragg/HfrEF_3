"""
╔══════════════════════════════════════════════════════════════════╗
║     PIXEL MINDS — HFrEF Medication Titration Logic Engine       ║
║     University at Buffalo · CDA Project · Spring 2026           ║
║     Client: Dr. Ciprian Ionita, QAS.AI                          ║
╚══════════════════════════════════════════════════════════════════╝

USAGE:
    # Run on default sample (first 100 patients from dataset)
    python logic_engine.py

    # Run on your own CSV file
    python logic_engine.py --input your_data.csv

    # Run on all patients (no limit)
    python logic_engine.py --input hfref_final_dataset.csv --all

    # Run on N patients
    python logic_engine.py --input hfref_final_dataset.csv --patients 500

    # Save results to a specific file
    python logic_engine.py --output my_results.csv

REQUIRED CSV COLUMNS (at minimum):
    subject_id, hadm_id, charttime,
    heart_rate, sbp, spo2, resp_rate,
    creatinine, potassium, egfr,
    has_afib, has_t1dm, has_copd

INSTALL:
    pip install -r requirements.txt
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime


# ════════════════════════════════════════════════════════════════
# HELPER
# ════════════════════════════════════════════════════════════════
def get(row, col, default=np.nan):
    """Safely read a column value, returning default if missing/NaN."""
    val = row.get(col, default)
    if pd.isna(val):
        return default
    return val


# ════════════════════════════════════════════════════════════════
# 7-STEP LOGIC ENGINE
# ════════════════════════════════════════════════════════════════
def run_logic(row):
    """
    Apply the 7-step HFrEF medication titration logic to one patient row.

    Returns a dict with decisions for all 7 steps.
    Based on:
      - AHA/ACC/HFSA 2022 HF Guidelines
      - ACC 2024 Expert Consensus Decision Pathway
      - DAPA-HF / EMPEROR-Reduced trials
      - 5 additional peer-reviewed clinical papers
    """
    result = {
        'subject_id':       row['subject_id'],
        'hadm_id':          row['hadm_id'],
        'charttime':        row['charttime'],
        'step1_emergency':  'SAFE',
        'step2_fluid':      'UNKNOWN',
        'step3_diuretic':   'HOLD',
        'step4_raas':       'HOLD',
        'step5_bb':         'HOLD',
        'step6_sglt2':      'HOLD',
        'step6_mra':        'HOLD',
        'step7_trajectory': 'UNKNOWN',
        'alert':             False,
        'alert_reason':      '',
    }

    # ── Read sensor + lab values ──────────────────────────────────
    hr        = get(row, 'heart_rate')
    sbp       = get(row, 'sbp')
    spo2      = get(row, 'spo2')
    resp      = get(row, 'resp_rate')
    creat     = get(row, 'creatinine')
    potassium = get(row, 'potassium')
    egfr      = get(row, 'egfr')
    has_afib  = get(row, 'has_afib',  0)
    has_t1dm  = get(row, 'has_t1dm',  0)
    has_copd  = get(row, 'has_copd',  0)

    # ════════════════════════════════════════════════════════════
    # STEP 1 — Emergency Gates
    # Any trigger → HOLD ALL drugs, alert clinician immediately
    # ════════════════════════════════════════════════════════════
    reasons = []
    if not pd.isna(spo2)      and spo2      < 90:   reasons.append(f"SpO2={spo2:.0f}% (< 90%)")
    if not pd.isna(sbp)       and sbp       < 90:   reasons.append(f"SBP={sbp:.0f} mmHg (< 90)")
    if not pd.isna(potassium) and potassium > 6.0:  reasons.append(f"K+={potassium:.1f} mEq/L (> 6.0)")
    if not pd.isna(creat)     and creat     > 3.5:  reasons.append(f"Creat={creat:.1f} mg/dL (> 3.5)")
    if not pd.isna(hr)        and hr        < 40:   reasons.append(f"HR={hr:.0f} bpm (< 40)")
    if not pd.isna(egfr)      and egfr      < 15:   reasons.append(f"eGFR={egfr:.0f} (< 15)")

    if reasons:
        result['step1_emergency'] = 'EMERGENCY'
        result['alert']           = True
        result['alert_reason']    = ' | '.join(reasons)
        for key in ['step3_diuretic', 'step4_raas', 'step5_bb', 'step6_sglt2', 'step6_mra']:
            result[key] = 'HOLD_EMERGENCY'
        return result  # Stop — do not proceed with any drug decisions

    # ════════════════════════════════════════════════════════════
    # STEP 2 — Fluid Classification
    # Uses resp_rate as proxy for thoracic impedance
    # (In real deployment: use impedance patch directly)
    # ════════════════════════════════════════════════════════════
    if not pd.isna(resp):
        if   resp > 22: result['step2_fluid'] = 'WET'
        elif resp < 16: result['step2_fluid'] = 'DRY'
        else:           result['step2_fluid'] = 'BORDERLINE'
    fluid = result['step2_fluid']

    # ════════════════════════════════════════════════════════════
    # STEP 3 — Diuretic Decision (Furosemide / Torsemide)
    # ════════════════════════════════════════════════════════════
    if fluid == 'WET':
        if not pd.isna(creat) and creat > 2.0:
            result['step3_diuretic'] = 'ESCALATE'       # Kidney struggling → IV route
        elif not pd.isna(potassium) and potassium < 3.5:
            result['step3_diuretic'] = 'REDUCE'          # Low K+ → losing too much
        else:
            result['step3_diuretic'] = 'INCREASE'        # WET + labs safe → increase
    elif fluid == 'DRY':
        if not pd.isna(creat) and creat > 1.5:
            result['step3_diuretic'] = 'REDUCE'          # Over-diuresis
        else:
            result['step3_diuretic'] = 'HOLD'
    else:
        result['step3_diuretic'] = 'HOLD'

    # ════════════════════════════════════════════════════════════
    # STEP 4 — RAAS Inhibitor (ARNI preferred: Sacubitril/Valsartan)
    # All 3 gates must pass simultaneously
    # ════════════════════════════════════════════════════════════
    gate_sbp  = pd.isna(sbp)       or sbp       >= 100
    gate_k    = pd.isna(potassium) or potassium  < 5.5
    gate_egfr = pd.isna(egfr)      or egfr       >= 30

    if gate_sbp and gate_k and gate_egfr:
        result['step4_raas'] = 'UPTITRATE (ARNI preferred)'
    else:
        failed = []
        if not gate_sbp:  failed.append(f"SBP={sbp:.0f} < 100")
        if not gate_k:    failed.append(f"K+={potassium:.1f} >= 5.5")
        if not gate_egfr: failed.append(f"eGFR={egfr:.0f} < 30")
        result['step4_raas'] = f"HOLD — gate failed: {', '.join(failed)}"

    # ════════════════════════════════════════════════════════════
    # STEP 5 — Beta Blocker (Carvedilol / Metoprolol / Bisoprolol)
    # Rule: DRY before you try
    # ════════════════════════════════════════════════════════════
    if fluid in ('WET', 'BORDERLINE'):
        result['step5_bb'] = 'SKIP — dry before you try'
    else:
        hr_target = 110 if has_afib else 70   # AF patients: lenient rate control
        if not pd.isna(hr):
            if hr > hr_target:
                if has_copd:
                    result['step5_bb'] = 'UPTITRATE — Bisoprolol/Metoprolol only (COPD)'
                else:
                    result['step5_bb'] = 'UPTITRATE'
            elif hr < 50:
                result['step5_bb'] = 'REDUCE — HR too low'
            elif 50 <= hr <= 60:
                result['step5_bb'] = 'HOLD — HR borderline low'
            else:
                result['step5_bb'] = 'HOLD — HR at target'
        else:
            result['step5_bb'] = 'HOLD — no HR data'

    # ════════════════════════════════════════════════════════════
    # STEP 6 — SGLT2 Inhibitor + MRA
    # SGLT2: Dapagliflozin / Empagliflozin — fixed 10mg
    # MRA:   Spironolactone / Eplerenone
    # ════════════════════════════════════════════════════════════

    # SGLT2
    if has_t1dm:
        result['step6_sglt2'] = 'CONTRAINDICATED — T1DM'
    elif not pd.isna(egfr) and egfr < 20:
        result['step6_sglt2'] = f'HOLD — eGFR={egfr:.0f} < 20'
    else:
        result['step6_sglt2'] = 'MAINTAIN / START 10mg'

    # MRA
    mra_k_ok    = pd.isna(potassium) or potassium <= 5.0
    mra_egfr_ok = pd.isna(egfr)      or egfr      >= 30
    mra_cr_ok   = pd.isna(creat)     or creat      <= 2.5

    if mra_k_ok and mra_egfr_ok and mra_cr_ok:
        result['step6_mra'] = 'MAINTAIN / ADD'
    elif not pd.isna(potassium) and potassium > 5.5:
        result['step6_mra'] = f'REDUCE — K+={potassium:.1f} > 5.5'
    else:
        result['step6_mra'] = 'HOLD — safety threshold not met'

    # ════════════════════════════════════════════════════════════
    # STEP 7 — Trajectory Check
    # Simplified: checks current row HR + SBP pattern
    # Full version: rolling window across last 3 readings
    # ════════════════════════════════════════════════════════════
    if not pd.isna(hr) and not pd.isna(sbp):
        if hr > 100 and sbp < 100:
            result['step7_trajectory'] = 'WORSENING'    # haemodynamic stress pattern
        elif hr < 80 and sbp > 110:
            result['step7_trajectory'] = 'IMPROVING'
        else:
            result['step7_trajectory'] = 'STABLE'

    return result


# ════════════════════════════════════════════════════════════════
# VALIDATE COLUMNS
# ════════════════════════════════════════════════════════════════
REQUIRED_COLS = [
    'subject_id', 'hadm_id', 'charttime',
    'heart_rate', 'sbp', 'spo2', 'resp_rate',
    'creatinine', 'potassium', 'egfr',
    'has_afib', 'has_t1dm', 'has_copd'
]

def validate_columns(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"\n❌ ERROR: Missing required columns: {missing}")
        print(f"   Your CSV has: {list(df.columns)}")
        print(f"\n   Required columns: {REQUIRED_COLS}")
        sys.exit(1)
    # Fill optional comorbidity columns with 0 if absent
    for col in ['has_afib', 'has_t1dm', 'has_copd']:
        if col not in df.columns:
            df[col] = 0
    return df


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='PIXEL MINDS — HFrEF Medication Titration Logic Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input',    type=str, default='hfref_final_dataset.csv',
                        help='Path to input CSV file (default: hfref_final_dataset.csv)')
    parser.add_argument('--output',   type=str, default=None,
                        help='Path to save results CSV (default: auto-named with timestamp)')
    parser.add_argument('--patients', type=int, default=100,
                        help='Number of unique patients to run (default: 100)')
    parser.add_argument('--all',      action='store_true',
                        help='Run on ALL patients in the dataset (overrides --patients)')
    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     PIXEL MINDS — HFrEF Medication Titration Logic Engine       ║
║     University at Buffalo · CDA Project · Spring 2026           ║
║     Client: Dr. Ciprian Ionita, QAS.AI                          ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # ── Load data ─────────────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        print(f"   Download MIMIC-IV from https://physionet.org/content/mimiciv/")
        print(f"   Or provide your own CSV with: --input your_file.csv")
        sys.exit(1)

    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')
    df = validate_columns(df)

    print(f"  ✓ {len(df):,} rows | {df['subject_id'].nunique():,} unique patients")

    # ── Select patients ───────────────────────────────────────────
    all_patients = df['subject_id'].unique()
    if args.all:
        selected = all_patients
        print(f"  Running on ALL {len(selected):,} patients")
    else:
        n = min(args.patients, len(all_patients))
        selected = all_patients[:n]
        print(f"  Running on {n} patients ({len(df[df['subject_id'].isin(selected)]):,} rows)")

    sample = df[df['subject_id'].isin(selected)].copy()

    # ── Run logic ─────────────────────────────────────────────────
    print(f"\nRunning 7-step logic engine...")
    results = sample.apply(run_logic, axis=1, result_type='expand')
    print(f"  ✓ {len(results):,} decisions made\n")

    # ── Print summary ─────────────────────────────────────────────
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  RESULTS SUMMARY")
    print(f"{sep}")

    steps = [
        ("Step 1 — Emergency",    'step1_emergency'),
        ("Step 2 — Fluid Status", 'step2_fluid'),
        ("Step 3 — Diuretic",     'step3_diuretic'),
        ("Step 4 — RAAS",         'step4_raas'),
        ("Step 5 — Beta Blocker", 'step5_bb'),
        ("Step 6 — SGLT2",        'step6_sglt2'),
        ("Step 6 — MRA",          'step6_mra'),
        ("Step 7 — Trajectory",   'step7_trajectory'),
    ]
    for label, col in steps:
        print(f"\n  {label}:")
        counts = results[col].value_counts()
        for val, cnt in counts.items():
            pct = cnt / len(results) * 100
            print(f"    {val:<40} {cnt:>8,}  ({pct:.1f}%)")

    print(f"\n  Total alerts triggered: {results['alert'].sum():,} "
          f"({results['alert'].sum()/len(results)*100:.1f}% of rows)")

    if results['alert'].sum() > 0:
        print(f"\n  Top alert reasons:")
        top = results[results['alert']]['alert_reason'].value_counts().head(5)
        for reason, cnt in top.items():
            print(f"    {reason}: {cnt:,}")

    # ── Save ──────────────────────────────────────────────────────
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"hfref_logic_results_{ts}.csv"

    results.to_csv(args.output, index=False)
    print(f"\n{'─'*55}")
    print(f"  ✅ Results saved to: {args.output}")
    print(f"{'─'*55}\n")


if __name__ == '__main__':
    main()
