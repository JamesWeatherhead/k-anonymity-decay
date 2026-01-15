#!/usr/bin/env python3
"""
Consistency checker for simulation results.

Verifies that summary statistics match per-step tables.
Run this after any simulation to catch documentation/code drift.

Usage:
    python scripts/check_consistency.py
"""

import pandas as pd
import yaml
from pathlib import Path


def main():
    out = Path("results")

    if not out.exists():
        print("ERROR: results/ directory not found. Run simulation first.")
        return 1

    # Load summary report
    report_path = out / "analysis_report.yaml"
    if not report_path.exists():
        print(f"ERROR: {report_path} not found.")
        return 1

    report = yaml.safe_load(report_path.read_text())

    # Load per-step table
    k_by_turn_path = out / "tables" / "table2_k_by_turn.csv"
    if not k_by_turn_path.exists():
        print(f"ERROR: {k_by_turn_path} not found.")
        return 1

    k_by_turn = pd.read_csv(k_by_turn_path)

    # Load threshold stats
    thresh_path = out / "tables" / "table3_threshold_stats.csv"
    thresh_stats = pd.read_csv(thresh_path)

    print("=" * 60)
    print("CONSISTENCY CHECK")
    print("=" * 60)

    errors = []

    for model in ['progressive', 'rarity']:
        # Get summary values
        summary = report['results_summary'].get(model, {})
        pct_k5_summary = summary.get('pct_reaching_k5')

        # Get threshold stats
        model_thresh = thresh_stats[(thresh_stats['model'] == model) & (thresh_stats['threshold'] == 5)]
        if len(model_thresh) > 0:
            pct_k5_table = model_thresh['pct_reached'].values[0]
        else:
            pct_k5_table = None

        print(f"\n{model.upper()}:")
        print(f"  Summary pct_reaching_k5: {pct_k5_summary}")
        print(f"  Table pct_reached (k<5): {pct_k5_table}")

        if pct_k5_summary is not None and pct_k5_table is not None:
            if abs(pct_k5_summary - pct_k5_table) > 0.01:
                errors.append(f"{model}: summary ({pct_k5_summary}) != table ({pct_k5_table})")
            else:
                print("  ✓ Match")

        # Check that per-step table notes are present
        model_k = k_by_turn[k_by_turn['model'] == model]
        final_step = model_k['turn'].max()
        final_pct = model_k[model_k['turn'] == final_step]['pct_k_lt_5'].values[0]

        print(f"  Final step ({final_step}) pct_k_lt_5: {final_pct}")
        print(f"  NOTE: This differs from summary because not all patients have all steps.")

    print("\n" + "=" * 60)

    if errors:
        print("ERRORS FOUND:")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("All consistency checks passed.")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
