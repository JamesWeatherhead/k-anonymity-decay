#!/usr/bin/env python3
"""
Regenerate fig3 (histogram) and fig4 (survival curves) with improved x-axis.

Changes from original:
- X-axis now has tick marks at every integer (1, 2, 3, ..., 11) instead of every 2
- This makes the median dashed line align with tick marks rather than falling between them

Author: James Weatherhead, UTMB
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Publication-quality matplotlib settings
STYLE_CONFIG = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Color palette (colorblind-friendly) - matching original
COLORS = {
    'progressive': '#2E86AB',  # Blue
    'random': '#F6AE2D',       # Orange/Gold
    'rarity': '#E94F37',       # Red
}

MODEL_NAMES = {
    'progressive': 'Progressive Refinement',
    'random': 'Random Ordering',
    'rarity': 'Rarity-Ordered',
}

# Apply style
plt.rcParams.update(STYLE_CONFIG)


def load_data(results_dir: Path):
    """Load k_by_turn and threshold_stats data."""
    k_by_turn = pd.read_csv(results_dir / 'tables' / 'table2_k_by_turn.csv')
    threshold_stats = pd.read_csv(results_dir / 'tables' / 'table3_threshold_stats.csv')
    return k_by_turn, threshold_stats


def compute_turns_distribution(k_by_turn: pd.DataFrame, threshold: int = 5) -> dict:
    """
    Compute the distribution of turns to reach threshold from k_by_turn data.

    Uses the difference in cumulative pct_k_lt_{threshold} to get the
    proportion reaching threshold AT each turn.
    """
    pct_col = f'pct_k_lt_{threshold}'
    distributions = {}

    for model_key in MODEL_NAMES.keys():
        model_data = k_by_turn[k_by_turn['model'] == model_key].sort_values('turn')

        if pct_col not in model_data.columns:
            continue

        turns = model_data['turn'].values
        pct_below = model_data[pct_col].values

        # Compute proportion reaching threshold at each turn
        # (difference in cumulative percentage)
        pct_at_turn = np.diff(pct_below, prepend=0)

        # Normalize to sum to 1 for density plot
        total = pct_at_turn.sum()
        if total > 0:
            density = pct_at_turn / total
        else:
            density = pct_at_turn

        distributions[model_key] = {
            'turns': turns,
            'density': density,
            'pct_at_turn': pct_at_turn,
        }

    return distributions


def plot_histogram_improved(
    k_by_turn: pd.DataFrame,
    threshold_stats: pd.DataFrame,
    threshold: int = 5,
    output_path: Path = None
) -> plt.Figure:
    """
    Figure 3: Distribution of Steps to Reach k < 5

    IMPROVED: X-axis now has ticks at every integer (1-11)
    """
    title = f"Distribution of Steps to Reach k < {threshold}"

    # Compute turn distributions from k_by_turn data
    distributions = compute_turns_distribution(k_by_turn, threshold)

    # Create figure with 3 subplots
    n_models = len(MODEL_NAMES)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)

    for ax, (model_key, model_name) in zip(axes, MODEL_NAMES.items()):
        if model_key not in distributions:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_name, fontweight='bold')
            continue

        dist = distributions[model_key]
        turns = dist['turns']
        density = dist['density']

        # Get color
        color = COLORS[model_key]

        # Plot histogram bars
        # Only plot turns where density > 0 (people actually reached threshold)
        mask = density > 0
        bar_turns = turns[mask]
        bar_density = density[mask]

        if len(bar_turns) > 0:
            ax.bar(bar_turns, bar_density, width=0.8, alpha=0.7,
                   color=color, edgecolor='white', linewidth=1)

        # Get median from threshold_stats
        model_thresh = threshold_stats[
            (threshold_stats['model'] == model_key) &
            (threshold_stats['threshold'] == threshold)
        ]

        if len(model_thresh) > 0:
            median = model_thresh['median_turns'].values[0]
            ax.axvline(median, color='black', linestyle='--', linewidth=2,
                      label=f'Median: {median:.1f}')

        # IMPROVED: Set x-axis ticks at every integer from 1 to 11
        ax.set_xticks(range(1, 12))
        ax.set_xlim(0.5, 11.5)

        # Formatting
        ax.set_xlabel('Steps', fontweight='bold')
        if ax == axes[0]:
            ax.set_ylabel('Proportion', fontweight='bold')
        ax.set_title(model_name, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(title, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

        # Also save PDF
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {pdf_path}")

    return fig


def plot_survival_improved(
    k_by_turn: pd.DataFrame,
    thresholds: list = [5, 11],
    output_path: Path = None
) -> plt.Figure:
    """
    Figure 4: Survival Above Small-Cell Thresholds

    IMPROVED: X-axis now has ticks at every integer (1-11)
    """
    title = "Survival Above Small-Cell Thresholds"

    # Create figure with 3 subplots
    n_models = len(MODEL_NAMES)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)

    # Line styles for different thresholds
    linestyles = {5: '-', 11: '--', 20: ':'}

    # Colors for survival lines (using a distinct palette)
    survival_colors = {5: '#2E86AB', 11: '#F6AE2D'}  # Blue and Orange

    for ax, (model_key, model_name) in zip(axes, MODEL_NAMES.items()):
        model_data = k_by_turn[k_by_turn['model'] == model_key].sort_values('turn')

        if len(model_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_name, fontweight='bold')
            continue

        turns = model_data['turn'].values
        max_turn = int(max(turns))

        for threshold in thresholds:
            pct_col = f'pct_k_lt_{threshold}'
            if pct_col not in model_data.columns:
                continue

            # Survival = 100% - percentage below threshold
            survival = 100 - model_data[pct_col].values

            linestyle = linestyles.get(threshold, '-')
            color = survival_colors.get(threshold, '#666666')

            ax.step(turns, survival, where='post', linewidth=2,
                   linestyle=linestyle, color=color, label=f'k ≥ {threshold}')

        # IMPROVED: Set x-axis ticks at every integer from 1 to 11
        ax.set_xticks(range(1, max_turn + 1))
        ax.set_xlim(0.5, max_turn + 0.5)

        # Formatting
        ax.set_xlabel('Disclosure Step', fontweight='bold')
        if ax == axes[0]:
            ax.set_ylabel('% Patients Above Threshold', fontweight='bold')
        ax.set_title(model_name, fontweight='bold')
        ax.set_ylim(-5, 105)
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

        # Also save PDF
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {pdf_path}")

    return fig


def main():
    # Paths
    project_dir = Path(__file__).parent.parent
    results_dir = project_dir / 'results'
    assets_dir = project_dir / 'assets'

    # Ensure assets directory exists
    assets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {results_dir}")

    # Load data
    k_by_turn, threshold_stats = load_data(results_dir)

    print(f"Loaded k_by_turn: {len(k_by_turn)} rows")
    print(f"Loaded threshold_stats: {len(threshold_stats)} rows")

    # Generate Figure 3 (histogram)
    print("\n--- Generating Figure 3 (Histogram) ---")
    fig3 = plot_histogram_improved(
        k_by_turn,
        threshold_stats,
        threshold=5,
        output_path=assets_dir / 'fig3.png'
    )

    # Generate Figure 4 (survival curves)
    print("\n--- Generating Figure 4 (Survival Curves) ---")
    fig4 = plot_survival_improved(
        k_by_turn,
        thresholds=[5, 11],
        output_path=assets_dir / 'fig4.png'
    )

    print("\n✓ Done! Updated figures saved to assets/")
    print("  - fig3.png (+ fig3.pdf)")
    print("  - fig4.png (+ fig4.pdf)")

    plt.show()


if __name__ == "__main__":
    main()
