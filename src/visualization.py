"""
visualization.py - Publication-quality figures for k-anonymity analysis.

Author: James Weatherhead
Institution: University of Texas Medical Branch (UTMB)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

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

# Color palette (colorblind-friendly)
COLORS = {
    'progressive': '#2E86AB',      # Blue
    'random': '#F6AE2D',           # Orange/Gold
    'rarity': '#E94F37',           # Red
    'threshold_5': '#666666',      # Dark gray
    'threshold_11': '#999999',     # Medium gray
    'threshold_20': '#CCCCCC',     # Light gray
    'ci_fill': '#DDDDDD',          # Very light gray for CI bands
}

MODEL_NAMES = {
    'progressive': 'Progressive Refinement',
    'random': 'Random Ordering',
    'rarity': 'Rarity-Ordered',
}


class Visualizer:
    """Creates publication-quality figures for k-anonymity analysis."""

    def __init__(
        self,
        output_dir: str = "results/figures",
        figsize: Tuple[float, float] = (7, 5),
        dpi: int = 300
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi

        # Apply style
        plt.rcParams.update(STYLE_CONFIG)

        # Storage for generated figures
        self._figures: Dict[str, plt.Figure] = {}

        logger.info(f"Visualizer initialized, output_dir: {self.output_dir}")

    def plot_k_decay_curves(
        self,
        k_by_turn: pd.DataFrame,
        title: str = "Anonymity Set Size (k) Across Disclosure Turns",
        thresholds: List[int] = [5, 11, 20],
        log_scale: bool = True,
        show_ci: bool = True
    ) -> plt.Figure:
        """Figure 1: k decay curves showing median k with IQR bands by disclosure model."""
        fig, ax = plt.subplots(figsize=self.figsize)

        for model_key, model_name in MODEL_NAMES.items():
            model_data = k_by_turn[k_by_turn['model'] == model_key].sort_values('turn')

            if len(model_data) == 0:
                continue

            turns = model_data['turn'].values
            median_k = model_data['median_k'].values

            # Plot median line
            color = COLORS[model_key]
            ax.plot(turns, median_k, '-o', color=color, label=model_name,
                   linewidth=2, markersize=6, markeredgecolor='white', markeredgewidth=1)

            # Add IQR band
            if show_ci and 'q25_k' in model_data.columns and 'q75_k' in model_data.columns:
                q25 = model_data['q25_k'].values
                q75 = model_data['q75_k'].values
                ax.fill_between(turns, q25, q75, alpha=0.2, color=color)

        # Add threshold reference lines
        for threshold in thresholds:
            linestyle = '--' if threshold == 5 else ':'
            ax.axhline(y=threshold, color=COLORS.get(f'threshold_{threshold}', '#999999'),
                      linestyle=linestyle, linewidth=1.5, alpha=0.7,
                      label=f'k = {threshold}')

        # Formatting
        ax.set_xlabel('Disclosure Turn', fontweight='bold')
        ax.set_ylabel('Anonymity Set Size (k)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)

        if log_scale:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        ax.set_xlim(0.5, max(k_by_turn['turn']) + 0.5)
        ax.set_xticks(range(1, int(max(k_by_turn['turn'])) + 1))

        ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        plt.tight_layout()

        self._figures['fig1_k_decay_curves'] = fig
        return fig

    def plot_turns_to_threshold_distribution(
        self,
        all_results: Dict[str, Any],
        threshold: int = 5,
        title: str = None
    ) -> plt.Figure:
        """Figure 2: Histogram of turns to reach small-cell threshold by model."""
        if title is None:
            title = f"Distribution of Turns to Reach k < {threshold}"

        # Collect turns data from all models
        turns_data = []
        for model_key, results in all_results.items():
            col_name = f'turns_to_k{threshold}'
            for result in results.raw_results:
                turns = result.turns_to_thresholds.get(threshold)
                if turns is not None:
                    turns_data.append({
                        'model': MODEL_NAMES.get(model_key, model_key),
                        'turns': turns
                    })

        turns_df = pd.DataFrame(turns_data)

        if len(turns_df) == 0:
            logger.warning(f"No data for threshold {threshold}")
            return None

        # Create figure with subplots for each model
        models = list(MODEL_NAMES.values())
        n_models = len(models)

        fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)

        if n_models == 1:
            axes = [axes]

        for ax, (model_key, model_name) in zip(axes, MODEL_NAMES.items()):
            model_turns = turns_df[turns_df['model'] == model_name]['turns']

            if len(model_turns) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(model_name, fontweight='bold')
                continue

            # Histogram with density
            color = COLORS[model_key]
            ax.hist(model_turns, bins=range(1, int(model_turns.max()) + 2),
                   density=True, alpha=0.7, color=color, edgecolor='white')

            # Add summary statistics
            median = model_turns.median()
            mean = model_turns.mean()
            ax.axvline(median, color='black', linestyle='--', linewidth=2, label=f'Median: {median:.1f}')

            # Formatting
            ax.set_xlabel('Turns', fontweight='bold')
            if ax == axes[0]:
                ax.set_ylabel('Proportion', fontweight='bold')
            ax.set_title(model_name, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xlim(0.5, None)

        fig.suptitle(title, fontweight='bold', y=1.02)
        plt.tight_layout()

        self._figures['fig2_turns_to_threshold'] = fig
        return fig

    def plot_qi_impact_heatmap(
        self,
        qi_impact_data: pd.DataFrame,
        title: str = "Quasi-Identifier Impact on Anonymity Set Size"
    ) -> plt.Figure:
        """Figure 3: Heatmap of QI impact on k reduction."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create heatmap
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(qi_impact_data, ax=ax, cmap='YlOrRd', annot=True,
                   fmt='.1f', linewidths=0.5, cbar_kws={'label': 'Median % k Reduction'})

        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_xlabel('Turn Position', fontweight='bold')
        ax.set_ylabel('Quasi-Identifier', fontweight='bold')

        plt.tight_layout()

        self._figures['fig3_qi_impact_heatmap'] = fig
        return fig

    def plot_survival_curves(
        self,
        k_by_turn: pd.DataFrame,
        thresholds: List[int] = [5, 11, 20],
        title: str = "Survival Above Small-Cell Thresholds"
    ) -> plt.Figure:
        """Figure 4: Proportion of patients remaining above k thresholds by turn."""
        fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(4 * len(MODEL_NAMES), 4), sharey=True)

        for ax, (model_key, model_name) in zip(axes, MODEL_NAMES.items()):
            model_data = k_by_turn[k_by_turn['model'] == model_key].sort_values('turn')

            if len(model_data) == 0:
                continue

            turns = model_data['turn'].values

            for threshold in thresholds:
                pct_col = f'pct_k_lt_{threshold}'
                if pct_col in model_data.columns:
                    # Survival = 100% - percentage below threshold
                    survival = 100 - model_data[pct_col].values
                    linestyle = '-' if threshold == 5 else ('--' if threshold == 11 else ':')
                    ax.step(turns, survival, where='post', linewidth=2,
                           linestyle=linestyle, label=f'k ≥ {threshold}')

            ax.set_xlabel('Disclosure Turn', fontweight='bold')
            if ax == axes[0]:
                ax.set_ylabel('% Patients Above Threshold', fontweight='bold')
            ax.set_title(model_name, fontweight='bold')
            ax.set_ylim(-5, 105)
            ax.set_xlim(0.5, max(turns) + 0.5)
            ax.legend(loc='lower left', fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontweight='bold', y=1.02)
        plt.tight_layout()

        self._figures['fig4_survival_curves'] = fig
        return fig

    def plot_rare_concept_scatter(
        self,
        concept_data: pd.DataFrame,
        title: str = "Rare Concept Disclosure Impact"
    ) -> plt.Figure:
        """Figure 5: QI frequency vs. privacy impact scatter plot."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Color by QI category
        categories = concept_data['category'].unique() if 'category' in concept_data.columns else ['all']
        palette = sns.color_palette('husl', len(categories))

        for cat, color in zip(categories, palette):
            if 'category' in concept_data.columns:
                mask = concept_data['category'] == cat
                data = concept_data[mask]
            else:
                data = concept_data
                cat = 'All QIs'

            ax.scatter(data['frequency'], data['k_reduction'],
                      alpha=0.6, s=50, label=cat, color=color)

        ax.set_xlabel('Population Frequency (log scale)', fontweight='bold')
        ax.set_ylabel('Median k Reduction (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_xscale('log')

        # Add reference line for rare threshold
        ax.axvline(x=0.001, color='red', linestyle='--', alpha=0.5, label='Rare threshold (0.1%)')

        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        self._figures['fig5_rare_concept_scatter'] = fig
        return fig

    def plot_summary_panel(
        self,
        all_results: Dict[str, Any],
        k_by_turn: pd.DataFrame,
        title: str = "Multi-Turn Disclosure Risk Summary"
    ) -> plt.Figure:
        """Summary panel combining key visualizations for graphical abstract."""
        fig = plt.figure(figsize=(14, 10))

        # Create grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Panel A: k decay curves
        ax1 = fig.add_subplot(gs[0, 0])
        for model_key, model_name in MODEL_NAMES.items():
            model_data = k_by_turn[k_by_turn['model'] == model_key].sort_values('turn')
            if len(model_data) > 0:
                ax1.plot(model_data['turn'], model_data['median_k'],
                        '-o', color=COLORS[model_key], label=model_name, linewidth=2)

        ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.7)
        ax1.axhline(y=11, color='gray', linestyle=':', alpha=0.7)
        ax1.set_yscale('log')
        ax1.set_xlabel('Disclosure Turn')
        ax1.set_ylabel('Median k (log scale)')
        ax1.set_title('A) k Decay by Model', fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel B: Survival curves for k<5
        ax2 = fig.add_subplot(gs[0, 1])
        for model_key, model_name in MODEL_NAMES.items():
            model_data = k_by_turn[k_by_turn['model'] == model_key].sort_values('turn')
            if len(model_data) > 0 and 'pct_k_lt_5' in model_data.columns:
                survival = 100 - model_data['pct_k_lt_5'].values
                ax2.step(model_data['turn'], survival, where='post',
                        color=COLORS[model_key], label=model_name, linewidth=2)

        ax2.set_xlabel('Disclosure Turn')
        ax2.set_ylabel('% Patients with k ≥ 5')
        ax2.set_title('B) Survival Above k=5', fontweight='bold')
        ax2.set_ylim(-5, 105)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel C: Bar chart of median turns to k<5
        ax3 = fig.add_subplot(gs[1, 0])
        medians = []
        model_names_list = []
        colors_list = []

        for model_key, model_name in MODEL_NAMES.items():
            if model_key in all_results:
                thresh_stats = all_results[model_key].threshold_stats
                k5_row = thresh_stats[thresh_stats['threshold'] == 5]
                if len(k5_row) > 0 and pd.notna(k5_row['median_turns'].values[0]):
                    medians.append(k5_row['median_turns'].values[0])
                    model_names_list.append(model_name)
                    colors_list.append(COLORS[model_key])

        if medians:
            bars = ax3.bar(model_names_list, medians, color=colors_list, edgecolor='white', linewidth=1.5)
            ax3.set_ylabel('Median Turns to k < 5')
            ax3.set_title('C) Turns to Small-Cell', fontweight='bold')

            # Add value labels on bars
            for bar, val in zip(bars, medians):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)

        # Panel D: Table with key statistics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        # Create summary table
        table_data = []
        for model_key, model_name in MODEL_NAMES.items():
            if model_key in all_results:
                thresh_stats = all_results[model_key].threshold_stats

                k5_row = thresh_stats[thresh_stats['threshold'] == 5]
                k11_row = thresh_stats[thresh_stats['threshold'] == 11]

                pct_k5 = k5_row['pct_reached'].values[0] if len(k5_row) > 0 else 'N/A'
                pct_k11 = k11_row['pct_reached'].values[0] if len(k11_row) > 0 else 'N/A'

                table_data.append([
                    model_name,
                    f"{pct_k5:.1f}%" if isinstance(pct_k5, (int, float)) else pct_k5,
                    f"{pct_k11:.1f}%" if isinstance(pct_k11, (int, float)) else pct_k11,
                ])

        table = ax4.table(
            cellText=table_data,
            colLabels=['Model', '% Reaching k<5', '% Reaching k<11'],
            loc='center',
            cellLoc='center',
            colColours=['#f0f0f0'] * 3
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('D) Threshold Statistics', fontweight='bold', pad=20)

        fig.suptitle(title, fontweight='bold', fontsize=14, y=0.98)

        self._figures['summary_panel'] = fig
        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        name: str,
        formats: List[str] = ['pdf', 'png']
    ) -> List[str]:
        """Save figure in multiple formats. Returns list of saved paths."""
        saved_paths = []

        for fmt in formats:
            filepath = self.output_dir / f"{name}.{fmt}"
            fig.savefig(filepath, format=fmt, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            saved_paths.append(str(filepath))
            logger.info(f"Saved figure: {filepath}")

        return saved_paths

    def save_all_figures(
        self,
        formats: List[str] = ['pdf', 'png']
    ) -> Dict[str, List[str]]:
        """Save all generated figures. Returns dict of figure names to paths."""
        all_paths = {}

        for name, fig in self._figures.items():
            paths = self.save_figure(fig, name, formats)
            all_paths[name] = paths

        logger.info(f"Saved {len(all_paths)} figures to {self.output_dir}")

        return all_paths

    def close_all(self) -> None:
        """Close all figures to free memory."""
        for fig in self._figures.values():
            plt.close(fig)
        self._figures.clear()


def create_publication_figures(
    all_results: Dict[str, Any],
    k_by_turn: pd.DataFrame,
    output_dir: str = "results/figures"
) -> Dict[str, str]:
    """Generate all publication figures. Returns dict of figure names to paths."""
    viz = Visualizer(output_dir=output_dir)

    # Generate all figures
    viz.plot_k_decay_curves(k_by_turn)
    viz.plot_turns_to_threshold_distribution(all_results, threshold=5)
    viz.plot_survival_curves(k_by_turn)
    viz.plot_summary_panel(all_results, k_by_turn)

    # Save all figures
    paths = viz.save_all_figures()

    viz.close_all()

    return paths


if __name__ == "__main__":
    import logging
    import tempfile

    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)

    k_by_turn_data = []
    for model in ['progressive', 'random', 'rarity']:
        for turn in range(1, 10):
            k_base = 10000 * np.exp(-0.3 * turn) + np.random.normal(0, 100)
            k_by_turn_data.append({
                'model': model,
                'turn': turn,
                'median_k': max(k_base, 1),
                'mean_k': max(k_base * 1.1, 1),
                'q25_k': max(k_base * 0.5, 1),
                'q75_k': max(k_base * 2, 1),
                'pct_k_lt_5': min(100, turn * 10 + np.random.normal(0, 5)),
                'pct_k_lt_11': min(100, turn * 8 + np.random.normal(0, 4)),
            })

    k_by_turn = pd.DataFrame(k_by_turn_data)
    print("Sample k_by_turn data:")
    print(k_by_turn.head(15))

    with tempfile.TemporaryDirectory() as tmpdir:
        viz = Visualizer(output_dir=tmpdir)
        fig1 = viz.plot_k_decay_curves(k_by_turn)
        fig4 = viz.plot_survival_curves(k_by_turn)
        paths = viz.save_all_figures()
        print(f"Saved figures: {paths}")
        viz.close_all()

    print("Visualization test complete.")
