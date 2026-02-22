#!/usr/bin/env python3
"""
main.py - CLI for k-anonymity disclosure simulations.

Usage:
    python main.py --data-dir /path/to/synthea/csv --output-dir results/

Author: James Weatherhead
Institution: University of Texas Medical Branch (UTMB)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd

# Import local modules
from src.data_loader import DataLoader, build_patient_profiles
from src.anonymity_engine import AnonymityEngine
from src.disclosure_models import (
    ProgressiveRefinementModel,
    RandomOrderingModel,
    RarityOrderedModel
)
from src.simulation_runner import SimulationRunner
from src.visualization import Visualizer, create_publication_figures


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Configure logging for the application."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_full_analysis(
    data_dir: str,
    output_dir: str,
    thresholds: list = [5, 11, 20],
    n_random_permutations: int = 100,
    sample_size: int = None,
    random_seed: int = 42
) -> dict:
    """
    Run the complete k-anonymity analysis pipeline.

    Args:
        data_dir: Path to Synthea CSV directory
        output_dir: Path for output files
        thresholds: k thresholds to track
        n_random_permutations: Number of random permutations per patient
        sample_size: Optional sample size (None = full population)
        random_seed: Random seed for reproducibility

    Returns:
        Dict with results summary
    """
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)

    # Step 1: Load and preprocess data
    logger.info("Loading patient data...")

    loader = DataLoader(data_dir)
    patient_profiles = loader.build_patient_profiles()

    logger.info(f"Loaded {len(patient_profiles)} patients")
    logger.info(f"Quasi-identifiers: {patient_profiles.columns.tolist()}")

    # Optional sampling - keep full population for k computation
    # We sample WHICH patients to simulate, but compute k against ALL patients
    sample_ids = None
    full_population_size = len(patient_profiles)
    if sample_size and sample_size < len(patient_profiles):
        logger.info(f"Sampling {sample_size} patients for simulation")
        logger.info(f"  -> k will be computed against FULL {full_population_size:,} patient population")
        sample_ids = patient_profiles.sample(n=sample_size, random_state=random_seed).index

    # Save population summary
    n_simulated = len(sample_ids) if sample_ids is not None else len(patient_profiles)
    summary_stats = {
        'n_patients_population': len(patient_profiles),
        'n_patients_simulated': n_simulated,
        'n_quasi_identifiers': len(patient_profiles.columns),
        'qi_columns': patient_profiles.columns.tolist(),
    }

    # Export demographic summary table
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    demo_summary = patient_profiles[['age_decade', 'gender', 'race', 'ethnicity', 'marital_status']].describe()
    demo_summary.to_csv(tables_dir / "table1_demographic_summary.csv")

    # Step 2: Run simulations
    logger.info("Running disclosure simulations...")

    # Initialize runner with FULL population (for k computation)
    runner = SimulationRunner(
        patient_profiles,
        thresholds=thresholds,
        random_seed=random_seed
    )

    # Run all three models (with optional patient sample for simulation)
    all_results = runner.run_all_models(
        patient_sample=sample_ids,  # Simulate only sampled patients, k computed against full pop
        n_random_permutations=n_random_permutations,
        show_progress=True
    )

    # Combine results
    combined_k_by_turn = runner.get_combined_k_by_turn(all_results)
    combined_threshold_stats = runner.get_combined_threshold_stats(all_results)

    # Export results
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    combined_k_by_turn.to_csv(tables_dir / "table2_k_by_turn.csv", index=False)
    combined_threshold_stats.to_csv(tables_dir / "table3_threshold_stats.csv", index=False)

    # Export raw results for each model
    sim_output_dir = output_dir / "simulation_output"
    sim_output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, results in all_results.items():
        runner.export_results(results, sim_output_dir, prefix=model_name)

    # Step 3: Generate figures
    logger.info("Generating publication figures...")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    viz = Visualizer(output_dir=str(figures_dir))

    # Figure 1: k decay curves
    logger.info("Generating Figure 1: k decay curves...")
    viz.plot_k_decay_curves(combined_k_by_turn)

    # Figure 2: Turns to threshold distribution
    logger.info("Generating Figure 2: Turns to threshold distribution...")
    viz.plot_turns_to_threshold_distribution(all_results, threshold=5)

    # Figure 4: Survival curves
    logger.info("Generating Figure 4: Survival curves...")
    viz.plot_survival_curves(combined_k_by_turn)

    # Summary panel
    logger.info("Generating summary panel...")
    viz.plot_summary_panel(all_results, combined_k_by_turn)

    # Save all figures
    figure_paths = viz.save_all_figures(formats=['pdf', 'png'])
    viz.close_all()

    # Step 4: Generate summary report
    logger.info("Generating summary report...")

    elapsed_time = time.time() - start_time

    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'elapsed_time_seconds': elapsed_time,
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'n_patients_population': len(patient_profiles),
        'n_patients_simulated': n_simulated,
        'thresholds': thresholds,
        'n_random_permutations': n_random_permutations,
        'random_seed': random_seed,
        'results_summary': {},
        'figure_paths': figure_paths,
    }

    # Add key findings
    for model_name, results in all_results.items():
        thresh_stats = results.threshold_stats
        k5_row = thresh_stats[thresh_stats['threshold'] == 5]
        k11_row = thresh_stats[thresh_stats['threshold'] == 11]

        report['results_summary'][model_name] = {
            'n_simulations': results.n_simulations,
            'pct_reaching_k5': float(k5_row['pct_reached'].values[0]) if len(k5_row) > 0 else None,
            'median_turns_to_k5': float(k5_row['median_turns'].values[0]) if len(k5_row) > 0 and pd.notna(k5_row['median_turns'].values[0]) else None,
            'pct_reaching_k11': float(k11_row['pct_reached'].values[0]) if len(k11_row) > 0 else None,
            'median_turns_to_k11': float(k11_row['median_turns'].values[0]) if len(k11_row) > 0 and pd.notna(k11_row['median_turns'].values[0]) else None,
        }

    # Save report
    report_path = output_dir / "analysis_report.yaml"
    with open(report_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False)

    logger.info(f"Analysis complete in {elapsed_time:.1f} seconds")
    logger.info(f"Results saved to: {output_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Population size (for k):  {len(patient_profiles):,}")
    print(f"Patients simulated:       {n_simulated:,}")
    print(f"Thresholds tracked:       {thresholds}")
    print(f"\nKey findings:")

    for model_name, summary in report['results_summary'].items():
        print(f"\n  {model_name.upper()}:")
        if summary['pct_reaching_k5'] is not None:
            print(f"    - {summary['pct_reaching_k5']:.1f}% reached k < 5")
        if summary['median_turns_to_k5'] is not None:
            print(f"    - Median turns to k < 5: {summary['median_turns_to_k5']:.1f}")
        if summary['pct_reaching_k11'] is not None:
            print(f"    - {summary['pct_reaching_k11']:.1f}% reached k < 11")

    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)

    return report


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="K-Anonymity Collapse in Multi-Turn Clinical LLM Conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python main.py --data-dir /path/to/synthea/csv

  # Run with custom output directory and sample size
  python main.py --data-dir /path/to/csv --output-dir results/ --sample-size 10000

  # Run with custom thresholds
  python main.py --data-dir /path/to/csv --thresholds 3 5 11 20

Author: James Weatherhead (jacweath@utmb.edu)
Institution: University of Texas Medical Branch (UTMB)
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=os.environ.get('SYNTHEA_DATA_DIR', './data'),
        help='Path to directory containing Synthea CSV files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Path for output files (default: results/)'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--thresholds',
        type=int,
        nargs='+',
        default=[5, 11, 20],
        help='k thresholds to track (default: 5 11 20)'
    )

    parser.add_argument(
        '--n-permutations',
        type=int,
        default=100,
        help='Number of random permutations per patient (default: 100)'
    )

    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Sample size for analysis (default: full population)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (optional)'
    )

    args = parser.parse_args()

    # Setup logging
    log_file = args.log_file or str(Path(args.output_dir) / 'simulation.log')
    setup_logging(args.log_level, log_file)

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        args.data_dir = config.get('data', {}).get('base_dir', args.data_dir)
        args.thresholds = config.get('simulation', {}).get('thresholds', args.thresholds)
        args.n_permutations = config.get('simulation', {}).get('n_random_permutations', args.n_permutations)
        args.seed = config.get('simulation', {}).get('random_seed', args.seed)

    # Run analysis
    try:
        report = run_full_analysis(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            thresholds=args.thresholds,
            n_random_permutations=args.n_permutations,
            sample_size=args.sample_size,
            random_seed=args.seed
        )
        return 0
    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
