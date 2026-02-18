import argparse
import os
from datetime import datetime

from src.marketing.pipeline import run_marketing_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Marketing pipeline runner")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/input/bank-full.csv",
        help="Path to bank-full.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output/marketing",
        help="Base output directory. A timestamped subfolder will be created under this directory.",
    )
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Top-K rows for call_list_top200.csv. rank is assigned as 1..K.",
    )
    parser.add_argument(
        "--enable-edl",
        action="store_true",
        help="Enable EDL model and uncertainty outputs.",
    )
    parser.add_argument(
        "--run-eda",
        action="store_true",
        help="Run minimal EDA and save into the run output directory.",
    )
    parser.add_argument(
        "--debug-call-list",
        action="store_true",
        help="Add model probabilities and uncertainty columns to call_list_top200.csv",
    )
    parser.add_argument(
        "--debug-score-columns",
        action="store_true",
        help="Backward-compatible alias of --debug-call-list",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default="mean_proba",
        choices=["mean_proba", "edl_uncertainty_weighted"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Always create a timestamped run directory under the provided base output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, run_id)

    run_marketing_pipeline(
        data_path=args.data_path,
        output_dir=output_dir,
        target_col=args.target_col,
        n_splits=args.n_splits,
        random_state=args.seed,
        top_k=args.top_k,
        include_debug_columns=(args.debug_call_list or args.debug_score_columns),
        enable_edl=args.enable_edl,
        score_mode=args.score_mode,
        run_eda=args.run_eda,
    )


if __name__ == "__main__":
    main()
