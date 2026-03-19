#!/usr/bin/env python3
import argparse
import os
import re
import sys
from glob import glob
from typing import Optional, List, Tuple


RMSE_REGEX = re.compile(r"['\"]rmse['\"]:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")

# Sequence name mapping: full name -> short identifier
TUM_RGBD_SEQUENCE_MAPPING = {
    "freiburg2_desk_with_person": "f2/dp",
    "freiburg3_sitting_halfsphere": "f3/shs", 
    "freiburg3_sitting_rpy": "f3/sr",
    "freiburg3_sitting_static": "f3/ss",
    "freiburg3_sitting_xyz": "f3/sx",
    "freiburg3_walking_halfsphere": "f3/whs",
    "freiburg3_walking_rpy": "f3/wr", 
    "freiburg3_walking_static": "f3/ws",
    "freiburg3_walking_xyz": "f3/wx"
}

# Bonn sequence name mapping: remove bonn_ prefix
BONN_SEQUENCE_MAPPING = {
    "bonn_balloon": "balloon",
    "bonn_balloon2": "balloon2",
    "bonn_crowd": "crowd",
    "bonn_crowd2": "crowd2",
    "bonn_person_tracking": "person",
    "bonn_person_tracking2": "person2",
    "bonn_moving_nonobstructing_box": "moving",
    "bonn_moving_nonobstructing_box2": "moving2"
}

# Define the desired column order for different datasets
FREIBURG_COLUMN_ORDER = ["dp", "ss", "sx", "sr", "shs", "ws", "wx", "wr", "whs"]
BONN_COLUMN_ORDER = ["balloon", "balloon2", "crowd", "crowd2", "person", "person2", "moving", "moving2"]

def extract_rmse_from_line(line: str) -> Optional[float]:
    match = RMSE_REGEX.search(line)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def find_metrics_files(base_dir: str) -> List[str]:
    pattern = os.path.join(base_dir, "**", "traj", "metrics_full_traj.txt")
    return sorted(glob(pattern, recursive=True))


def read_ninth_line(filepath: str) -> Optional[str]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                if idx == 9:
                    return line.strip()
        return None
    except FileNotFoundError:
        return None
    except OSError as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect rmse from metrics_kf_traj.txt (line 9), multiply by 100 and format to two decimals.")
    parser.add_argument("--base", "-b", default=os.path.join("output", "TUM_RGBD_ours"), help="Base directory to scan (default: output/TUM_RGBD_ours)")
    parser.add_argument("--csv", action="store_true", help="Output in CSV format: sequence,rmse_percent")
    args = parser.parse_args()

    base_dirs = sorted(os.listdir(args.base))
    for base_dir in base_dirs:
        
        if base_dir == "Bonn":
            Dataset = "Bonn"
        elif base_dir == "TUM_RGBD":
            Dataset = "TUM_RGBD"
        elif base_dir == "DyCheck":
            Dataset = "DyCheck"
        elif base_dir == "DROID-W":
            Dataset = "DROID-W"
        else:
            print(f"Unknown dataset: {base_dir}", file=sys.stderr)
            continue

        base_dir_path = os.path.join(args.base, base_dir)
        metrics_files = find_metrics_files(base_dir_path)
        if not metrics_files:
            print(f"No metrics files found under: {base_dir_path}", file=sys.stderr)
            continue

        results: List[Tuple[str, float]] = []
        for metrics_path in metrics_files:
            ninth_line = read_ninth_line(metrics_path)
            if ninth_line is None:
                continue
            rmse_value = extract_rmse_from_line(ninth_line)
            if rmse_value is None:
                # Fallback: sometimes the rmse value might be on line 9 without key label, try to parse float directly
                try:
                    rmse_value = float(ninth_line)
                except ValueError:
                    continue
            if Dataset == "DyCheck":
                rmse_percent = rmse_value * 1.0
            else:
                rmse_percent = rmse_value * 100.0

            # sequence name is the parent directory of the "traj" folder
            traj_dir = os.path.dirname(metrics_path)
            sequence_dir = os.path.basename(os.path.dirname(traj_dir))
            
            # Use short identifier if mapping exists, otherwise use full name
            # Check mappings in order: Freiburg -> Bonn
            if sequence_dir.startswith("freiburg"):
                display_name = TUM_RGBD_SEQUENCE_MAPPING.get(sequence_dir)
            elif sequence_dir.startswith("bonn"):
                display_name = BONN_SEQUENCE_MAPPING.get(sequence_dir)
            else:
                display_name = sequence_dir
            
            results.append((display_name, rmse_percent))
            # print(f"Debug: {sequence_dir} -> {display_name}, RMSE: {rmse_percent:.3f}", file=sys.stderr)

        # Output
        # detect if results_dict is empty
        if not results:
            print("No valid rmse values parsed.", file=sys.stderr)
            continue
        
        if Dataset == "TUM_RGBD":
            print("Evaluating TUM_RGBD RMSE ...")
            print("----------------------------------------------------------")
            # Convert results to dictionary for easy lookup
            results_dict = dict(results)
            
            # Prepare ordered output based on FREIBURG_COLUMN_ORDER
            ordered_sequences = []
            ordered_values = []
            
            # Define the exact order we want (matching the table header)
            desired_order = ["f2/dp", "f3/ss", "f3/sx", "f3/sr", "f3/shs", "f3/ws", "f3/wx", "f3/wr", "f3/whs"]
            
            for seq_name in desired_order:
                if seq_name in results_dict:
                    ordered_sequences.append(seq_name)
                    ordered_values.append(results_dict[seq_name])
            
            # Calculate average
            avg = sum(ordered_values) / len(ordered_values) if ordered_values else 0

            if args.csv:
                first_row = ",".join(ordered_sequences + ["Avg."])
                second_row = ",".join([f"{v:.1f}" for v in ordered_values] + [f"{avg:.2f}"])
                print(first_row)
                print(second_row)
            else:
                # First row: sequence names
                first_row = " ".join(ordered_sequences + ["Avg."])
                print(first_row)
                # Second row: LaTeX table format
                latex_row = " & ".join([f"{v:.1f}" for v in ordered_values] + [f"{avg:.2f}"])
                print(latex_row)
        
        elif Dataset == "Bonn":
            print("Evaluating Bonn RMSE ...")
            print("----------------------------------------------------------")
            # Convert results to dictionary for easy lookup
            results_dict = dict(results)
            
            # Prepare ordered output based on BONN_COLUMN_ORDER
            ordered_sequences = []
            ordered_values = []
            
            for seq_name in BONN_COLUMN_ORDER:
                if seq_name in results_dict:
                    ordered_sequences.append(seq_name)
                    ordered_values.append(results_dict[seq_name])
            
            # Calculate average
            avg = sum(ordered_values) / len(ordered_values) if ordered_values else 0

            if args.csv:
                first_row = ",".join(ordered_sequences + ["Avg."])
                second_row = ",".join([f"{v:.1f}" for v in ordered_values] + [f"{avg:.2f}"])
                print(first_row)
                print(second_row)
            else:
                # First row: sequence names
                first_row = " ".join(ordered_sequences + ["Avg."])
                print(first_row)
                # Second row: LaTeX table format
                latex_row = " & ".join([f"{v:.1f}" for v in ordered_values] + [f"{avg:.2f}"])
                print(latex_row)
                
        elif Dataset == "DyCheck":
            print("Evaluating DyCheck RMSE ...")
            print("----------------------------------------------------------")
            # Convert results to dictionary for easy lookup
            results_dict = dict(results)
            
            # Prepare ordered output based on alphabetical order
            ordered_sequences = sorted(results_dict.keys())
            ordered_values = [results_dict[seq_name] for seq_name in ordered_sequences]
            
            # Calculate average
            avg = sum(ordered_values) / len(ordered_values) if ordered_values else 0

            if args.csv:
                first_row = ",".join(ordered_sequences + ["Avg."])
                second_row = ",".join([f"{v:.3f}" for v in ordered_values] + [f"{avg:.3f}"])
                print(first_row)
                print(second_row)
            else:
                # First row: sequence names
                first_row = " ".join(ordered_sequences + ["Avg."])
                print(first_row)
                # Second row: LaTeX table format
                latex_row = " & ".join([f"{v:.3f}" for v in ordered_values] + [f"{avg:.3f}"])
                print(latex_row)

        elif Dataset == "DROID-W":
            print("Evaluating DROID-W RMSE ...")
            print("----------------------------------------------------------")
            # Convert results to dictionary for easy lookup
            results_dict = dict(results)

            # Prepare ordered output based on alphabetical order
            ordered_sequences = sorted(results_dict.keys())
            ordered_values = [results_dict[seq_name] for seq_name in ordered_sequences]
            
            # Calculate average
            avg = sum(ordered_values) / len(ordered_values) if ordered_values else 0

            if args.csv:
                first_row = ",".join(ordered_sequences + ["Avg."])
                second_row = ",".join([f"{v/100:.2f}" for v in ordered_values] + [f"{avg/100:.3f}"])
                print(first_row)
                print(second_row)
            else:
                # First row: sequence names
                first_row = " ".join(ordered_sequences + ["Avg."])
                print(first_row)
                # Second row: LaTeX table format
                latex_row = " & ".join([f"{v/100:.2f}" for v in ordered_values] + [f"{avg/100:.3f}"])
                print(latex_row)
        else:
            print(f"Unknown dataset: {Dataset}", file=sys.stderr)
            continue

        # plot yellow dot line
        print("\033[93m" + "----------------------------------------------------------" + "\033[0m")
        print("\n")

if __name__ == "__main__":
    raise SystemExit(main())


