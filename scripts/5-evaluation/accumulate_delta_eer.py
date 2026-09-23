import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", nargs="+", required=True)
    parser.add_argument("--group-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="hrs")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--genres", nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--authbench-dir", type=str, required=True)
    parser.add_argument("--hiatus-dir", type=str, required=True)
    args = parser.parse_args()

    results = []

    methods = args.method
    group_name = args.group_name if args.group_name else methods[0]

    for method in methods:
        for genre in args.genres:
            # AuthBench Delta EER
            genre_tag = f"_{genre.replace('/', '_')}" if genre else ""
            auth_file = Path(args.authbench_dir) / f"{args.dataset}_{args.language}{genre_tag}_{args.model_tag}_{method}.json"
            
            auth_delta = None
            if auth_file.exists():
                with open(auth_file, "r") as f:
                    data = json.load(f)
                    auth_delta = data.get("delta_eer", None)
            else:
                print(f"Warning: AuthBench result file not found: {auth_file}")

            # Hiatus Delta EER
            hiatus_file = Path(args.hiatus_dir) / f"{genre}_{method}_ta3_results.csv"
            hiatus_delta = None
            if hiatus_file.exists():
                df = pd.read_csv(hiatus_file)
                if "Delta Equal Error Rate" in df.columns:
                    hiatus_delta = df["Delta Equal Error Rate"].iloc[0]
            else:
                print(f"Warning: Hiatus result file not found: {hiatus_file}")

            results.append({
                "genre": genre,
                "method": group_name,
                "original_method": method,
                "hiatus_delta_eer": hiatus_delta,
                "authbench_delta_eer": auth_delta
            })

    df_res = pd.DataFrame(results)
    
    baseline_out_csv = Path(args.baseline_dir) / f"{group_name}_{args.language}_per_genre_delta_eer.csv"
    baseline_out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(baseline_out_csv, index=False)
    print(f"Per-genre metrics saved to {baseline_out_csv}")

    df_avg = df_res.groupby("method")[["authbench_delta_eer", "hiatus_delta_eer"]].mean().reset_index()
    df_avg = df_avg[["method", "authbench_delta_eer", "hiatus_delta_eer"]]

    out_csv = Path(args.out_dir) / f"baseline_{args.dataset}_{args.language}_delta_eer_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = out_csv.exists()
    
    if file_exists:
        try:
            df_existing = pd.read_csv(out_csv)
            if list(df_existing.columns) == ["method", "authbench_delta_eer", "hiatus_delta_eer"]:
                df_combined = pd.concat([df_existing, df_avg])
                df_combined = df_combined.drop_duplicates(subset=["method"], keep="last")
                df_combined.to_csv(out_csv, index=False)
            else:
                df_avg.to_csv(out_csv, index=False)
        except Exception:
            df_avg.to_csv(out_csv, index=False)
    else:
        df_avg.to_csv(out_csv, index=False)
        
    print(f"Accumulated summary updated in {out_csv}")

if __name__ == "__main__":
    main()
