import argparse
import concurrent.futures
import subprocess

EXPERIMENTS = [
    {
        "script": "ChessPPO.py",
        "args": [
            "--total-timesteps", "1000",
            "--n-envs", "10",
            "--n-steps", "2048",
            "--batch-size", "512",
            "--n-epochs", "10",
        ]
    },
    {
        "script": "ChessMaskablePPO.py",
        "args": [
            "--total-timesteps", "1000",
            "--n-envs", "10",
            "--n-steps", "2048",
            "--batch-size", "512",
            "--n-epochs", "10",
        ]
    }
]

def build_cmd(script, args):
    return ["python", script] + args

def run_exp(exp):
    cmd = build_cmd(exp["script"], exp["args"])
    # Use single quotes inside f-string to avoid escaping
    print(f"Running script: {exp['script']} with args {exp['args']}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--parallel",    action="store_true",
                        help="Run experiments in parallel")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="How many processes to spawn in parallel mode")
    args = parser.parse_args()
    
    print("Parsed arguments:")
    for name, val in vars(args).items():
        print(f"  {name}: {val}")

    if not args.parallel:
        print("--> Running experiments sequentially...")
        for exp in EXPERIMENTS:
            run_exp(exp)
    else:
        print(f"--> Running experiments in parallel with {args.max_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            pool.map(run_exp, EXPERIMENTS)
