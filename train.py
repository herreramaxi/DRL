import argparse
import os
import subprocess
import time     
from common import build_cmd
from concurrent.futures import ThreadPoolExecutor, as_completed

from custom_logging import important, error, important2, info, success

# Examples:
# python train.py --total-timesteps 1000000 --n-envs 4 --num-repeats 3
# python train.py --total-timesteps 2000000 --n-envs 4 --num-repeats 3
# python train.py --total-timesteps 3000000 --n-envs 4 --num-repeats 3
# python train.py --total-timesteps 5000000 --n-envs 4 --num-repeats 3
PRE_PROCESSING_TASKS = [
    {
        "script": "ae_pretrain.py", "args": ["--n-samples", "100000","--n-epochs", "20",  "--force-clean", "False"]
    },
    {
        "script": "ae_rnn_pretrain.py", "args": ["--n-samples", "100000","--n-epochs", "20",  "--force-clean", "False"]
    },
]

EXPERIMENTS = [
    {
        "script": "Chess_1_PPO.py", "args": []
    },
    {
        "script": "Chess_2_MaskablePPO.py", "args": []
    },
    {
        "script": "Chess_3_MaskableRecurrentPPO.py", "args": []
    },
    {
        "script": "Chess_4_FF_AutoEncoder_MaskablePPO.py", "args": []
    },  
    {
        "script": "Chess_5_FF_Autoencoder_MaskableRecurrentPPO.py", "args": []
    },      
    {
        "script": "Chess_6_LSTM_Autoencoder_MaskablePPO.py", "args": []
    },
    {
        "script": "Chess_7_LSTM_Autoencoder_MaskableRecurrentPPO.py", "args": []
    },      
    {
        "script": "Chess_8_Transformer.py", "args": []
    },
]

# EXPERIMENTS = [
#     {
#         "script": "Chess_1_PPO.py", "args": ["--agent-name", "1_PPO", "--total-timesteps", "100", "--batch-size", "64", "--n-steps", "10" ]
#     },
#     {
#         "script": "Chess_2_MaskablePPO.py", "args": ["--agent-name", "2_MaskablePPO (baseline)", "--total-timesteps", "100", "--batch-size", "64", "--n-steps", "10" ]
#     },
# ]

def run_experiments(num_repeats, parallel):
    mode = "Parallel" if parallel else "Sequential"
    start = time.time()

    important2(f"Running experiments, num_repeats: {num_repeats}, mode: {mode}")

    if(parallel):
        for repeat in range(args.num_repeats):
            important(f"Iteration: {repeat+1}")
            
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(run_exp, exp, unknown) for exp in EXPERIMENTS]
                # wait and re-raise if any fails
                for fut in as_completed(futures):
                    fut.result()  # will raise CalledProcessError if the subprocess failed
    else:
        for repeat in range(num_repeats):
            important(f"Iteration: {repeat+1}")

            for exp in EXPERIMENTS:
                run_exp(exp, unknown)

    total_time = time.time() - start
    success(f"Experiments have completed in {total_time/60:.2f} minutes ({total_time:.1f}s)")

def run_exp(exp, unknown):
    if len(exp["args"]) == 0 :        
        exp["args"] = unknown
   
    cmd = build_cmd(exp["script"], exp["args"])
    important2(f"Running script: {exp['script']} with args {exp['args']}")

    t0 = time.time()
    subprocess.run(cmd, check=True)
    t1 = time.time()

    success(f"[{exp['script']}] Finished in {t1 - t0:.1f}s")

if __name__ == "__main__":    
    important("Starting experiments...")

    os.makedirs("boards", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("chess_logs", exist_ok=True)
    
    overall_start = time.time()
    parser = argparse.ArgumentParser()    
    parser.add_argument("--parallel",    type=str, default="False",help="Run experiments in parallel")
    parser.add_argument("--max-workers", type=int, default=3,help="How many processes to spawn in parallel mode")
    parser.add_argument("--num-repeats", type=int, default=1, help="Number of times to re-run each experiment")

    args, unknown = parser.parse_known_args()

    info("Parsed arguments:")
    for name, val in vars(args).items():
        info(f"  {name}: {val}")

    info(f"Arguments to be forwarded: {unknown}")

    important("Running Preprocessing Tasks...")
        
    for exp in PRE_PROCESSING_TASKS:
        run_exp(exp, unknown)       
    
    run_experiments(args.num_repeats, args.parallel == "True")    

    total_time = time.time() - overall_start
    success(f"All done in {total_time/60:.2f} minutes ({total_time:.1f}s)")