#!/usr/bin/env python3

import argparse
import sys
import optuna
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor


# ==============================================================================
# YOUR TARGET PROGRAM GOES HERE
# ==============================================================================
def my_target_program(params):
    command = [
        "bin/bptt_learning",
        "networks/risp_f_emptynet.json",
        "/home/jackson/framework/cpp-apps/applications/classify/datasets/regular/iris_data.csv",
        "/home/jackson/framework/cpp-apps/applications/classify/datasets/regular/iris_label.csv",
    ]
    command.append(f"{params["learning_rate"]}")
    command.append(f"{params["decay_rate"]}")
    command.append(f"{params["tau"]}")
    command.append(f"{params["rho"]}")
    avg_score = 0.0

    trials = 10
    for i in range(trials):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            output = result.stdout.strip()
            score = float(output.split("\n")[-1])

            avg_score += (
                score  # Returning negative error so that Maximize = Minimum Error
            )

        except subprocess.CalledProcessError as e:
            print(f"Error: External program failed with exit code {e.returncode}")
            print(f"Stderr: {e.stderr}")
            # Return a very bad score so Optuna learns to avoid this parameter range
            return -float("inf")
        except ValueError:
            print(f"Error: Could not parse the output '{output}' as a float.")
            return -float("inf")

    return avg_score / float(trials)


# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamic Bayesian Optimization Wrapper"
    )
    parser.add_argument(
        "--direction",
        choices=["minimize", "maximize"],
        default="maximize",
        help="The direction of the optimization (default: maximize)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of optimization trials (default: 20)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel processes"
    )
    parser.add_argument(
        "params",
        nargs="*",
        help="Parameters in triplets: name min max [name min max ...]",
    )

    args = parser.parse_args()

    # Validate that parameters come in triplets
    if len(args.params) % 3 != 0:
        parser.error("Each parameter must have exactly 3 values: name, min, and max.")

    # Group the flat list into a list of tuples: [('name', min, max), ...]
    param_triplets = []
    for i in range(0, len(args.params), 3):
        name = args.params[i]
        try:
            vmin = float(args.params[i + 1])
            vmax = float(args.params[i + 2])
        except ValueError:
            parser.error(f"Min/Max for '{name}' must be numeric.")
        param_triplets.append((name, vmin, vmax))

    return args, param_triplets


def objective(trial, param_triplets):
    """
    Optuna objective function that maps trial suggestions to our target program.
    """
    # 1. Suggest values for each parameter defined via CLI
    suggested_params = {}
    for name, vmin, vmax in param_triplets:
        suggested_params[name] = trial.suggest_float(name, vmin, vmax)

    # 2. Call your actual program
    # Note: We pass the dictionary of parameters to your function
    score = my_target_program(suggested_params)

    # 3. Return the score to Optuna
    return score


def run_worker(study_name, storage, param_triplets, trials_to_run):
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(
        lambda trial: objective(trial, param_triplets), n_trials=trials_to_run
    )
    return f"Worker finished {trials_to_run} trials"


def main():
    args, param_triplets = parse_args()

    if not param_triplets:
        print("Error: No parameters provided. Use 'name min max' format.")
        sys.exit(1)

    db_file = "optuna_study.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    storage_url = f"sqlite:///{db_file}"
    study_name = "parallel_optimization"

    # Create a new study
    # TPE is the standard Bayesian algorithm used by Optuna
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=args.direction,
    )

    print(f"Starting optimization...")
    print(f"Parameters to optimize: {param_triplets}")
    print(f"Direction: {args.direction}")
    print(f"Workers: {args.workers}")
    print(f"Total Trials: {args.trials}\n")

    trials_per_worker = args.trials // args.workers
    remaining_trials = args.trials % args.workers

    tasks = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i in range(args.workers):
            n_trials = trials_per_worker + (remaining_trials if i == 0 else 0)
            if n_trials <= 0:
                continue

            tasks.append(
                executor.submit(
                    run_worker,
                    study_name,
                    storage_url,
                    param_triplets,
                    n_trials,
                )
            )

        for future in tasks:
            print(future.result())

    study = optuna.load_study(study_name=study_name, storage=storage_url)

    # Results
    print("\n" + "=" * 30)
    print("OPTIMIZATION COMPLETE")
    print("=" * 30)
    print(f"Best Score: {study.best_value}")
    print(f"Best Params: {study.best_params}")
    print(f"Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
