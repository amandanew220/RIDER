
import os
import subprocess
import yaml
 
from validate_true_chain_boundaries import validate_target, MULTI_CHAIN_TARGETS
 
BASE_CONFIG_PATH = "configs/default_rl.yaml"
GENERATED_DIR = "configs/generated"
REPO_DIR = "/scratch/anew/RIDER_ss_experiment"
ACCOUNT = "def-htsang"
GRES = "gpu:h100:1"
MEM = "16G"
TIME = "12:00:00"
 
LAMBDA_SS = 0.2
SS_BONUS_SCALE = 8.0
SS_PROBE_EPOCHS = 5
SS_TARGET_NONZERO_FRAC = 0.15
SEEDS = [0, 1]  # SAME seeds as the original midpoint sweep -- paired comparison
 
 
def determine_rerun_targets():
    """Re-derives which multi-chain targets actually need a rerun by
    calling the real validation logic, rather than trusting a hardcoded
    list that could go stale."""
    rerun_targets = {}
    for pdb_id, sample_index in MULTI_CHAIN_TARGETS.items():
        result = validate_target(pdb_id, sample_index)
        if result.get("needs_rerun"):
            rerun_targets[pdb_id] = sample_index
    return rerun_targets
 
 
def load_base_config():
    with open(BASE_CONFIG_PATH) as f:
        return yaml.safe_load(f)
 
 
def make_run_config(base_config, target, sample_index, seed):
    config = yaml.safe_load(yaml.safe_dump(base_config))  # deep copy
    for key, val in [
        ("sample_index", sample_index),
        ("lambda_ss", LAMBDA_SS),
        ("ss_bonus_scale", SS_BONUS_SCALE),
        ("ss_probe_epochs", SS_PROBE_EPOCHS),
        ("ss_target_nonzero_frac", SS_TARGET_NONZERO_FRAC),
        ("target_name", target),
        ("seed", seed),
    ]:
        if key not in config:
            config[key] = {}
        config[key]["value"] = val
    return config
 
 
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
 
    os.makedirs(GENERATED_DIR, exist_ok=True)
    base_config = load_base_config()
 
    print("Re-running validation to determine which targets actually need "
          "new jobs (not trusting a hardcoded list)...\n")
    rerun_targets = determine_rerun_targets()
 
    if not rerun_targets:
        print("No targets show a pair-set difference -- nothing to rerun.")
        return
 
    planned_runs = []
    for target, sample_index in rerun_targets.items():
        for seed in SEEDS:
            expt_name = f"ss_reward_{target}_true_chain_seed{seed}"
            config_path = os.path.join(GENERATED_DIR, f"{expt_name}.yaml")
 
            run_config = make_run_config(base_config, target, sample_index, seed)
            with open(config_path, "w") as f:
                yaml.safe_dump(run_config, f)
 
            planned_runs.append((expt_name, config_path))
 
    print(f"\n{len(planned_runs)} runs planned (true-chain fix, matched seeds "
          f"to original midpoint sweep):\n")
    for expt_name, config_path in planned_runs:
        print(f"  {expt_name}")
 
    if args.dry_run:
        print("\n[DRY RUN] No jobs submitted.")
        return
 
    print("\nSubmitting jobs...\n")
    for expt_name, config_path in planned_runs:
        log_file = f"{expt_name}.log"
        wrap_cmd = (
            f"cd {REPO_DIR} && source ~/RIDER_env/bin/activate && "
            f"python3 trainer_rl.py --config {config_path} --expt_name {expt_name}"
        )
        sbatch_cmd = [
            "sbatch", f"--account={ACCOUNT}", f"--gres={GRES}",
            f"--mem={MEM}", f"--time={TIME}", f"--output={log_file}",
            f"--wrap={wrap_cmd}",
        ]
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
        print(f"{expt_name}: {result.stdout.strip() or result.stderr.strip()}")
 
 
if __name__ == "__main__":
    main()
    