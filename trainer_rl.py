# NOTICE: This file has been modified from the original RIDER codebase
# (https://github.com/COLA-Laboratory/RIDER) as part of research
# conducted at Trinity Western University (2026), under the Apache
# License, Version 2.0.
#
# Modifications: RL fine-tuning loop extended with a secondary-structure
# reward term (lambda_ss, ss_bonus_scale) decoded from RhoFold's
# ss_head output, including a measurement-only probe period and
# adaptive bonus scaling (compute_adaptive_bonus_scale) based on
# observed per-epoch base-pair recovery, and automatic intra-chain
# pair filtering for multi-chain benchmark targets (MULTI_CHAIN_TARGETS).
# RhoFold remains the sole folding oracle on this branch.


"""Reinforcement learning fine-tuning loop for the diffusion model."""

import argparse
import math
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List

import dotenv
import ml_collections
import numpy as np
import torch
import torch.nn.utils as utils
from torch.optim import Adam
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm

dotenv.load_dotenv(".env")

from src.constants import DATA_PATH, PROJECT_PATH
from src.data.dataset import RNADesignDataset
from src.evaluator_rl import evaluate
from src.model import GVPDiff
from src.noise_schedule import NoiseScheduleVP
from src.diffusion import ddim_sample_with_logprob
from tools.rhofold.config import rhofold_config
from tools.rhofold.rf import RhoFold
from tools.rhofold.secondary_structure_reward import compute_r_ss, dotbracket_to_pairs, filter_intrachain_pairs



warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__)


def _checkpoint_dir() -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    run_id = getattr(wandb.run, "id", "local")
    out_dir = os.path.join(project_root, "outputs", "checkpoints", run_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    model_state = model.state_dict()
    loadable_state = {}
    skipped_shape = []
    unexpected_keys = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        loadable_state[key] = value

    missing_keys = [key for key in model_state.keys() if key not in loadable_state]
    model.load_state_dict(loadable_state, strict=False)

    logger.info(
        "Checkpoint loaded from %s: matched=%d, skipped_shape=%d, unexpected=%d, missing=%d",
        checkpoint_path,
        len(loadable_state),
        len(skipped_shape),
        len(unexpected_keys),
        len(missing_keys),
    )
    if skipped_shape:
        logger.warning("Skipped shape-mismatched keys (first 10): %s", skipped_shape[:10])
    if unexpected_keys:
        logger.warning("Unexpected checkpoint keys (first 10): %s", unexpected_keys[:10])


def compute_adaptive_bonus_scale(ema_nonzero_frac: float, base_scale: float,
                                  target_frac: float = 0.15) -> float:
    """
    Scales base_scale down linearly as ema_nonzero_frac approaches
    target_frac, clamped to [0, base_scale].
 
    ema_nonzero_frac=0.0   -> returns base_scale (full strength, dead/near-dead regime)
    ema_nonzero_frac>=target_frac -> returns 0.0 (bonus tapered off, working regime)
 
    target_frac=0.15 is a starting guess -- 2GCS's observed nonzero_frac
    climbed well above this once the bonus unlocked it; 2GDI's baseline
    (89.58%) is far above it. Adjust based on where the "already working"
    boundary actually sits once more targets are characterized.
    """
    if target_frac <= 0:
        return base_scale
    factor = 1.0 - (ema_nonzero_frac / target_frac)
    factor = max(0.0, min(1.0, factor))
    return base_scale * factor

def _update_baseline(previous: float, mean_reward: float, epoch: int, beta: float) -> float:
    """EMA-style baseline update used to stabilize policy gradient."""
    if epoch == 0 or mean_reward > previous:
        return mean_reward
    return beta * previous + (1 - beta) * mean_reward


@dataclass
class TrajectorySample:
    """Container for RL sample trajectories."""

    log_probs_traj: List[torch.Tensor]
    latents_traj: List[torch.Tensor]
    reward: float
    raw_data: dict
    advantage: float = 0.0

    def set_advantage(self, baseline: float) -> None:
        self.advantage = self.reward - baseline


def get_data_splits(split_type: str = "structsim_v2"):
    """Return train/val/test raw data lists for the requested split."""

    data_list = list(torch.load(os.path.join(DATA_PATH, "processed.pt"), weights_only=False).values())

    def index_list_by_indices(lst, indices):
        return [lst[index] for index in indices]

    train_idx_list, val_idx_list, test_idx_list = torch.load(
        os.path.join(DATA_PATH, f"{split_type}_split.pt"),
        weights_only=False,
    )
    train_list = index_list_by_indices(data_list, train_idx_list)
    val_list = index_list_by_indices(data_list, val_idx_list)
    test_list = index_list_by_indices(data_list, test_idx_list)
    return train_list, val_list, test_list


def get_dataset(config, data_list, split="train"):
    return RNADesignDataset(
        data_list=data_list,
        split=split,
        radius=config.radius,
        top_k=config.top_k,
        num_rbf=config.num_rbf,
        num_posenc=config.num_posenc,
        max_num_conformers=config.max_num_conformers,
        noise_scale=config.noise_scale
    )


def get_model(config):
    return GVPDiff(config)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def reward_fn(rhofold, dataset, raw_data, pred_seq, device, parallel_id=None,
              target_pairs=None, lambda_ss=0.0, ss_bonus_scale=0.0,
              ss_bonus_pair_cap=7):
    """Compute downstream reward using structural evaluation metrics.
 
    target_pairs: precomputed set of (i, j) target base pairs.
    lambda_ss: weight for the R_SS (F1) term. 0.0 disables it.
    ss_bonus_scale: weight for a raw correct-pair-COUNT bonus, applied
                    whenever at least one predicted pair matches a target
                    pair (tp > 0). 0.0 disables it. Unlike lambda_ss *
                    R_SS, this fires on partial credit even when overall
                    F1 rounds to a value PPO can't yet distinguish from
                    zero-overlap cases.
    """
    use_ss = (lambda_ss > 0 or ss_bonus_scale > 0) and target_pairs is not None
 
    results = evaluate(
        rhofold,
        dataset,
        raw_data,
        pred_seq,
        1,
        device=device,
        parallel_id=parallel_id,
        return_ss=use_ss,
    )
    score_gddt = results['sc_score_gddt'][0]
    score_tm = results['sc_score_tm'][0]
    score_rmsd = results['sc_score_rmsd'][0]
 
    reward = -(score_rmsd * 0.5) ** 2 + (score_gddt * 5) ** 2
 
    if score_gddt > 0.45:
        reward += (score_gddt - 0.45) * 100
    elif score_rmsd < 3:
        reward += (3 - score_rmsd) * 20
 
    score_ss = None
    n_correct_pairs = None
    if use_ss:
        ss_logits = results["ss_logits_list"][0]
        r_ss_result = compute_r_ss(ss_logits, target_pairs=target_pairs)
        score_ss = r_ss_result["f1"]
        n_correct_pairs = r_ss_result["tp"]
 
        if lambda_ss > 0:
            reward = reward + lambda_ss * score_ss
 
        # partial-credit bonus: fires on ANY correct pair, independent
        # of overall F1 -- this is what's supposed to break the
        # zero-variance trap that scaling lambda_ss alone couldn't fix
        if ss_bonus_scale > 0 and n_correct_pairs > 0:
            capped_pairs = min(n_correct_pairs, ss_bonus_pair_cap)
            reward += capped_pairs * ss_bonus_scale
 
    return reward, score_gddt, score_tm, score_rmsd, score_ss, n_correct_pairs

def sample_once(
    raw_data, dataset, model, rhofold, noise_scheduler, config, autocast,
    device, parallel_id=None, deterministic=False, temperature=None,
    target_pairs=None, lambda_ss=0.0, ss_bonus_scale=0.0, ss_bonus_pair_cap=7,
    ):
    """Generate a single trajectory sample and return it with reward statistics."""
 
    data = dataset.featurizer(raw_data).to(device)
    sample_temperature = config.temperature if temperature is None else temperature
    with autocast():
        x0_pred, log_probs_traj, latents_traj = ddim_sample_with_logprob(
            model,
            noise_scheduler,
            data,
            n_steps=config.n_steps,
            device=device,
            temperature=sample_temperature,
            deterministic=deterministic,
        )
    pred_seq = torch.argmax(x0_pred, dim=-1)
 
    reward, score_gddt, score_tm, score_rmsd, score_ss, n_correct_pairs = reward_fn(
        rhofold, dataset, raw_data, pred_seq, device=device,
        parallel_id=parallel_id, target_pairs=target_pairs,
        lambda_ss=lambda_ss, ss_bonus_scale=ss_bonus_scale,
        ss_bonus_pair_cap=ss_bonus_pair_cap,
    )
    
    if n_correct_pairs is not None and n_correct_pairs > 0:
        print(f"[NONZERO] parallel_id={parallel_id}, n_correct_pairs={n_correct_pairs}, "
              f"score_ss={score_ss:.3f}")

    trajectory = TrajectorySample(
        log_probs_traj=log_probs_traj,
        latents_traj=latents_traj,
        reward=reward,
        raw_data=raw_data,
    )
    return trajectory, score_gddt, score_tm, score_rmsd, score_ss, n_correct_pairs


def collect_parallel_samples(raw_data, dataset, model, rhofold, noise_scheduler,
                              config, autocast, device, target_pairs=None,
                              lambda_ss=0.0, ss_bonus_scale=0.0, ss_bonus_pair_cap=7):
    """Launch additional sampling tasks in parallel to diversify exploration.
 
    Returns (samples, n_correct_pairs_list).
    """
    n_parallel_rollouts = max(config.rollouts_per_round - 1, 0)
    if n_parallel_rollouts == 0:
        return [], []
 
    samples = []
    n_correct_pairs_list = []
    with ThreadPoolExecutor(max_workers=n_parallel_rollouts) as executor:
        futures = [
            executor.submit(
                sample_once,
                raw_data, dataset, model, rhofold, noise_scheduler, config,
                autocast, device,
                parallel_id=parallel_id,
                deterministic=config.deterministic,
                target_pairs=target_pairs,
                lambda_ss=lambda_ss,
                ss_bonus_scale=ss_bonus_scale,
                ss_bonus_pair_cap=ss_bonus_pair_cap,
            )
            for parallel_id in range(n_parallel_rollouts)
        ]
        for future in as_completed(futures):
            sample, _, _, _, _, n_correct_pairs = future.result()
            samples.append(sample)
            if n_correct_pairs is not None:
                n_correct_pairs_list.append(n_correct_pairs)
    return samples, n_correct_pairs_list
 


def train_diffusion_rl(config, model, dataset, device, accelerator, optimizer):
    """Fine-tune the diffusion model with trajectory-level RL updates."""

    autocast = accelerator.autocast
    clip_range = config.clip_range

    device_rho = torch.device("cuda")
    rhofold = RhoFold(rhofold_config, device_rho)
    rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
    print(f"Loading RhoFold checkpoint: {rhofold_path}")
    rhofold.load_state_dict(torch.load(rhofold_path, map_location=torch.device("cpu"))["model"])
    rhofold = rhofold.to(device_rho)
    rhofold.eval()

    noise_scheduler = NoiseScheduleVP(
        config.sde_schedule,
        continuous_beta_0=config.continuous_beta_0,
        continuous_beta_1=config.continuous_beta_1,
        dtype=torch.float32,
    )
    noise_scheduler.eps = config.eps

    time_grid = torch.linspace(noise_scheduler.T, noise_scheduler.eps, config.n_steps, device=device)

    avg_reward_best = float("-inf")
    baseline = 0.0
    beta_baseline = 0.8
    ckpt_dir = _checkpoint_dir()
    if len(dataset.data_list) == 0:
        raise ValueError("RL dataset is empty after preprocessing.")
    sample_index = int(config.sample_index) % len(dataset.data_list)
    raw_data_fixed = dataset.data_list[sample_index]
    lambda_ss = float(getattr(config, 'lambda_ss', 0.0))
    ss_bonus_scale = float(getattr(config, 'ss_bonus_scale', 0.0))
    if ss_bonus_scale > 0:
        print(f"SS bonus enabled: ss_bonus_scale={ss_bonus_scale}")
    MULTI_CHAIN_TARGETS = {"1XPE", "1CSL", "1LNT", "354D", "1Q9A", "1X9C"}

    target_pairs = None
    if lambda_ss > 0 or ss_bonus_scale > 0:
        target_name = getattr(config, 'target_name', None)
        raw_pairs = dotbracket_to_pairs(raw_data_fixed['sec_struct_list'][0])

        if target_name in MULTI_CHAIN_TARGETS:
            target_pairs = filter_intrachain_pairs(
                raw_pairs, len(raw_data_fixed['sequence'])
            )
        else:
            target_pairs = raw_pairs

        print(f"R_SS enabled: lambda_ss={lambda_ss}, ss_bonus_scale={ss_bonus_scale}, "
              f"{len(target_pairs)} target base pairs loaded"
              + (" (intra-chain filtered)" if target_name in MULTI_CHAIN_TARGETS else ""))

    base_ss_bonus_scale = ss_bonus_scale
    ema_nonzero_frac = 0.0
    ema_beta_ss = 0.8
    target_nonzero_frac = float(getattr(config, 'ss_target_nonzero_frac', 0.15))
    ss_probe_epochs = int(getattr(config, 'ss_probe_epochs', 5))
    ss_bonus_pair_cap = int(getattr(config, 'ss_bonus_pair_cap', 7))


    for epoch in range(config.epochs):
        model.eval()
        print(f"Sample epoch: {epoch}")
 
        if epoch < ss_probe_epochs:
            effective_ss_bonus_scale = 0.0
            print(f"  [ADAPTIVE] Probe epoch {epoch}/{ss_probe_epochs} -- "
                  f"measuring only, bonus forced to 0.0")
        else:
            effective_ss_bonus_scale = compute_adaptive_bonus_scale(
                ema_nonzero_frac, base_ss_bonus_scale, target_nonzero_frac
            )
            print(f"  [ADAPTIVE] ema_nonzero_frac={ema_nonzero_frac:.3f}, "
                  f"effective_ss_bonus_scale={effective_ss_bonus_scale:.2f}")
 
        sampled_data: List[TrajectorySample] = []
        epoch_n_correct_pairs: List[int] = []
        target_samples = int(config.target_samples_per_epoch)
        with tqdm(total=target_samples, desc=f"Epoch {epoch} Sampling", leave=False) as pbar:
            while len(sampled_data) < target_samples:
                # Anchor each round with one deterministic rollout, then expand with stochastic ones.
                deterministic_sample, score_gddt, score_tm, score_rmsd, score_ss, n_correct_pairs = sample_once(
                    raw_data_fixed,
                    dataset,
                    model,
                    rhofold,
                    noise_scheduler,
                    config,
                    autocast,
                    device,
                    deterministic=True,
                    temperature=0.0,
                    target_pairs=target_pairs,
                    lambda_ss=lambda_ss,
                    ss_bonus_scale=effective_ss_bonus_scale,
                    ss_bonus_pair_cap=ss_bonus_pair_cap
                )
                sampled_data.append(deterministic_sample)
                if n_correct_pairs is not None:
                    epoch_n_correct_pairs.append(n_correct_pairs)
 
                log_dict = {
                    "Test/reward": deterministic_sample.reward,
                    "Test/score_gddt": score_gddt,
                    "Test/score_tm": score_tm,
                    "Test/score_rmsd": score_rmsd,
                }
                if score_ss is not None:
                    log_dict["Test/score_ss"] = score_ss
                    log_dict["Test/n_correct_pairs"] = n_correct_pairs
                    log_dict["Test/effective_ss_bonus_scale"] = effective_ss_bonus_scale
                wandb.log(log_dict)
 
                extra_samples, extra_n_correct_pairs = collect_parallel_samples(
                    raw_data_fixed, dataset, model, rhofold, noise_scheduler, config,
                    autocast, device, target_pairs=target_pairs, lambda_ss=lambda_ss,
                    ss_bonus_scale=effective_ss_bonus_scale, ss_bonus_pair_cap=ss_bonus_pair_cap
                )
                sampled_data.extend(extra_samples)
                epoch_n_correct_pairs.extend(extra_n_correct_pairs)
 
                current_count = min(len(sampled_data), target_samples)
                pbar.n = current_count
                if sampled_data:
                    recent_rewards = [s.reward for s in sampled_data[:current_count]]
                    pbar.set_postfix(avg_reward=f"{np.mean(recent_rewards):.3f}")
                pbar.refresh()
 
        if epoch_n_correct_pairs:
            epoch_nonzero_frac = sum(1 for n in epoch_n_correct_pairs if n > 0) / len(epoch_n_correct_pairs)
            if epoch == 0:
                ema_nonzero_frac = epoch_nonzero_frac
            else:
                ema_nonzero_frac = ema_beta_ss * ema_nonzero_frac + (1 - ema_beta_ss) * epoch_nonzero_frac
            wandb.log({"Test/epoch_nonzero_frac": epoch_nonzero_frac,
                       "Test/ema_nonzero_frac": ema_nonzero_frac, "epoch": epoch})

        sampled_data = sampled_data[:target_samples]
        rewards = [sample.reward for sample in sampled_data]
        mean_reward = np.mean(rewards)
        baseline = _update_baseline(baseline, mean_reward, epoch, beta_baseline)

        for sample in sampled_data:
            sample.set_advantage(baseline)

        random.shuffle(sampled_data)

        rl_losses: List[float] = []
        for _ in range(config.rl_update_epochs):
            for sample in sampled_data:
                traj_losses = []
                old_log_probs_traj = sample.log_probs_traj
                latents_traj = sample.latents_traj[:-1]
                next_latents_traj = sample.latents_traj[1:]

                total_steps = len(old_log_probs_traj)
                num_chunks = config.num_chunks
                chunk_size = (total_steps + num_chunks - 1) // num_chunks
                data = dataset.featurizer(sample.raw_data).to(device)

                for chunk_id in range(num_chunks):
                    start = chunk_id * chunk_size
                    end = min(start + chunk_size, total_steps)
                    if start >= end:
                        break

                    with accelerator.accumulate(model):
                        with autocast():
                            old_lp_chunk = torch.stack(old_log_probs_traj[start:end], dim=0).to(device)
                            z_t_chunk = torch.stack(latents_traj[start:end], dim=0).to(device)
                            z_tp1_chunk = torch.stack(next_latents_traj[start:end], dim=0).to(device)

                            t_chunk = time_grid[start:end]
                            t_next_chunk = time_grid[start + 1 : end + 1]
                            alpha_t, sigma_t = noise_scheduler.marginal_prob(t_chunk.unsqueeze(-1))
                            alpha_tp1, sigma_tp1 = noise_scheduler.marginal_prob(t_next_chunk.unsqueeze(-1))
                            noise_levels = torch.log(alpha_t**2 / sigma_t**2).to(device)

                            data.z_t = z_t_chunk
                            pred_noise = model.sample(
                                data,
                                n_samples=(end - start),
                                time=None,
                                noise_level=noise_levels,
                            )

                            x0_hat = (z_t_chunk - sigma_t.view(-1, 1, 1) * pred_noise) / alpha_t.view(-1, 1, 1)
                            mu = alpha_tp1.view(-1, 1, 1) * x0_hat
                            std = sigma_tp1.view(-1, 1, 1)

                            logp = -((z_tp1_chunk - mu) ** 2) / (2 * std**2) - std.log() - 0.5 * math.log(2 * math.pi)
                            new_lp = logp.view(end - start, -1).mean(dim=1)

                            ratio = torch.exp(new_lp - old_lp_chunk)

                            # clipped objective on trajectory chunks.
                            unclipped = -sample.advantage * ratio
                            clipped = -sample.advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                            chunk_loss = torch.max(unclipped, clipped).mean()
                            if not chunk_loss.requires_grad:
                                raise RuntimeError(
                                    "chunk_loss has no grad_fn. "
                                    "Check that model forward path is not under torch.no_grad()."
                                )

                            accelerator.backward(chunk_loss)
                            if accelerator.sync_gradients:
                                utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                                optimizer.step()
                                optimizer.zero_grad()

                            traj_losses.append(chunk_loss.item())

                if traj_losses:
                    rl_losses.append(float(np.mean(traj_losses)))

        avg_loss = float(np.mean(rl_losses)) if rl_losses else 0.0
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        logger.info(f"Epoch {epoch}: RL Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}")
        wandb.log(
            {
                "train/rl_loss": avg_loss,
                "train/avg_reward": avg_reward,
                "train/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
        )

        if config.save and accelerator.is_main_process and avg_reward >= avg_reward_best:
            avg_reward_best = avg_reward
            checkpoint_path = os.path.join(ckpt_dir, "current_checkpoint_rl.h5")
            torch.save(model.state_dict(), checkpoint_path)
            wandb.run.summary["best_checkpoint_rl"] = checkpoint_path

    logger.info("RL fine-tuning finished.")


def main(config, device):
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.accumulate_steps * config.num_chunks
    )

    set_seed(config.seed)

    model = get_model(config).to(device)
    total_param = sum(np.prod(list(p.size())) for p in model.parameters())
    wandb.run.summary["total_param"] = total_param

    if config.model_path:
        _load_model_checkpoint(model, config.model_path, device)

    params_to_update = [param for param in model.parameters() if param.requires_grad]

    optimizer = Adam(params_to_update, lr=config.lr)

    _, _, test_list = get_data_splits(split_type=config.split)
    dataset = get_dataset(config, test_list, split="train")

    model, optimizer = accelerator.prepare(model, optimizer)

    train_diffusion_rl(config, model, dataset, device, accelerator, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs/default_rl.yaml', type=str)
    parser.add_argument('--expt_name', dest='expt_name', default=None, type=str)
    parser.add_argument('--tags', nargs='+', dest='tags', default=[])
    parser.add_argument('--no_wandb', action="store_true")
    args, _ = parser.parse_known_args()

    wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        config=args.config,
        name=args.expt_name,
        tags=args.tags,
        mode="disabled" if args.no_wandb else "online",
    )
    config = wandb.config

    config_dict = dict(wandb.config)
    config = ml_collections.ConfigDict(config_dict)

    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")

    main(config, device)
