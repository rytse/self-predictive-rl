import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob

import yaml

plt.rcParams.update({"font.size": 24})
FIG_WIDTH = 16
FIG_HEIGHT = 20
DPI = 300

mujoco_auxs = ["l2", "rkl", "bisim", "bisim_critic", "zp_critic"]
minigrid_auxs = ["ZP", "bisim", "bisim_critic", "zp_critic"]

token_mujoco_env = "HalfCheetah-v2"
token_minigrid_env = "MiniGrid-LavaCrossingS9N3-v0"


def get_envs(envs_file: Path) -> List[str]:
    envs = []
    with open(envs_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                envs.append(stripped)
    return envs


def get_results(
    base_path: Path, env_type: Union[Literal["mujoco"], Literal["minigrid"]]
) -> Dict[str, Dict[str, Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]]]:
    results = dict()

    if env_type == "mujoco":
        cfg_file_name = "flags.yml"
        cfg_load_fn = yaml.safe_load
    elif env_type == "minigrid":
        cfg_file_name = "config.json"
        cfg_load_fn = json.load
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    cfg_paths = glob.glob(f"{str(base_path)}/**/{cfg_file_name}", recursive=True)

    for cfg_path in cfg_paths:
        res_path = cfg_path.replace(cfg_file_name, "progress.csv")
        with open(cfg_path, "r") as cfg_file:
            cfg = cfg_load_fn(cfg_file)
            aux = cfg["aux"]
            env = cfg_path.split("/")[-3]

            if env not in results:
                results[env] = dict()
            if aux not in results[env]:
                results[env][aux] = dict()

            try:
                results[env][aux][res_path] = (pd.read_csv(res_path), cfg)
            except Exception as e:
                print(f"Error reading {res_path}: {e}")

    return results


mujoco_envs = get_envs(Path("mujoco_code/envs.txt"))
assert len(mujoco_envs) == 4
minigrid_envs = get_envs(Path("minigrid_code/envs.txt"))

# Plot nominal mujoco results
nominal_mujoco_res = get_results(Path("mujoco_code/logs_nominal_mujoco"), "mujoco")
assert len(nominal_mujoco_res) == len(mujoco_envs)
plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT * 2))
for env_idx, env in enumerate(mujoco_envs):
    plt.subplot(2, 2, env_idx + 1)
    legend_labels = []
    for aux in mujoco_auxs:
        for exp_idx, (df, _cfg) in enumerate(nominal_mujoco_res[env][aux].values()):
            valid_idxs = ~df["return"].isna()
            env_steps = df["env_steps"][valid_idxs]
            rewards = df["return"][valid_idxs]

            plt.plot(env_steps, rewards, label=f"{aux} {exp_idx}")

            if len(nominal_mujoco_res[env][aux]) == 1:
                legend_labels.append(aux)
            else:
                legend_labels.append(f"{aux} {exp_idx}")
    plt.xlabel("Env Steps")
    plt.ylabel("Return")
    plt.title(env)
    plt.legend(legend_labels)
plt.suptitle("Nominal Mujoco Results", fontsize=32)
plt.savefig("./plots/nominal_mujoco.png", dpi=DPI)

# Nominal minigrid results
nominal_minigrid_res = get_results(
    Path("minigrid_code/logs_nominal_minigrid"), "minigrid"
)
assert len(nominal_minigrid_res) == len(minigrid_envs)
plt.figure(figsize=(FIG_WIDTH * 3, FIG_HEIGHT * 3))
for env_idx, env in enumerate(minigrid_envs):
    plt.subplot(3, 3, env_idx + 1)
    legend_labels = []
    for aux in minigrid_auxs:
        for exp_idx, (df, _cfg) in enumerate(nominal_minigrid_res[env][aux].values()):
            valid_idxs = ~df["return"].isna()
            env_steps = df["env_steps"][valid_idxs]
            rewards = df["return"][valid_idxs]

            plt.plot(env_steps, rewards, label=f"{aux} {exp_idx}")

            if len(nominal_minigrid_res[env][aux]) == 1:
                legend_labels.append(aux)
            else:
                legend_labels.append(f"{aux} {exp_idx}")
    plt.xlabel("Env Steps")
    plt.ylabel("Return")
    plt.title(env)
    plt.legend(legend_labels)
plt.suptitle("Nominal Minigrid Results", fontsize=32)
plt.savefig("./plots/nominal_minigrid.png", dpi=DPI)

# bisim_gamma ablation
bisim_gammas = [0.25, 0.50, 0.75, 1.0]  # baseline is 0.5
baseline_bisim_gamma = 0.50

# bisim_gamma mujoco ablation
ablate_bisim_gamma_mujoco_res = get_results(
    Path("mujoco_code/logs_ablate/bisim_gamma"), "mujoco"
)
plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT))
for aux_idx, aux in enumerate(["bisim", "bisim_critic"]):
    plt.subplot(1, 2, aux_idx + 1)
    legend_labels = []

    for bisim_gamma in bisim_gammas:
        if bisim_gamma == baseline_bisim_gamma:
            dfs = nominal_mujoco_res[token_mujoco_env][aux]
            assert len(dfs) == 1
            df = list(dfs.values())[0][0]
        else:
            cand_dfs = ablate_bisim_gamma_mujoco_res[token_mujoco_env][aux]
            dfs = []
            for cand_df, cand_cfg in cand_dfs.values():
                if cand_cfg["bisim_gamma"] == bisim_gamma:
                    dfs.append(cand_df)
            assert len(dfs) == 1
            df = dfs[0]

        valid_idxs = ~df["return"].isna()
        env_steps = df["env_steps"][valid_idxs]
        rewards = df["return"][valid_idxs]

        plt.plot(env_steps, rewards, label=f"{aux} {bisim_gamma}")
        legend_labels.append(f"bisim_gamma={bisim_gamma}")

    plt.xlabel("Env Steps")
    plt.ylabel("Return")
    plt.title(f"{token_mujoco_env} {aux}")
    plt.legend(legend_labels)
plt.suptitle("Bisim Gamma Ablation for Mujoco", fontsize=32)
plt.savefig("./plots/bisim_gamma_mujoco_ablation.png", dpi=DPI)

# bisim_gamma minigrid ablation
ablate_bisim_gamma_minigrid_res = get_results(
    Path("minigrid_code/logs_ablate/bisim_gamma"), "minigrid"
)
plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT))
for aux_idx, aux in enumerate(["bisim", "bisim_critic"]):
    plt.subplot(1, 2, aux_idx + 1)
    legend_labels = []

    for bisim_gamma in bisim_gammas:
        if bisim_gamma == baseline_bisim_gamma:
            dfs = nominal_minigrid_res[token_minigrid_env][aux]
            assert len(dfs) == 1
            df = list(dfs.values())[0][0]
        else:
            cand_dfs = ablate_bisim_gamma_minigrid_res[token_minigrid_env][aux]
            dfs = []
            for cand_df, cand_cfg in cand_dfs.values():
                if cand_cfg["bisim_gamma"] == bisim_gamma:
                    dfs.append(cand_df)
            assert len(dfs) == 1
            df = dfs[0]

        valid_idxs = ~df["return"].isna()
        env_steps = df["env_steps"][valid_idxs]
        rewards = df["return"][valid_idxs]

        plt.plot(env_steps, rewards, label=f"{aux} {bisim_gamma}")
        legend_labels.append(f"bisim_gamma={bisim_gamma}")

    plt.xlabel("Env Steps")
    plt.ylabel("Return")
    plt.title(f"{token_minigrid_env} {aux}")
    plt.legend(legend_labels)
plt.suptitle("Bisim Gamma Ablation for Minigrid", fontsize=32)
plt.savefig("./plots/bisim_gamma_minigrid_ablation.png", dpi=DPI)


# wass_critic_train_steps ablation
wass_critic_train_steps_all = [1, 5, 10]
baseline_wass_critic_train_steps = 1

# wass_critic_train_steps mujoco ablation
ablate_wass_critic_train_steps_mujoco_res = get_results(
    Path("mujoco_code/logs_ablate/wass_critic_train_steps"), "mujoco"
)
plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT))
for aux_idx, aux in enumerate(["bisim_critic", "zp_critic"]):
    plt.subplot(1, 2, aux_idx + 1)
    legend_labels = []

    for wass_critic_train_steps in wass_critic_train_steps_all:
        if wass_critic_train_steps == baseline_wass_critic_train_steps:
            dfs = nominal_mujoco_res[token_mujoco_env][aux]
            assert len(dfs) == 1
            df = list(dfs.values())[0][0]
        else:
            cand_dfs = ablate_wass_critic_train_steps_mujoco_res[token_mujoco_env][aux]
            dfs = []
            for cand_df, cand_cfg in cand_dfs.values():
                if cand_cfg["wass_critic_train_steps"] == wass_critic_train_steps:
                    dfs.append(cand_df)
            assert len(dfs) == 1
            df = dfs[0]

        valid_idxs = ~df["return"].isna()
        env_steps = df["env_steps"][valid_idxs]
        rewards = df["return"][valid_idxs]

        plt.plot(env_steps, rewards, label=f"{aux} {wass_critic_train_steps}")
        legend_labels.append(f"wass_critic_train_steps={wass_critic_train_steps}")

    plt.xlabel("Env Steps")
    plt.ylabel("Return")
    plt.title(f"{token_mujoco_env} {aux}")
    plt.legend(legend_labels)
plt.suptitle("Wass Critic Train Steps Ablation for Mujoco", fontsize=32)
plt.savefig("./plots/wass_critic_train_steps_mujoco_ablation.png", dpi=DPI)

# wass_critic_train_steps minigrid ablation
ablate_wass_critic_train_steps_minigrid_res = get_results(
    Path("minigrid_code/logs_ablate/wass_critic_train_steps"), "minigrid"
)
plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT))
for aux_idx, aux in enumerate(["bisim_critic", "zp_critic"]):
    plt.subplot(1, 2, aux_idx + 1)
    legend_labels = []

    for wass_critic_train_steps in wass_critic_train_steps_all:
        if wass_critic_train_steps == baseline_wass_critic_train_steps:
            dfs = nominal_minigrid_res[token_minigrid_env][aux]
            assert len(dfs) == 1
            df = list(dfs.values())[0][0]
        else:
            cand_dfs = ablate_wass_critic_train_steps_minigrid_res[token_minigrid_env][
                aux
            ]
            dfs = []
            for cand_df, cand_cfg in cand_dfs.values():
                if cand_cfg["wass_critic_train_steps"] == wass_critic_train_steps:
                    dfs.append(cand_df)
            assert len(dfs) == 1
            df = dfs[0]

        valid_idxs = ~df["return"].isna()
        env_steps = df["env_steps"][valid_idxs]
        rewards = df["return"][valid_idxs]

        plt.plot(env_steps, rewards, label=f"{aux} {wass_critic_train_steps}")
        legend_labels.append(f"wass_critic_train_steps={wass_critic_train_steps}")

    plt.xlabel("Env Steps")
    plt.ylabel("Return")
    plt.title(f"{token_minigrid_env} {aux}")
    plt.legend(legend_labels)
plt.suptitle("Wass Critic Train Steps Ablation for Minigrid", fontsize=32)
plt.savefig("./plots/wass_critic_train_steps_minigrid_ablation.png", dpi=DPI)

# distractors mujoco
plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT * 2))
token_mujoco_distractor_env = token_mujoco_env + "-d32"
for aux_idx, aux in enumerate(mujoco_auxs):
    if aux == "l2":
        continue

    plt.subplot(2, 2, aux_idx)  # skipped l2, the first one
    legend_labels

    for distractor in ["none", "gaussian", "interleaved_gaussian_mixture"]:
        if distractor == "none":
            dfs = nominal_mujoco_res[token_mujoco_env][aux]
            assert len(dfs) == 1
            df = list(dfs.values())[0][0]
        else:
            cand_dfs = get_results(
                Path(f"mujoco_code/logs_distractors/{distractor}"), "mujoco"
            )[token_mujoco_distractor_env][aux]
            assert len(cand_dfs) == 1
            df = list(cand_dfs.values())[0][0]

        valid_idxs = ~df["return"].isna()
        env_steps = df["env_steps"][valid_idxs]
        rewards = df["return"][valid_idxs]

        plt.plot(env_steps, rewards, label=distractor)
        legend_labels.append(distractor)

    plt.xlabel("Env Steps")
    plt.ylabel("Return")
    plt.title(f"{token_mujoco_env} {aux}")
    plt.legend()

plt.suptitle("Distractors Ablation for Mujoco", fontsize=32)
plt.savefig("./plots/distractors_mujoco.png", dpi=DPI)

# distractors minigrid
# TODO minigrid
