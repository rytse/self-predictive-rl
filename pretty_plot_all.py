import json
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import yaml

# Style settings remain the same
plt.style.use("seaborn")
sns.set_palette("husl")

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.5,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "figure.figsize": (12, 8),
        "figure.dpi": 300,
    }
)

FIG_WIDTH = 12
FIG_HEIGHT = 8
DPI = 300

mujoco_auxs = ["l2", "rkl", "bisim", "bisim_critic", "zp_critic"]
minigrid_auxs = ["ZP", "bisim", "bisim_critic", "zp_critic"]

token_mujoco_env = "HalfCheetah-v2"
token_minigrid_env = "MiniGrid-LavaCrossingS9N3-v0"


def style_plot(ax, title, xlabel="Env Steps", ylabel="Return"):
    """Helper function to consistently style plots"""
    ax.set_title(title, pad=20, fontweight="bold")
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_results(envs, results, auxs, plot_title, filename):
    """Generic function to plot results with consistent styling"""
    if math.sqrt(len(envs)).is_integer():
        nrows = ncols = int(math.sqrt(len(envs)))
    else:
        nrows = int(math.sqrt(len(envs)))
        ncols = int(math.ceil(len(envs) / nrows))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(FIG_WIDTH * ncols, FIG_HEIGHT * nrows)
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        legend_labels = []

        for aux in auxs:
            if aux not in results[env]:
                continue

            for exp_idx, (path, (df, cfg)) in enumerate(results[env][aux].items()):
                if not isinstance(df, pd.DataFrame):
                    continue

                valid_idxs = ~df["return"].isna()
                env_steps = df["env_steps"][valid_idxs]
                rewards = df["return"][valid_idxs]

                label = aux if len(results[env][aux]) == 1 else f"{aux} {exp_idx}"
                ax.plot(env_steps, rewards, label=label)
                legend_labels.append(label)

        style_plot(ax, env)

    for idx in range(len(envs), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(plot_title, fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_ablation(
    results,
    baseline_results,
    token_env,
    auxs,
    param_values,
    baseline_value,
    param_name,
    plot_title,
    filename,
):
    """Generic function for ablation plots with consistent styling"""
    fig, axes = plt.subplots(1, len(auxs), figsize=(FIG_WIDTH * len(auxs), FIG_HEIGHT))
    if len(auxs) == 1:
        axes = [axes]

    for aux_idx, aux in enumerate(auxs):
        ax = axes[aux_idx]

        for param_value in param_values:
            if param_value == baseline_value:
                dfs = baseline_results[token_env][aux]
                assert len(dfs) == 1
                df, _ = list(dfs.values())[0]
            else:
                cand_dfs = results[token_env][aux]
                dfs = []
                for _, (cand_df, cand_cfg) in cand_dfs.items():
                    if cand_cfg[param_name] == param_value:
                        dfs.append(cand_df)
                assert len(dfs) == 1
                df = dfs[0]

            valid_idxs = ~df["return"].isna()
            env_steps = df["env_steps"][valid_idxs]
            rewards = df["return"][valid_idxs]

            ax.plot(env_steps, rewards, label=f"{param_name}={param_value}")

        style_plot(ax, f"{token_env} {aux}")

    plt.suptitle(plot_title, fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close()


# Your existing get_envs function
def get_envs(envs_file: Path) -> List[str]:
    envs = []
    with open(envs_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                envs.append(stripped)
    return envs


# Modified get_results function with debugging
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

        try:
            with open(cfg_path, "r") as cfg_file:
                cfg = cfg_load_fn(cfg_file)
                aux = cfg["aux"]
                env = cfg_path.split("/")[-3]

                if env not in results:
                    results[env] = dict()
                if aux not in results[env]:
                    results[env][aux] = dict()

                df = pd.read_csv(res_path)
                results[env][aux][res_path] = (df, cfg)

        except Exception as e:
            print(f"Error processing {cfg_path}: {e}")

    return results


# Get environments
mujoco_envs = get_envs(Path("mujoco_code/envs.txt"))
minigrid_envs = get_envs(Path("minigrid_code/envs.txt"))

# Plot nominal results
nominal_mujoco_res = get_results(Path("mujoco_code/logs_nominal_mujoco"), "mujoco")
plot_results(
    mujoco_envs,
    nominal_mujoco_res,
    mujoco_auxs,
    "Nominal Mujoco Results",
    "./plots/nominal_mujoco.png",
)
nominal_minigrid_res = get_results(
    Path("minigrid_code/logs_nominal_minigrid"), "minigrid"
)
plot_results(
    minigrid_envs,
    nominal_minigrid_res,
    minigrid_auxs,
    "Nominal Minigrid Results",
    "./plots/nominal_minigrid.png",
)

# Bisim gamma ablation
bisim_gammas = [0.25, 0.50, 0.75, 1.0]  # baseline is 0.5
baseline_bisim_gamma = 0.50

# Bisim gamma mujoco ablation
ablate_bisim_gamma_mujoco_res = get_results(
    Path("mujoco_code/logs_ablate/bisim_gamma"), "mujoco"
)
plot_ablation(
    ablate_bisim_gamma_mujoco_res,
    nominal_mujoco_res,
    token_mujoco_env,
    ["bisim", "bisim_critic"],
    bisim_gammas,
    baseline_bisim_gamma,
    "bisim_gamma",
    "Bisim Gamma Ablation for Mujoco",
    "./plots/bisim_gamma_mujoco_ablation.png",
)

# Bisim gamma minigrid ablation
ablate_bisim_gamma_minigrid_res = get_results(
    Path("minigrid_code/logs_ablate/bisim_gamma"), "minigrid"
)
plot_ablation(
    ablate_bisim_gamma_minigrid_res,
    nominal_minigrid_res,
    token_minigrid_env,
    ["bisim", "bisim_critic"],
    bisim_gammas,
    baseline_bisim_gamma,
    "bisim_gamma",
    "Bisim Gamma Ablation for Minigrid",
    "./plots/bisim_gamma_minigrid_ablation.png",
)

# Wasserstein critic training steps ablation
wass_critic_train_steps_all = [1, 5, 10]
baseline_wass_critic_train_steps = 1

# Wasserstein critic training steps mujoco ablation
ablate_wass_critic_train_steps_mujoco_res = get_results(
    Path("mujoco_code/logs_ablate/wass_critic_train_steps"), "mujoco"
)
plot_ablation(
    ablate_wass_critic_train_steps_mujoco_res,
    nominal_mujoco_res,
    token_mujoco_env,
    ["bisim_critic", "zp_critic"],
    wass_critic_train_steps_all,
    baseline_wass_critic_train_steps,
    "wass_critic_train_steps",
    "Wass Critic Train Steps Ablation for Mujoco",
    "./plots/wass_critic_train_steps_mujoco_ablation.png",
)

# Wasserstein critic training steps minigrid ablation
ablate_wass_critic_train_steps_minigrid_res = get_results(
    Path("minigrid_code/logs_ablate/wass_critic_train_steps"), "minigrid"
)
plot_ablation(
    ablate_wass_critic_train_steps_minigrid_res,
    nominal_minigrid_res,
    token_minigrid_env,
    ["bisim_critic", "zp_critic"],
    wass_critic_train_steps_all,
    baseline_wass_critic_train_steps,
    "wass_critic_train_steps",
    "Wass Critic Train Steps Ablation for Minigrid",
    "./plots/wass_critic_train_steps_minigrid_ablation.png",
)

# Distractors mujoco
plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT * 2))
token_mujoco_distractor_env = token_mujoco_env + "-d32"

# Create a 2x2 grid for the 4 auxiliary tasks (skipping l2)
fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH * 2, FIG_HEIGHT * 2))
axes = axes.flatten()

plot_idx = 0
for aux in mujoco_auxs:
    if aux == "l2":
        continue

    ax = axes[plot_idx]
    plot_idx += 1

    for distractor in ["none", "gaussian", "interleaved_gaussian_mixture"]:
        if distractor == "none":
            dfs = nominal_mujoco_res[token_mujoco_env][aux]
            assert len(dfs) == 1
            df, _ = list(dfs.values())[0]
        else:
            cand_dfs = get_results(
                Path(f"mujoco_code/logs_distractors/{distractor}"), "mujoco"
            )[token_mujoco_distractor_env][aux]
            assert len(cand_dfs) == 1
            df, _ = list(cand_dfs.values())[0]

        valid_idxs = ~df["return"].isna()
        env_steps = df["env_steps"][valid_idxs]
        rewards = df["return"][valid_idxs]

        ax.plot(env_steps, rewards, label=distractor)

    style_plot(ax, f"{token_mujoco_env} {aux}")

plt.suptitle("Distractors Ablation for Mujoco", fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig("./plots/distractors_mujoco.png", dpi=DPI, bbox_inches="tight")
plt.close()
