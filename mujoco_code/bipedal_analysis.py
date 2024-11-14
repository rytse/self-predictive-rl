import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import glob

INCLUDE_OLD = False
BISIM_ONLY = False

# Data loading remains the same
if INCLUDE_OLD:
    bipedal_new_cfg_files = glob.glob(
        "./logs/BipedalWalker-v3/**/flags.yml", recursive=True
    )
    bipedal_old_cfg_files = glob.glob(
        "./old_logs/BipedalWalker-v3/**/flags.yml", recursive=True
    )
    bipedal_cfg_files = bipedal_new_cfg_files + bipedal_old_cfg_files
else:
    bipedal_cfg_files = glob.glob(
        "./logs/BipedalWalker-v3/**/flags.yml", recursive=True
    )

print(f"{len(bipedal_cfg_files)=}")

bipedal_data = dict()
for bipedal_cfg_file in bipedal_cfg_files:
    bipedal_log_file = bipedal_cfg_file.replace("flags.yml", "progress.csv")
    try:
        bipedal_df = pd.read_csv(bipedal_log_file)
        assert "return" in bipedal_df.columns
        with open(bipedal_cfg_file, "r") as f:
            bipedal_cfg = yaml.safe_load(f)

            if BISIM_ONLY and "bisim" not in bipedal_cfg["aux"]:
                continue

            bipedal_data[bipedal_cfg_file] = dict()
            bipedal_data[bipedal_cfg_file]["cfg"] = bipedal_cfg
            bipedal_data[bipedal_cfg_file]["df"] = bipedal_df
    except:
        continue

# Create an interactive plot using plotly
fig = go.Figure()

# Add traces for each dataset
for cfg_file, data in bipedal_data.items():
    df = data["df"]
    idxs = ~df["return"].isna()
    env_steps = df["env_steps"][idxs]
    returns = df["return"][idxs]

    # Create a trace for each dataset
    fig.add_trace(
        go.Scatter(
            x=env_steps,
            y=returns,
            name=cfg_file.split("/")[-2],  # This will appear in hover text
            hovertemplate=(
                f"Path: {cfg_file}<br>"
                + "Environment Steps: %{x}<br>"
                + "Return: %{y}<br>"
                + "<extra></extra>"  # This removes the trace name from the hover box
            ),
            line=dict(width=2),
            opacity=1,
        )
    )

# Update layout
fig.update_layout(
    title="Experiment Results",
    xaxis_title="Environment Steps",
    yaxis_title="Return",
    width=1200,
    height=800,
    showlegend=False,  # Hide the legend as requested
    hovermode="closest",  # Show hover info for the closest point
    plot_bgcolor="white",  # White background
    paper_bgcolor="white",
)

# Add grid lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

# Add click-to-hide functionality through custom JavaScript
fig.update_traces(
    visible=True,
    opacity=1,
)

# Show the plot
fig.show()
