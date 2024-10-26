import matplotlib.pyplot as plt
import glob
import json
import pandas as pd

PLOT_INCOMPLETE = True # False

def main():

    results = {}

    cfg_files = glob.glob("./logs/**/config.json", recursive=True)
    for cfg_file in cfg_files:
        cfg = json.load(open(cfg_file))
        env = cfg["env_name"]
        aux = cfg["aux"]   
        n_steps = cfg["num_steps"]
        ts = cfg["logdir"].split("/")[-1]

        res_file = cfg_file.replace("config.json", "progress.csv")
        if not glob.glob(res_file):
            print(f"Skipping {env} {aux} {n_steps} because it is missing")
            continue
        else:
            print(f"Loading {env} {aux} {n_steps} from {res_file}")
        df = pd.read_csv(res_file)

        if not PLOT_INCOMPLETE and str(int(n_steps)) not in df["env_steps"]:
            print(f"Skipping {env} {aux} {n_steps} because it is incomplete")
            continue

        if env not in results:
            results[env] = {}
        if aux not in results[env]:
            results[env][aux] = {}

        results[env][aux][ts] = df

    for env in results:
        legend_items = []
        for aux in results[env]:
            for i, ts in enumerate(results[env][aux]):
                df = results[env][aux][ts]
                valid_idxs = ~df["return"].isna()
                plt.plot(df[valid_idxs]["env_steps"], df[valid_idxs]["return"])

                if len(results[env][aux]) > 1:
                    legend_items.append(f"{aux}, {i}")
                else:
                    legend_items.append(f"{aux}")
        plt.legend(legend_items)
        plt.xlabel("Env Steps")
        plt.ylabel("Return")
        plt.title(f"Learning curves for {env}")
        plt.show()


if __name__ == "__main__":
    main()
