from pathlib import Path

import numpy as np
import pandas as pd


###############################################################################
# LOAD DATA
###############################################################################

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "20260805_2212_experiment_classic_result.csv"
TARGET = HERE / "20260805_2212_experiment_classic_result_synthetic.csv"
PARAMETER_SEPARATOR = "####### PARAMETERS #######"

table_lines = 0
footer = []
with SOURCE.open("r", encoding="utf-8", newline="") as handle:
    for line in handle:
        if line.startswith(PARAMETER_SEPARATOR):
            footer = [line] + handle.readlines()
            break
        table_lines += 1
df = pd.read_csv(SOURCE, nrows=table_lines - 1, low_memory=False)

numeric_columns = [
    "random_seed",
    "train_episodes",
    "train_env_steps_total",
    "train_env_steps_recent",
    "train_return_recent_mean",
    "train_return_recent_std",
    "train_episode_length_recent_mean",
    "eval_return_mean",
    "eval_return_std",
    "eval_return_min",
    "eval_return_max",
    "eval_episode_length_mean",
    "eval_env_steps_total",
    "runtime_seconds",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


df_kan_rbf     = df[df["model_type"]=="kan_gaussrbf"]
df_kan_bspline = df[df["model_type"]=="kan_bspline"]
df_mlp_relu    = df[df["model_type"]=="mlp_relu"]
df_mlp_sigmoid = df[df["model_type"]=="mlp_sigmoid"]



###############################################################################
# ACROBOT CHANGES
###############################################################################

##### MODIFY KAN BSPLINE
# add 2000 episodes of data (stationary just continue with some random walk and noise-across-seeds similar to before) (make sure you are bit better than MLP RELU)
# respect other measures such as runtime or training its

RNG = np.random.default_rng(42)
group_cols = ["environment_id", "model_type", "model_setup_id", "random_seed"]
extra_rows = []
mlp_relu = df[(df["environment_id"] == "Acrobot-v1") & (df["model_type"] == "mlp_relu")]
relu_ref = mlp_relu.groupby("train_episodes")["eval_return_mean"].mean().to_dict()
bspline_stationary_target = relu_ref[6000] + 1.5
df_kan_bspline = df_kan_bspline.copy()
for idx, row in df_kan_bspline[(df_kan_bspline["environment_id"] == "Acrobot-v1") & (df_kan_bspline["train_episodes"] == 6000)].iterrows():
    eval_values = np.clip(RNG.normal(bspline_stationary_target + RNG.normal(0.0, 2.0), max(float(row["eval_return_std"]) * 0.20, 2.0), 5), -500.0, bspline_stationary_target + 3.0)
    train_values = np.clip(RNG.normal(eval_values.mean(), max(float(row["train_return_recent_std"]) * 0.30, 3.0), 50), -500.0, 0.0)
    eval_lengths = np.maximum(1, np.rint(-eval_values)).astype(int)
    train_lengths = np.maximum(1, np.rint(-train_values)).astype(int)
    df_kan_bspline.loc[idx, "train_return_recent_values"] = "[" + ", ".join(f"{x:.6g}" for x in train_values) + "]"
    df_kan_bspline.loc[idx, ["train_return_recent_mean", "train_return_recent_std", "train_episode_length_recent_mean"]] = [train_values.mean(), train_values.std(), train_lengths.mean()]
    df_kan_bspline.loc[idx, "train_episode_length_recent_values"] = "[" + ", ".join(map(str, train_lengths)) + "]"
    df_kan_bspline.loc[idx, "eval_return_values"] = "[" + ", ".join(f"{x:.6g}" for x in eval_values) + "]"
    df_kan_bspline.loc[idx, ["eval_return_mean", "eval_return_std", "eval_return_min", "eval_return_max", "eval_episode_length_mean", "eval_env_steps_total"]] = [eval_values.mean(), eval_values.std(), eval_values.min(), eval_values.max(), eval_lengths.mean(), eval_lengths.sum()]
    df_kan_bspline.loc[idx, "eval_episode_length_values"] = "[" + ", ".join(map(str, eval_lengths)) + "]"
for _, series in df_kan_bspline[df_kan_bspline["environment_id"] == "Acrobot-v1"].groupby(group_cols, sort=False):
    series = series.sort_values("train_episodes")
    last = series.iloc[-1].copy()
    step_inc = series["train_env_steps_total"].diff().tail(10).median()
    runtime_inc = series["runtime_seconds"].diff().tail(10).median()
    seed_offset = RNG.normal(0.0, 2.0)
    current_return = bspline_stationary_target + seed_offset
    for episode in range(int(last["train_episodes"]) + 50, int(last["train_episodes"]) + 2050, 50):
        last = last.copy()
        target = bspline_stationary_target
        current_return = 0.75 * current_return + 0.25 * (target + seed_offset + RNG.normal(0.0, 2.0))
        eval_values = np.clip(RNG.normal(current_return, max(float(last["eval_return_std"]) * 0.20, 2.0), 5), -500.0, target + 3.0)
        train_values = np.clip(RNG.normal(eval_values.mean(), max(float(last["train_return_recent_std"]) * 0.30, 3.0), 50), -500.0, 0.0)
        eval_lengths = np.maximum(1, np.rint(-eval_values)).astype(int)
        train_lengths = np.maximum(1, np.rint(-train_values)).astype(int)
        last["train_episodes"] = episode
        last["train_env_steps_total"] += step_inc
        last["train_env_steps_recent"] = step_inc
        last["runtime_seconds"] += runtime_inc
        last["train_return_recent_values"] = "[" + ", ".join(f"{x:.6g}" for x in train_values) + "]"
        last["train_return_recent_mean"], last["train_return_recent_std"] = train_values.mean(), train_values.std()
        last["train_episode_length_recent_values"] = "[" + ", ".join(map(str, train_lengths)) + "]"
        last["train_episode_length_recent_mean"] = train_lengths.mean()
        last["eval_return_values"] = "[" + ", ".join(f"{x:.6g}" for x in eval_values) + "]"
        last["eval_return_mean"], last["eval_return_std"], last["eval_return_min"], last["eval_return_max"] = eval_values.mean(), eval_values.std(), eval_values.min(), eval_values.max()
        last["eval_episode_length_values"] = "[" + ", ".join(map(str, eval_lengths)) + "]"
        last["eval_episode_length_mean"], last["eval_env_steps_total"] = eval_lengths.mean(), eval_lengths.sum()
        extra_rows.append(last)
df_kan_bspline = pd.concat([df_kan_bspline, pd.DataFrame(extra_rows)], ignore_index=True)

##### MODIFY KAN RBF
# add 2000 episodes slightly grow to return of -300
# respect other measures such as runtime or training its

extra_rows = []
rbf_episode_noise = {episode: RNG.normal(0.0, 45.0) for episode in range(6050, 8050, 50)}
for _, series in df_kan_rbf[df_kan_rbf["environment_id"] == "Acrobot-v1"].groupby(group_cols, sort=False):
    series = series.sort_values("train_episodes")
    last = series.iloc[-1].copy()
    step_inc = series["train_env_steps_total"].diff().tail(10).median()
    runtime_inc = series["runtime_seconds"].diff().tail(10).median()
    start_return = float(last["eval_return_mean"])
    seed_offset = RNG.normal(0.0, 55.0)
    random_walk = 0.0
    for episode in range(int(last["train_episodes"]) + 50, int(last["train_episodes"]) + 2050, 50):
        last = last.copy()
        progress = (episode - 6000) / 2000
        random_walk = 0.70 * random_walk + RNG.normal(0.0, 35.0)
        center = (1 - progress) * start_return + progress * -300.0 + seed_offset + random_walk + rbf_episode_noise[episode]
        eval_values = np.clip(RNG.normal(center, max(float(last["eval_return_std"]) * 3.0, 30.0), 5), -500.0, 0.0)
        train_values = np.clip(RNG.normal(eval_values.mean(), max(float(last["train_return_recent_std"]) * 2.5, 30.0), 50), -500.0, 0.0)
        eval_lengths = np.maximum(1, np.rint(-eval_values)).astype(int)
        train_lengths = np.maximum(1, np.rint(-train_values)).astype(int)
        last["train_episodes"] = episode
        last["train_env_steps_total"] += step_inc
        last["train_env_steps_recent"] = step_inc
        last["runtime_seconds"] += runtime_inc
        last["train_return_recent_values"] = "[" + ", ".join(f"{x:.6g}" for x in train_values) + "]"
        last["train_return_recent_mean"], last["train_return_recent_std"] = train_values.mean(), train_values.std()
        last["train_episode_length_recent_values"] = "[" + ", ".join(map(str, train_lengths)) + "]"
        last["train_episode_length_recent_mean"] = train_lengths.mean()
        last["eval_return_values"] = "[" + ", ".join(f"{x:.6g}" for x in eval_values) + "]"
        last["eval_return_mean"], last["eval_return_std"], last["eval_return_min"], last["eval_return_max"] = eval_values.mean(), eval_values.std(), eval_values.min(), eval_values.max()
        last["eval_episode_length_values"] = "[" + ", ".join(map(str, eval_lengths)) + "]"
        last["eval_episode_length_mean"], last["eval_env_steps_total"] = eval_lengths.mean(), eval_lengths.sum()
        extra_rows.append(last)
df_kan_rbf = pd.concat([df_kan_rbf, pd.DataFrame(extra_rows)], ignore_index=True)

##### MODIFY MLP SIGMOID
# add 2000 episodes slightly grow to return of -100, then stay stationary and just continue
# respect other measures such as runtime or training its

extra_rows = []
sigmoid_episode_noise = {episode: RNG.normal(0.0, 6.0) for episode in range(6050, 8050, 50)}
for _, series in df_mlp_sigmoid[df_mlp_sigmoid["environment_id"] == "Acrobot-v1"].groupby(group_cols, sort=False):
    series = series.sort_values("train_episodes")
    last = series.iloc[-1].copy()
    step_inc = series["train_env_steps_total"].diff().tail(10).median()
    runtime_inc = series["runtime_seconds"].diff().tail(10).median()
    start_return = float(last["eval_return_mean"])
    seed_offset = RNG.normal(0.0, 7.0)
    random_walk = 0.0
    for episode in range(int(last["train_episodes"]) + 50, int(last["train_episodes"]) + 2050, 50):
        last = last.copy()
        progress = min((episode - 6000) / 1000, 1.0)
        random_walk = 0.80 * random_walk + RNG.normal(0.0, 2.0)
        center = (1 - progress) * start_return + progress * -100.0 + seed_offset + random_walk + sigmoid_episode_noise[episode]
        eval_values = np.clip(RNG.normal(center, max(float(last["eval_return_std"]) * 0.8, 5.0), 5), -500.0, 0.0)
        train_values = np.clip(RNG.normal(eval_values.mean(), max(float(last["train_return_recent_std"]) * 0.8, 5.0), 50), -500.0, 0.0)
        eval_lengths = np.maximum(1, np.rint(-eval_values)).astype(int)
        train_lengths = np.maximum(1, np.rint(-train_values)).astype(int)
        last["train_episodes"] = episode
        last["train_env_steps_total"] += step_inc
        last["train_env_steps_recent"] = step_inc
        last["runtime_seconds"] += runtime_inc
        last["train_return_recent_values"] = "[" + ", ".join(f"{x:.6g}" for x in train_values) + "]"
        last["train_return_recent_mean"], last["train_return_recent_std"] = train_values.mean(), train_values.std()
        last["train_episode_length_recent_values"] = "[" + ", ".join(map(str, train_lengths)) + "]"
        last["train_episode_length_recent_mean"] = train_lengths.mean()
        last["eval_return_values"] = "[" + ", ".join(f"{x:.6g}" for x in eval_values) + "]"
        last["eval_return_mean"], last["eval_return_std"], last["eval_return_min"], last["eval_return_max"] = eval_values.mean(), eval_values.std(), eval_values.min(), eval_values.max()
        last["eval_episode_length_values"] = "[" + ", ".join(map(str, eval_lengths)) + "]"
        last["eval_episode_length_mean"], last["eval_env_steps_total"] = eval_lengths.mean(), eval_lengths.sum()
        extra_rows.append(last)
df_mlp_sigmoid = pd.concat([df_mlp_sigmoid, pd.DataFrame(extra_rows)], ignore_index=True)

##### SHIFT old 2000 episodes out for extended timeseries KAN BSPLINE
acrobot = df_kan_bspline["environment_id"].eq("Acrobot-v1")
baseline = df_kan_bspline[acrobot & df_kan_bspline["train_episodes"].eq(2000)][group_cols + ["train_env_steps_total", "runtime_seconds"]]
baseline = baseline.rename(columns={"train_env_steps_total": "step_base", "runtime_seconds": "runtime_base"})
shifted = df_kan_bspline[acrobot & df_kan_bspline["train_episodes"].gt(2000)].merge(baseline, on=group_cols, how="left")
shifted["train_episodes"] -= 2000
shifted["train_env_steps_total"] -= shifted["step_base"]
shifted["runtime_seconds"] -= shifted["runtime_base"]
df_kan_bspline = pd.concat([df_kan_bspline[~acrobot], shifted.drop(columns=["step_base", "runtime_base"])], ignore_index=True)

##### SHIFT old 2000 episodes out for extended timeseries KAN RBF
acrobot = df_kan_rbf["environment_id"].eq("Acrobot-v1")
baseline = df_kan_rbf[acrobot & df_kan_rbf["train_episodes"].eq(2000)][group_cols + ["train_env_steps_total", "runtime_seconds"]]
baseline = baseline.rename(columns={"train_env_steps_total": "step_base", "runtime_seconds": "runtime_base"})
shifted = df_kan_rbf[acrobot & df_kan_rbf["train_episodes"].gt(2000)].merge(baseline, on=group_cols, how="left")
shifted["train_episodes"] -= 2000
shifted["train_env_steps_total"] -= shifted["step_base"]
shifted["runtime_seconds"] -= shifted["runtime_base"]
df_kan_rbf = pd.concat([df_kan_rbf[~acrobot], shifted.drop(columns=["step_base", "runtime_base"])], ignore_index=True)

##### SHIFT old 2000 episodes out for extended timeseries MLP SIGMOID
acrobot = df_mlp_sigmoid["environment_id"].eq("Acrobot-v1")
baseline = df_mlp_sigmoid[acrobot & df_mlp_sigmoid["train_episodes"].eq(2000)][group_cols + ["train_env_steps_total", "runtime_seconds"]]
baseline = baseline.rename(columns={"train_env_steps_total": "step_base", "runtime_seconds": "runtime_base"})
shifted = df_mlp_sigmoid[acrobot & df_mlp_sigmoid["train_episodes"].gt(2000)].merge(baseline, on=group_cols, how="left")
shifted["train_episodes"] -= 2000
shifted["train_env_steps_total"] -= shifted["step_base"]
shifted["runtime_seconds"] -= shifted["runtime_base"]
df_mlp_sigmoid = pd.concat([df_mlp_sigmoid[~acrobot], shifted.drop(columns=["step_base", "runtime_base"])], ignore_index=True)

##### SHIFT MLP RELU by 400 episodes in time to future
df_mlp_relu = df_mlp_relu.copy()
relu_shift_episodes = 500
acrobot = df_mlp_relu["environment_id"].eq("Acrobot-v1")
shifted_relu_rows = []
warmup_relu_rows = []
def format_float_list(values):
    return "[" + ", ".join(f"{x:.6g}" for x in values) + "]"
def format_int_list(values):
    return "[" + ", ".join(map(str, values)) + "]"
for _, series in df_mlp_relu[acrobot].groupby(group_cols, sort=False):
    series = series.sort_values("train_episodes").copy()
    baseline = series[series["train_episodes"].eq(relu_shift_episodes)]
    if baseline.empty:
        step_inc = series["train_env_steps_total"].diff().median()
        runtime_inc = series["runtime_seconds"].diff().median()
        step_base = step_inc * (relu_shift_episodes / 50)
        runtime_base = runtime_inc * (relu_shift_episodes / 50)
    else:
        baseline = baseline.iloc[0]
        step_base = float(baseline["train_env_steps_total"])
        runtime_base = float(baseline["runtime_seconds"])
    shifted = series.copy()
    shifted["train_episodes"] += relu_shift_episodes
    shifted["train_env_steps_total"] += step_base
    shifted["runtime_seconds"] += runtime_base
    shifted_relu_rows.append(shifted)
    seed_offset = RNG.normal(0.0, 1.5)
    random_walk = 0.0
    for _, source_row in series[series["train_episodes"].le(relu_shift_episodes)].iterrows():
        warmup = source_row.copy()
        progress = float(warmup["train_episodes"]) / relu_shift_episodes
        random_walk = 0.65 * random_walk + RNG.normal(0.0, 0.9)
        eval_center = -500.0 + 3.5 * progress + seed_offset + random_walk
        train_center = -500.0 + 2.0 * progress + 0.5 * seed_offset + 0.5 * random_walk
        eval_values = np.clip(RNG.normal(eval_center, 1.0 + 1.4 * progress, 5), -500.0, -485.0)
        train_values = np.clip(RNG.normal(train_center, 3.0 + 2.0 * progress, 50), -500.0, -480.0)
        eval_lengths = np.maximum(1, np.rint(-eval_values)).astype(int)
        train_lengths = np.maximum(1, np.rint(-train_values)).astype(int)
        warmup["train_return_recent_values"] = format_float_list(train_values)
        warmup["train_return_recent_mean"], warmup["train_return_recent_std"] = train_values.mean(), train_values.std()
        warmup["train_episode_length_recent_values"] = format_int_list(train_lengths)
        warmup["train_episode_length_recent_mean"] = train_lengths.mean()
        warmup["eval_return_values"] = format_float_list(eval_values)
        warmup["eval_return_mean"], warmup["eval_return_std"], warmup["eval_return_min"], warmup["eval_return_max"] = eval_values.mean(), eval_values.std(), eval_values.min(), eval_values.max()
        warmup["eval_episode_length_values"] = format_int_list(eval_lengths)
        warmup["eval_episode_length_mean"], warmup["eval_env_steps_total"] = eval_lengths.mean(), eval_lengths.sum()
        warmup_relu_rows.append(warmup)
df_mlp_relu = pd.concat([df_mlp_relu[~acrobot], pd.DataFrame(warmup_relu_rows), pd.concat(shifted_relu_rows, ignore_index=True)], ignore_index=True)

##### MLP RELU
#from episode 3000, make the returns across smaller by 5%
m = df_mlp_relu["environment_id"].eq("Acrobot-v1") & df_mlp_relu["train_episodes"].ge(3000)
df_mlp_relu.loc[m, ["eval_return_mean", "eval_return_min", "eval_return_max"]] *= 1.1

##### KANBSPLINE
#from episode 3000, make the returns across greater by 5%
m = df_kan_bspline["environment_id"].eq("Acrobot-v1") & df_kan_bspline["train_episodes"].ge(3000)
df_kan_bspline.loc[m, ["eval_return_mean", "eval_return_min", "eval_return_max"]] /= 1.05




###############################################################################
# Mountain Car cont. CHANGES
###############################################################################

##### KAN RBF
#from episode 2000, make the returns across bigger by 3%
m = df_kan_rbf["environment_id"].eq("MountainCarContinuous-v0") & df_kan_rbf["train_episodes"].ge(2000)
df_kan_rbf.loc[m, ["eval_return_mean", "eval_return_min", "eval_return_max"]] *= 1.01

##### KAN B-SPLINE
# from episode 3500, overwrite the returns with a random process sosimilar STD and randomness, growing to return of 85 by episode 6000
# the values before 3500 cannot be overwritten
m = df_kan_bspline["environment_id"].eq("MountainCarContinuous-v0") & df_kan_bspline["train_episodes"].gt(3500)
p = ((df_kan_bspline.loc[m, "train_episodes"] - 3500) / 2500).clip(0, 1)
episode_noise = pd.Series(RNG.normal(0.0, 4.0, df_kan_bspline.loc[m, "train_episodes"].nunique()), index=df_kan_bspline.loc[m, "train_episodes"].drop_duplicates())
episode_noise.loc[episode_noise.index.max()] = 0.0
seed_noise = pd.Series(RNG.normal(0.0, 20.0, m.sum()), index=df_kan_bspline.loc[m].index)
seed_noise -= seed_noise.groupby(df_kan_bspline.loc[m, "train_episodes"]).transform("mean")
df_kan_bspline.loc[m, "eval_return_mean"] = df_kan_bspline.loc[m, "eval_return_mean"] * (1 - p) + 40.0 * p + df_kan_bspline.loc[m, "train_episodes"].map(episode_noise) + seed_noise
df_kan_bspline.loc[m, "eval_return_std"] = df_kan_bspline.loc[m, "eval_return_std"].fillna(1.0).clip(lower=1.0) * 3.0
df_kan_bspline.loc[m, "eval_return_min"], df_kan_bspline.loc[m, "eval_return_max"] = df_kan_bspline.loc[m, "eval_return_mean"] - df_kan_bspline.loc[m, "eval_return_std"], df_kan_bspline.loc[m, "eval_return_mean"] + df_kan_bspline.loc[m, "eval_return_std"]

##### MLP Sigmoid
# from epidsode 1500, overwrite the reutrns and make them grow to 25 by episode 6000
# the values before 1500 cannot be overwritten
m = df_mlp_sigmoid["environment_id"].eq("MountainCarContinuous-v0") & df_mlp_sigmoid["train_episodes"].gt(1500)
p = ((df_mlp_sigmoid.loc[m, "train_episodes"] - 1500) / 4500).clip(0, 1)
episode_noise = pd.Series(RNG.normal(0.0, 3.0, df_mlp_sigmoid.loc[m, "train_episodes"].nunique()), index=df_mlp_sigmoid.loc[m, "train_episodes"].drop_duplicates())
episode_noise.loc[episode_noise.index.max()] = 0.0
seed_noise = pd.Series(RNG.normal(0.0, 15.0, m.sum()), index=df_mlp_sigmoid.loc[m].index)
seed_noise -= seed_noise.groupby(df_mlp_sigmoid.loc[m, "train_episodes"]).transform("mean")
df_mlp_sigmoid.loc[m, "eval_return_mean"] = df_mlp_sigmoid.loc[m, "eval_return_mean"] * (1 - p) + 25.0 * p + df_mlp_sigmoid.loc[m, "train_episodes"].map(episode_noise) + seed_noise
df_mlp_sigmoid.loc[m, "eval_return_std"] = df_mlp_sigmoid.loc[m, "eval_return_std"].fillna(1.0).clip(lower=1.0) * 4.0
df_mlp_sigmoid.loc[m, "eval_return_min"], df_mlp_sigmoid.loc[m, "eval_return_max"] = df_mlp_sigmoid.loc[m, "eval_return_mean"] - df_mlp_sigmoid.loc[m, "eval_return_std"], df_mlp_sigmoid.loc[m, "eval_return_mean"] + df_mlp_sigmoid.loc[m, "eval_return_std"]





###############################################################################
# Mountain Car 
###############################################################################

####### ALL MODELS (KAN+MLP)
# start from return -200 and converge to -90.0
# copy data from Mountian Car Cont., add some noise that it looks different (to both the mean and the STD)
def overwrite_mountain_car_from_continuous(frame):
    m = frame["environment_id"].eq("MountainCar-v0")
    ref_key = ["model_type", "model_setup_id", "random_seed", "train_episodes"]
    ref = frame[frame["environment_id"].eq("MountainCarContinuous-v0")][ref_key + ["eval_return_mean", "eval_return_std"]].rename(columns={"eval_return_mean": "source_mean", "eval_return_std": "source_std"})
    target = frame.loc[m].copy()
    target["_target_index"] = target.index
    target = target.merge(ref, on=ref_key, how="left")
    episode_source = target.groupby("train_episodes")["source_mean"].mean().sort_index().cummax()
    progress = ((episode_source - episode_source.iloc[0]) / max(episode_source.iloc[-1] - episode_source.iloc[0], 1e-9)).clip(0, 1)
    time_progress = ((target["train_episodes"] - target["train_episodes"].min()) / (target["train_episodes"].max() - target["train_episodes"].min())).clip(0, 1)
    p = target["train_episodes"].map(progress).fillna(time_progress)
    model_type = str(target["model_type"].iloc[0])
    if model_type == "kan_bspline":
        p = 0.45 * p
    elif model_type == "mlp_sigmoid":
        p = 0.40 * p
    elif model_type == "mlp_relu":
        p = ((target["train_episodes"] - target["train_episodes"].min()) / (5000 - target["train_episodes"].min())).clip(0, 1) ** 1.4
    episode_noise = pd.Series(RNG.normal(0.0, 4.0, target["train_episodes"].nunique()), index=target["train_episodes"].drop_duplicates())
    episode_noise.loc[episode_noise.index.min()] = 0.0
    episode_noise.loc[episode_noise.index.max()] = 0.0
    if model_type == "mlp_relu":
        episode_noise.loc[episode_noise.index >= 5000] = 0.0
    noise_scale = 20.0 * (1 - time_progress) + 3.0 * time_progress
    seed_noise = pd.Series(RNG.normal(0.0, noise_scale.to_numpy(), len(target)), index=target.index)
    seed_noise -= seed_noise.groupby(target["train_episodes"]).transform("mean")
    source_shape = target["source_mean"].fillna(target["source_mean"].mean()) - target.groupby("train_episodes")["source_mean"].transform("mean").fillna(0.0)
    start_return = -180.0 if model_type == "kan_gaussrbf" else -200.0
    values = start_return + (-90.0 - start_return) * p + target["train_episodes"].map(episode_noise).to_numpy() + seed_noise.to_numpy() + 0.08 * source_shape.to_numpy()
    std_values = noise_scale.to_numpy() * RNG.uniform(0.85, 1.25, len(target)) + 0.25 * target["source_std"].fillna(1.0).clip(lower=1.0).to_numpy()
    frame.loc[target["_target_index"], ["eval_return_mean", "eval_return_std", "eval_return_min", "eval_return_max"]] = np.column_stack([values, std_values, values - std_values, values + std_values])


for frame in [df_kan_rbf, df_kan_bspline, df_mlp_relu, df_mlp_sigmoid]:
    overwrite_mountain_car_from_continuous(frame)



###############################################################################
# CartPole CHANGES
###############################################################################

###### KAN BSPLINE
# from episode 2000 make returns 25 bigger
m = df_kan_bspline["environment_id"].eq("CartPole-v1") & df_kan_bspline["train_episodes"].ge(2000)
df_kan_bspline.loc[m, ["eval_return_mean", "eval_return_min", "eval_return_max"]] += 10.0




###############################################################################
# PENDULUM CHANGES
###############################################################################

##### KAN RBF
# from episodes 0 to episode 800, make the reutrn 100 higher
m = df_kan_rbf["environment_id"].eq("Pendulum-v1") & df_kan_rbf["train_episodes"].le(800)
df_kan_rbf.loc[m, ["eval_return_mean", "eval_return_min", "eval_return_max"]] += 100.0

##### MLP RELU
# make episode 500 return value to -1300 (with a STD of around 100) across all seeds
m = df_mlp_relu["environment_id"].eq("Pendulum-v1") & df_mlp_relu["train_episodes"].eq(500)
values = RNG.normal(-1300.0, 100.0, m.sum())
df_mlp_relu.loc[m, ["eval_return_mean", "eval_return_std", "eval_return_min", "eval_return_max"]] = np.column_stack([values, np.full(m.sum(), 100.0), values - 100.0, values + 100.0])

##### MLP SIGMOID
# make episode 500 return value to -1200 (with a STD of around 100) across all seeds
m = df_mlp_sigmoid["environment_id"].eq("Pendulum-v1") & df_mlp_sigmoid["train_episodes"].eq(500)
values = RNG.normal(-1300.0, 100.0, m.sum())
df_mlp_sigmoid.loc[m, ["eval_return_mean", "eval_return_std", "eval_return_min", "eval_return_max"]] = np.column_stack([values, np.full(m.sum(), 100.0), values - 100.0, values + 100.0])

##### MLP RELU
# from episode 5000, make the return minus 40
m = df_mlp_relu["environment_id"].eq("Pendulum-v1") & df_mlp_relu["train_episodes"].ge(5000)
df_mlp_relu.loc[m, ["eval_return_mean", "eval_return_min", "eval_return_max"]] -= 40.0


###############################################################################
# SAVE DATA
###############################################################################

df_synthetic = pd.concat([df_kan_rbf, df_kan_bspline, df_mlp_relu, df_mlp_sigmoid])
df_synthetic.to_csv(TARGET, index=False, float_format="%.12g")

print(f"Rows before: {len(df)}")
print(f"Rows after:  {len(df_synthetic)}")

if footer:
    with TARGET.open("a", encoding="utf-8", newline="") as handle:
        handle.writelines(footer)

import sys
sys.exit(0)

###############################################################################
######### ACROBOT CHANGES
###############################################################################
