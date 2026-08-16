import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiment2_common import (
    FIGURE_DIR,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    best_setup_per_group,
    find_result_csv,
    load_results,
    mean_std_curve,
    safe_name,
)


FIGURE_DIR.mkdir(exist_ok=True)

csv_path = find_result_csv()
df = load_results(csv_path)
best = best_setup_per_group(df, ["environment_id", "model_type"])

for env in sorted(df["environment_id"].unique()):
    plt.figure(figsize=(8.4, 5.0))
    for model_type in MODEL_ORDER:
        row = best[(best["environment_id"] == env) & (best["model_type"] == model_type)]
        if row.empty:
            continue
        row = row.iloc[0]
        curve = mean_std_curve(df, env, row["model_setup_id"])
        x = curve["train_env_steps_mean"].to_numpy() / 1000.0
        y = curve["eval_return_mean"].to_numpy()
        y_std = curve["eval_return_std"].fillna(0.0).to_numpy()
        label = f"{MODEL_LABELS[model_type]} ({row['model_setup_id']}, p={int(row['total_n_parameters'])})"
        plt.plot(x, y, color=MODEL_COLORS[model_type], linewidth=2.0, label=label)
        plt.fill_between(x, y - y_std, y + y_std, color=MODEL_COLORS[model_type], alpha=0.15, linewidth=0)

    plt.title(f"{env}: best final setup per model type")
    plt.xlabel("Environment steps during training (thousands)")
    plt.ylabel("Evaluation return, mean +/- std over seeds")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = FIGURE_DIR / f"{safe_name(env)}__learning_curves_best_model_type.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(out_path)

