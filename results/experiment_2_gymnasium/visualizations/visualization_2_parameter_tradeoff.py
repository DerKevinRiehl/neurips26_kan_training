import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiment2_common import (
    FIGURE_DIR,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    final_by_setup,
    find_result_csv,
    load_results,
    safe_name,
)


FIGURE_DIR.mkdir(exist_ok=True)

csv_path = find_result_csv()
df = load_results(csv_path)
final = final_by_setup(df)

for env in sorted(final["environment_id"].unique()):
    env_final = final[final["environment_id"] == env].copy()
    plt.figure(figsize=(8.0, 5.0))
    for model_type in MODEL_ORDER:
        sub = env_final[env_final["model_type"] == model_type].sort_values("total_n_parameters")
        if sub.empty:
            continue
        plt.errorbar(
            sub["total_n_parameters"],
            sub["eval_return_mean"],
            yerr=sub["eval_return_std"].fillna(0.0),
            fmt="o-",
            capsize=3,
            color=MODEL_COLORS[model_type],
            label=MODEL_LABELS[model_type],
            linewidth=1.5,
            markersize=4.5,
        )

    plt.xscale("log")
    plt.title(f"{env}: final return versus parameter count")
    plt.xlabel("Actor + critic parameters (log scale)")
    plt.ylabel("Final evaluation return, mean +/- std over seeds")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = FIGURE_DIR / f"{safe_name(env)}__final_return_vs_parameters.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(out_path)

