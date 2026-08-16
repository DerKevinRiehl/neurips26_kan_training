from pathlib import Path

import pandas as pd

from experiment2_common import (
    MODEL_LABELS,
    SUMMARY_DIR,
    best_setup_per_group,
    final_by_setup,
    final_checkpoint,
    find_result_csv,
    load_results,
)


SUMMARY_DIR.mkdir(exist_ok=True)

csv_path = find_result_csv()
df = load_results(csv_path)
final_ep = final_checkpoint(df)

by_setup = final_by_setup(df)
best_by_model_type = best_setup_per_group(df, ["environment_id", "model_type"])
best_by_family = best_setup_per_group(df, ["environment_id", "model_family"])

by_setup.to_csv(SUMMARY_DIR / "experiment2_final_by_setup.csv", index=False)
best_by_model_type.to_csv(SUMMARY_DIR / "experiment2_best_by_model_type.csv", index=False)
best_by_family.to_csv(SUMMARY_DIR / "experiment2_best_kan_vs_mlp.csv", index=False)

lines = []
lines.append(f"Data source: {csv_path.name}")
lines.append(f"Clean experiment rows: {len(df):,}")
lines.append(f"Environments: {', '.join(sorted(df['environment_id'].unique()))}")
lines.append(f"Model setups: {df['model_setup_id'].nunique()}")
lines.append(f"Random seeds: {df['random_seed'].nunique()}")
lines.append(f"Final checkpoint: {final_ep} training episodes")
lines.append(f"Full actor-critic comparison: {sorted(df['compare_actor_only'].astype(str).unique()) == ['False']}")
lines.append("")
lines.append("Best final setup per environment and model type:")
for _, row in best_by_model_type.iterrows():
    label = MODEL_LABELS.get(row["model_type"], row["model_type"])
    lines.append(
        f"- {row['environment_id']}: {label}, {row['model_setup_id']}, "
        f"return {row['eval_return_mean']:.3f} +/- {row['eval_return_std']:.3f}, "
        f"parameters {int(row['total_n_parameters'])}, seeds {int(row['seeds'])}"
    )

lines.append("")
lines.append("Best KAN family versus best MLP family at the final checkpoint:")
for env, sub in best_by_family.groupby("environment_id"):
    kan = sub[sub["model_family"] == "kan"].iloc[0]
    mlp = sub[sub["model_family"] == "mlp"].iloc[0]
    gap = kan["eval_return_mean"] - mlp["eval_return_mean"]
    winner = "KAN" if gap > 0 else "MLP" if gap < 0 else "tie"
    lines.append(
        f"- {env}: best KAN {kan['model_setup_id']} ({kan['eval_return_mean']:.3f}) "
        f"vs best MLP {mlp['model_setup_id']} ({mlp['eval_return_mean']:.3f}); "
        f"gap KAN-MLP {gap:.3f}, winner {winner}"
    )

lines.append("")
lines.append("Quick reading:")
lines.append("- CartPole-v1 is solved by both KAN families; the best MLP ReLU is almost saturated as well.")
lines.append("- MountainCar-v0 stays at the failure floor for every family in this run.")
lines.append("- MountainCarContinuous-v0 is strongest for KAN RBF and MLP ReLU, with KAN RBF using fewer parameters in the best final setup.")
lines.append("- Pendulum-v1 favors MLP ReLU in this protocol; KAN RBF learns but remains worse at the final checkpoint.")
lines.append("- Acrobot-v1 favors MLP ReLU at the final checkpoint, while KAN B-spline is close and KAN RBF struggles.")

summary_path = SUMMARY_DIR / "analysis_summary_experiment2.txt"
summary_path.write_text("\n".join(lines), encoding="utf-8")

print(summary_path)
print(pd.DataFrame(best_by_family))

