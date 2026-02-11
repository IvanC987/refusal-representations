import json
from collections import defaultdict

from refusal_representations.paths import PROJECT_ROOT


def main():
    input_path = PROJECT_ROOT / "experiments" / "llama_outputs_1b" / "sequential_rollouts_tier_2_evaluated.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group by (layer, alpha)
    buckets = defaultdict(list)
    for entry in data:
        if entry["response"].strip():
            layer = entry["layer"]
            alpha = entry["alpha"]
            buckets[(layer, alpha)].append(entry)

    aggregated = []

    for (layer, alpha), entries in buckets.items():
        compliance_vals = []
        coherence_vals = []
        product_vals = []

        compliance_counts = {0.0: 0, 0.5: 0, 1.0: 0}
        coherence_counts = {0.0: 0, 0.5: 0, 1.0: 0}

        for e in entries:
            c = float(e["judge_metrics"]["compliance"])
            h = float(e["judge_metrics"]["coherence"])

            compliance_vals.append(c)
            coherence_vals.append(h)
            product_vals.append(c ** 3 * h)

            compliance_counts[c] += 1
            coherence_counts[h] += 1

        n = len(entries)

        aggregated.append({
            "layer": layer,
            "alpha": alpha,
            "num_samples": n,

            "mean_compliance": sum(compliance_vals) / n,
            "mean_coherence": sum(coherence_vals) / n,
            "mean_compliance_coherence_product": sum(product_vals) / n,

            "compliance_counts": compliance_counts,
            "coherence_counts": coherence_counts,
        })

    # Sort by mean product (descending)
    aggregated.sort(
        key=lambda x: x["mean_compliance_coherence_product"],
        reverse=True,
    )

    output_path = PROJECT_ROOT / "experiments" / "llama_outputs_1b" / "sequential_tier_2_aggregate_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=4, ensure_ascii=False)

    print(f"Saved aggregate metrics to {output_path}")


if __name__ == "__main__":
    main()
