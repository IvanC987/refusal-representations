import torch

from refusal_representations.extraction.llama.sequential_layer_sweep import sweep_layers
from refusal_representations.paths import PROJECT_ROOT


"""
Tier 1

Loaded pretrained model meta-llama/Llama-3.2-1B-Instruct into HookedTransformer
Moving model to device:  cuda
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:58<00:00, 17.08it/s]
k=0, v=nan
k=15, v=6.841723439919401
k=13, v=6.764501404829828
k=14, v=6.580004533330428
k=11, v=6.495865305200624
k=12, v=6.490000683620485
k=10, v=6.382524069548681
k=9, v=6.047731833582353
k=8, v=4.640843620210902
k=7, v=3.737893902030972
k=6, v=2.854287739896825
k=5, v=2.662293651732618
k=4, v=2.127323102103064
k=3, v=1.3090393642419995
k=2, v=1.234300177192351
k=1, v=1.091896776697546

Tier 2
k=0, v=nan
k=15, v=5.421250354582013
k=14, v=5.341846720872403
k=13, v=5.230778805813765
k=12, v=5.042117217125537
k=10, v=5.024492809842924
k=11, v=5.0173866451826195
k=9, v=4.853614308765755
k=8, v=3.850004438761051
k=7, v=3.1118378208105772
k=6, v=2.272457320042407
k=5, v=2.000291876582343
k=4, v=1.5020170761145977
k=3, v=1.1314363682454887
k=2, v=0.9836438675674024
k=1, v=0.7251015064438723

"""


def main():
    torch.random.manual_seed(89)
    torch.cuda.manual_seed(89)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    harmful_tier = 2
    json_path = PROJECT_ROOT / "data" / f"generated_pairs_tier_{harmful_tier}.json"

    refusal_vectors, cohens_d = sweep_layers(model_id=model_id, device=device, json_path=json_path)

    sorted_d = sorted(cohens_d.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_d:
        print(f"{k=}, {v=}")

    out_dir = PROJECT_ROOT / "models" / "llama_1b" / "refusal_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(refusal_vectors, out_dir / f"sequential_rv_dict_tier_{harmful_tier}.pt")
    torch.save(cohens_d, out_dir / f"sequential_cohens_d_tier_{harmful_tier}.pt")
    # old: k=7, v=3.7610558428271905

if __name__ == "__main__":
    main()
