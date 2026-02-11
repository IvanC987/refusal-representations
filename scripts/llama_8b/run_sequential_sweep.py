import torch

from refusal_representations.extraction.llama.sequential_layer_sweep import sweep_layers
from refusal_representations.paths import PROJECT_ROOT


"""
Tier 2

Loaded pretrained model meta-llama/Llama-3.1-8B-Instruct into HookedTransformer
Moving model to device:  cuda
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [04:40<00:00,  3.56it/s]
k=0, v=nan
k=31, v=6.760859550047855
k=22, v=6.708834698974862
k=23, v=6.666791142215067
k=27, v=6.650096194640932
k=29, v=6.6349651130944896
k=28, v=6.632571845993968
k=24, v=6.629882058968319
k=21, v=6.628668414205586
k=25, v=6.624671436383718
k=30, v=6.6165533422343215
k=26, v=6.551225439944374
k=20, v=6.538492041104092
k=17, v=6.398113707274071
k=19, v=6.381339294615385
k=18, v=6.344357479697538
k=16, v=6.231160085065611
k=15, v=6.038467883462133
k=14, v=5.59179903770748
k=13, v=5.209518579950708
k=12, v=4.597523143634803
k=11, v=3.8964400825027523
k=10, v=3.2808800657769575
k=9, v=2.951464264072352
k=8, v=2.569875913683625
k=7, v=2.3251877754091517
k=6, v=1.7230723184810606
k=5, v=1.7102411832691415
k=4, v=1.380246148764945
k=3, v=0.8892285747903269
k=2, v=0.8551846804808888
k=1, v=0.81822303904717
"""


def main():
    torch.random.manual_seed(89)
    torch.cuda.manual_seed(89)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    harmful_tier = 2
    json_path = PROJECT_ROOT / "data" / f"generated_pairs_tier_{harmful_tier}.json"

    refusal_vectors, cohens_d = sweep_layers(model_id=model_id, device=device, json_path=json_path)

    sorted_d = sorted(cohens_d.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_d:
        print(f"{k=}, {v=}")

    out_dir = PROJECT_ROOT / "models" / "llama_8b" / "refusal_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(refusal_vectors, out_dir / f"sequential_rv_dict_tier_{harmful_tier}.pt")
    torch.save(cohens_d, out_dir / f"sequential_cohens_d_tier_{harmful_tier}.pt")


if __name__ == "__main__":
    main()
