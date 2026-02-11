import torch

from refusal_representations.extraction.llama.batched_layer_sweep import sweep_layers
from refusal_representations.paths import PROJECT_ROOT


"""
Loaded pretrained model meta-llama/Llama-3.2-1B-Instruct into HookedTransformer
Moving model to device:  cuda
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:15<00:00,  2.13it/s]
k=0, v=nan
k=15, v=5.518659552062471
k=14, v=5.399649788840797
k=13, v=5.234913975380837
k=12, v=5.146792026459497
k=11, v=4.984390969348963
k=10, v=4.888322697385382
k=9, v=4.711758391543621
k=8, v=3.6648126365020706
k=7, v=3.0596504174309493
k=6, v=2.1118608096358096
k=4, v=1.8622148920093002
k=5, v=1.7908978138350915
k=2, v=1.0023201113714468
k=3, v=0.8460697025149106
k=1, v=0.5534580393063488
"""


def main():
    torch.random.manual_seed(89)
    torch.cuda.manual_seed(89)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    json_path = PROJECT_ROOT / "data" / "generated_pairs.json"
    batch_size = 32

    refusal_vectors, cohens_d = sweep_layers(model_id=model_id, device=device, json_path=json_path, batch_size=batch_size)

    sorted_d = sorted(cohens_d.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_d:
        print(f"{k=}, {v=}")

    out_dir = PROJECT_ROOT / "models" / "llama_1b" / "refusal_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(refusal_vectors, out_dir / "batched_rv_dict.pt")
    torch.save(cohens_d, out_dir / "batched_cohens_d.pt")


if __name__ == "__main__":
    main()
