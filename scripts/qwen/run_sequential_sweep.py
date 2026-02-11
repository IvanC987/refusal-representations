import torch

from refusal_representations.extraction.qwen.sequential_layer_sweep import sweep_layers
from refusal_representations.paths import PROJECT_ROOT


"""
Loaded pretrained model Qwen/Qwen-1_8B-chat into HookedTransformer
Moving model to device:  cuda
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:01<00:00, 16.15it/s]
k=0, v=nan
k=19, v=5.330492380337148
k=23, v=5.325045039327355
k=17, v=5.322999368822017
k=22, v=5.279259661950217
k=20, v=5.2779695925780565
k=21, v=5.277817167908854
k=18, v=5.258214149760252
k=16, v=5.159259846415958
k=15, v=5.128443292972094
k=14, v=4.580003133134045
k=13, v=4.279398781608142
k=12, v=2.9767086042649304
k=11, v=2.8496165431231293
k=10, v=1.7310830599017855
k=9, v=1.6203216284296502
k=7, v=1.6032286503911
k=8, v=1.4559586045142656
k=4, v=1.3458120098042887
k=6, v=1.3042419412775716
k=5, v=1.266141359641412
k=2, v=1.1795877101144758
k=3, v=1.171018990809439
k=1, v=0.6205930512503793
"""


def main():
    torch.random.manual_seed(89)
    torch.cuda.manual_seed(89)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen-1_8B-chat"
    json_path = PROJECT_ROOT / "data" / "generated_pairs.json"

    refusal_vectors, cohens_d = sweep_layers(model_id=model_id, device=device, json_path=json_path)

    sorted_d = sorted(cohens_d.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_d:
        print(f"{k=}, {v=}")

    out_dir = PROJECT_ROOT / "models" / "qwen" / "refusal_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(refusal_vectors, out_dir / "sequential_rv_dict.pt")
    torch.save(cohens_d, out_dir / "sequential_cohens_d.pt")


if __name__ == "__main__":
    main()
