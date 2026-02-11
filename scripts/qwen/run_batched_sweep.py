import torch

from refusal_representations.extraction.qwen.batched_layer_sweep import sweep_layers
from refusal_representations.paths import PROJECT_ROOT


"""
Loaded pretrained model Qwen/Qwen-1_8B-chat into HookedTransformer
Moving model to device:  cuda
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:11<00:00,  2.78it/s]
k=0, v=nan
k=20, v=3.573412946565351
k=22, v=3.5675764798782597
k=19, v=3.5611888625226285
k=21, v=3.549885067393972
k=18, v=3.458774850469483
k=17, v=3.4199112498895543
k=23, v=3.215188025833442
k=16, v=3.096828385921432
k=15, v=2.744500849819607
k=14, v=2.0129602382810443
k=13, v=1.866260175578753
k=6, v=1.0839934367418487
k=3, v=0.971071785090842
k=2, v=0.8979985589299458
k=4, v=0.897830829924786
k=12, v=0.8854086594828383
k=5, v=0.8438792607863654
k=11, v=0.6444382230432225
k=1, v=0.4902140885870263
k=10, v=0.31350419361938153
k=9, v=0.2325895632415843
k=7, v=0.19785115364311784
k=8, v=0.178996454492501
"""


def main():
    torch.random.manual_seed(89)
    torch.cuda.manual_seed(89)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen-1_8B-chat"
    json_path = PROJECT_ROOT / "data" / "generated_pairs.json"
    batch_size = 32

    refusal_vectors, cohens_d = sweep_layers(model_id=model_id, device=device, json_path=json_path, batch_size=batch_size)

    sorted_d = sorted(cohens_d.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_d:
        print(f"{k=}, {v=}")

    out_dir = PROJECT_ROOT / "models" / "qwen" / "refusal_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(refusal_vectors, out_dir / "batched_rv_dict.pt")
    torch.save(cohens_d, out_dir / "batched_cohens_d.pt")


if __name__ == "__main__":
    main()
