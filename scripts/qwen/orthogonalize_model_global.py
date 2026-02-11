import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

from refusal_representations.paths import PROJECT_ROOT


def orthogonalize_weights(weight_matrix: torch.Tensor, refusal_vec: torch.Tensor, alpha: float):
    assert len(refusal_vec.shape) == 1
    assert weight_matrix.shape[-1] == refusal_vec.shape[0]

    # (..., d_model) @ (d_model, 1) -> (..., 1) * (d_model) -> (..., d_model)
    proj = (weight_matrix @ refusal_vec.unsqueeze(1)) * refusal_vec
    return weight_matrix - (alpha * proj)


@torch.no_grad()
def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    for (refusal_layer, alpha) in intervention_configs:
        hooked_model = HookedTransformer.from_pretrained_no_processing(model_id, trust_remote_code=True).to(device)
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
        hf_transformer = hf_model.transformer

        refusal_vec = (refusal_dict[refusal_layer] / refusal_dict[refusal_layer].norm()).to(device)
        for block in hooked_model.blocks:
            block.attn.W_O.copy_(orthogonalize_weights(block.attn.W_O, refusal_vec, alpha))
            block.mlp.W_out.copy_(orthogonalize_weights(block.mlp.W_out, refusal_vec, alpha))

        hooked_state_dict = hooked_model.state_dict()

        for l in range(hooked_model.cfg.n_layers):
            n_heads, h_dim, d_model = hooked_state_dict[f"blocks.{l}.attn.W_O"].shape
            hf_transformer.h[l].attn.c_proj.weight = torch.nn.Parameter((hooked_state_dict[f"blocks.{l}.attn.W_O"]
                                                                             .reshape(n_heads * h_dim, d_model))
                                                                             .permute(1, 0).contiguous())
            hf_transformer.h[l].mlp.c_proj.weight = torch.nn.Parameter(
                torch.transpose(hooked_state_dict[f"blocks.{l}.mlp.W_out"], 0, 1).contiguous()
            )

        output_dir = save_dir / f"{method}_layer_{refusal_layer}_alpha_{int(alpha*100)}"
        hf_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"{'*' * 20}\n\n\nSaved to: {output_dir}\n\n\n{'*' * 20}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen-1_8B-chat"
    save_dir = PROJECT_ROOT / "models" / "qwen" / "orthogonalized"

    # (layer_num, alpha) per tuple
    # intervention_configs = [(15, 0.75), (14, 0.75), (14, 1.0)]
    # method = "sequential"

    intervention_configs = [(15, 1.0), (15, 1.25), (16, 1.25)]
    method = "batched"
    refusal_dict_path = PROJECT_ROOT / "models" / "qwen" / "refusal_analysis" / f"{method}_rv_dict.pt"
    refusal_dict = torch.load(refusal_dict_path, map_location=device)

    assert input(f"{intervention_configs=}\n{method=}\nConfirm [Y/N]: ").lower() == "y"
    main()
