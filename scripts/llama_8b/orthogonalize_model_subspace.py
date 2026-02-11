import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

from refusal_representations.paths import PROJECT_ROOT


def orthogonalize_weights(weight_matrix: torch.Tensor, alpha: float):
    # (..., d_model) @ (d_model, k) -> (..., k) @ (k, d_model) -> (..., d_model)
    proj = weight_matrix @ refusal_subspace @ refusal_subspace.T
    return weight_matrix - (alpha * proj)


@torch.no_grad()
def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for alpha in [0.50, 1.0, 1.50]:
        hooked_model = HookedTransformer.from_pretrained_no_processing(model_id, torch_dtype=torch.bfloat16).to(device)
        hf_transformer = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        hf_model = hf_transformer.model


        for block in hooked_model.blocks:
            block.attn.W_O.copy_(orthogonalize_weights(block.attn.W_O, alpha))
            block.mlp.W_out.copy_(orthogonalize_weights(block.mlp.W_out, alpha))

        hooked_state_dict = hooked_model.state_dict()

        for l in range(hooked_model.cfg.n_layers):
            n_heads, h_dim, d_model = hooked_state_dict[f"blocks.{l}.attn.W_O"].shape
            hf_model.layers[l].self_attn.o_proj.weight = torch.nn.Parameter((hooked_state_dict[f"blocks.{l}.attn.W_O"]
                                                                             .reshape(n_heads * h_dim, d_model))
                                                                             .permute(1, 0).contiguous())
            hf_model.layers[l].mlp.down_proj.weight = torch.nn.Parameter(
                torch.transpose(hooked_state_dict[f"blocks.{l}.mlp.W_out"], 0, 1).contiguous()
            )

        output_dir = save_dir / f"sequential_alpha_{int(alpha*100)}_tier_{harm_tier}"
        hf_transformer.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"{'*' * 20}\n\n\nSaved to: {output_dir}\n\n\n{'*' * 20}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    save_dir = PROJECT_ROOT / "models" / "llama_8b" / "orthogonalized"

    # Respective compliance scores: 0.795, 0.570, 0.335, 0.300, 0.100
    selected_layers = [9, 11, 10, 12, 8]
    harm_tier = 2
    refusal_dict_path = PROJECT_ROOT / "models" / "llama_8b" / "refusal_analysis" / f"sequential_rv_dict_tier_{harm_tier}.pt"
    refusal_dict = torch.load(refusal_dict_path, map_location=device)

    V = torch.stack([refusal_dict[l] for l in selected_layers], dim=1)
    V = V / V.norm(dim=0, keepdim=True)
    refusal_subspace, _ = torch.linalg.qr(V)  # (d_model, k) tensor
    refusal_subspace = refusal_subspace.to(device).to(torch.bfloat16)

    main()
