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


def get_rv_dict():
    result = {}

    refusal_matrix = torch.stack([v for v in refusal_dict.values()], dim=0)
    refusal_matrix = refusal_matrix / refusal_matrix.norm(dim=1, keepdim=True)
    mean_vec = refusal_matrix.mean(dim=0)

    # First is layer-wise application, which is just the entire dict
    result["layerwise"] = refusal_dict

    # Next is mean rv
    result["mean"] = mean_vec

    # Then PC1
    refusal_matrix_centered = refusal_matrix - refusal_matrix.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(refusal_matrix_centered, q=3)
    pc1 = V[:, 0]
    pc1 = pc1 / pc1.norm()
    if torch.dot(pc1, mean_vec) < 0:
        pc1 = -pc1
    result["pca"] = pc1

    explained_var = (S ** 2) / (S ** 2).sum()
    print(f"{'*' * 20}\n\n\nPC1 Variance (Full): {explained_var}\n\n\n{'*' * 20}")

    # Use subset for layers [11, 16]
    refusal_matrix_subset = torch.stack([refusal_dict[k] for k in refusal_dict.keys() if 11 <= k <= 16], dim=0)
    assert refusal_matrix_subset.shape[0] == 6, "Should have 6 RVs"
    refusal_matrix_subset = refusal_matrix_subset / refusal_matrix_subset.norm(dim=1, keepdim=True)
    mean_vec_subset = refusal_matrix_subset.mean(dim=0)

    result["mean_subset_11_16"] = mean_vec_subset

    refusal_matrix_subset_centered = refusal_matrix_subset - refusal_matrix_subset.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(refusal_matrix_subset_centered, q=3)
    pc1 = V[:, 0]
    pc1 = pc1 / pc1.norm()
    if torch.dot(pc1, mean_vec_subset) < 0:
        pc1 = -pc1
    result["pca_subset_11_16"] = pc1

    explained_var = (S ** 2) / (S ** 2).sum()
    print(f"{'*' * 20}\n\n\nPC1 Variance (Subset): {explained_var}\n\n\n{'*' * 20}")

    return result


@torch.no_grad()
def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    for rv_type, rv_repr in aggregate_rv_dict.items():
        hooked_model = HookedTransformer.from_pretrained_no_processing(model_id, trust_remote_code=True).to(device)
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
        hf_transformer = hf_model.transformer

        if rv_type == "layerwise":
            for l, block in enumerate(hooked_model.blocks):
                if l == 0:
                    continue
                rv = rv_repr[l].to(device)
                rv = rv / rv.norm()
                block.attn.W_O.copy_(orthogonalize_weights(block.attn.W_O, rv, alpha))
                block.mlp.W_out.copy_(orthogonalize_weights(block.mlp.W_out, rv, alpha))
        else:
            rv = rv_repr.to(device)
            rv = rv / rv.norm()
            for block in hooked_model.blocks:
                block.attn.W_O.copy_(orthogonalize_weights(block.attn.W_O, rv, alpha))
                block.mlp.W_out.copy_(orthogonalize_weights(block.mlp.W_out, rv, alpha))

        hooked_state_dict = hooked_model.state_dict()

        for l in range(hooked_model.cfg.n_layers):
            n_heads, h_dim, d_model = hooked_state_dict[f"blocks.{l}.attn.W_O"].shape
            hf_transformer.h[l].attn.c_proj.weight = torch.nn.Parameter((hooked_state_dict[f"blocks.{l}.attn.W_O"]
                                                                             .reshape(n_heads * h_dim, d_model))
                                                                             .permute(1, 0).contiguous())
            hf_transformer.h[l].mlp.c_proj.weight = torch.nn.Parameter(
                torch.transpose(hooked_state_dict[f"blocks.{l}.mlp.W_out"], 0, 1).contiguous()
            )

        output_dir = save_dir / f"{rv_type}_alpha_{int(alpha*100)}"
        hf_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"{'*' * 20}\n\n\nSaved to: {output_dir}\n\n\n{'*' * 20}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen-1_8B-chat"
    alpha = 1.0
    save_dir = PROJECT_ROOT / "models" / "qwen" / "orthogonalized"

    refusal_dict_path = PROJECT_ROOT / "models" / "qwen" / "refusal_analysis" / "sequential_rv_dict.pt"
    refusal_dict = torch.load(refusal_dict_path, map_location=device)
    del refusal_dict[0]  # Remove nan RV
    aggregate_rv_dict = get_rv_dict()
    main()
