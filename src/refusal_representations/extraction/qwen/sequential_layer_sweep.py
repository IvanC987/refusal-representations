from transformer_lens import HookedTransformer
import torch
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from collections import defaultdict

from refusal_representations.utils import tokenize_instructions_qwen_chat


def sweep_layers(model_id: str, device: str, json_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = HookedTransformer.from_pretrained_no_processing(model_id, trust_remote_code=True).to(device)
    model.eval()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert all([len(pair) == 2 for pair in data])

    harmful_layer_cache = defaultdict(list)
    safe_layer_cache = defaultdict(list)

    for harmful_prompt, safe_prompt in tqdm(data):
        harmful_token = tokenize_instructions_qwen_chat([harmful_prompt], tokenizer)
        safe_token = tokenize_instructions_qwen_chat([safe_prompt], tokenizer)

        with torch.no_grad():
            _, harmful_cache = model.run_with_cache(harmful_token, names_filter=lambda name: name.endswith("hook_resid_pre"))
            _, safe_cache = model.run_with_cache(safe_token, names_filter=lambda name: name.endswith("hook_resid_pre"))

        for l in range(model.cfg.n_layers):  # Can try mid/post, but pre is most meaningful to test for now
            harmful_layer_cache[l].append(harmful_cache[f"blocks.{l}.hook_resid_pre"][0, -1, :].detach().cpu())
            safe_layer_cache[l].append(safe_cache[f"blocks.{l}.hook_resid_pre"][0, -1, :].detach().cpu())

    # Now compute mean and find avg refusal vector
    refusal_vectors = {}
    for l in range(model.cfg.n_layers):
        harmful_layer_mean = torch.stack(harmful_layer_cache[l], dim=0).mean(dim=0)
        safe_layer_mean = torch.stack(safe_layer_cache[l], dim=0).mean(dim=0)
        refusal_vec = harmful_layer_mean - safe_layer_mean
        refusal_vectors[l] = refusal_vec / refusal_vec.norm()

    # Iterate through each layer, compute the projections of harmful/safe activations onto the refusal vector
    cohens_d = {}
    for l in range(model.cfg.n_layers):
        harmful_layer_residuals = harmful_layer_cache[l]
        safe_layer_residuals = safe_layer_cache[l]

        # (b, d_model) @ (d_model) = (b)
        harmful_projections = torch.stack(harmful_layer_residuals, dim=0) @ refusal_vectors[l]
        harmful_mean, harmful_var = harmful_projections.mean().item(), harmful_projections.var().item()
        safe_projections = torch.stack(safe_layer_residuals, dim=0) @ refusal_vectors[l]
        safe_mean, safe_var = safe_projections.mean().item(), safe_projections.var().item()

        d_l = (harmful_mean - safe_mean) / (0.5 * (harmful_var + safe_var)) ** 0.5
        cohens_d[l] = d_l

    return refusal_vectors, cohens_d


