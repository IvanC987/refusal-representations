import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from refusal_representations.ablation.residual_ablation import intervention_generation, create_hook
from refusal_representations.utils import tokenize_prompt
from refusal_representations.paths import PROJECT_ROOT


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    max_tokens = 250
    temperature = 1.0
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    eos_token_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]

    model = HookedTransformer.from_pretrained_no_processing(model_id).to(device)
    model.eval()

    harmful_test_path = PROJECT_ROOT / "data" / "harmful_100.json"
    with open(harmful_test_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # A json list of string prompts

    sequential_dict_path = PROJECT_ROOT / "models" / "llama_1b" / "refusal_analysis" / "sequential_rv_dict_tier_1.pt"
    rv_dict = torch.load(sequential_dict_path, map_location=device)

    result = []
    for layer_num, refusal_vector in rv_dict.items():
        print(f"{layer_num=}")
        hook = create_hook(refusal_vec=refusal_vector, alpha=1.0)
        fwd_hooks = [(f"blocks.{layer}.hook_resid_pre", hook) for layer in rv_dict.keys()]
        for prompt_idx in tqdm(range(0, len(data), batch_size)):
            prompt_list = data[prompt_idx: prompt_idx + batch_size]
            input_tokens = tokenize_prompt(prompt_list, tokenizer).to(device)
            output_tokens = intervention_generation(model=model, input_tokens=input_tokens,
                                                    max_tokens=max_tokens,
                                                    temperature=temperature, fwd_hooks=fwd_hooks,
                                                    eos_token=eos_token_id)

            for batch_idx, batch in enumerate(output_tokens):
                batch = batch[len(input_tokens[batch_idx]):]  # Remove prompt portion
                eos_idx = len(batch)
                while eos_idx > 0 and batch[eos_idx - 1] == eos_token_id:  # Remove trailing EOS
                    eos_idx -= 1
                batch = batch[:eos_idx]
                decoded_str = tokenizer.decode(batch.tolist())
                result.append({
                    "layer": layer_num,
                    "alpha": 1.0,
                    "prompt": prompt_list[batch_idx],
                    "response": decoded_str,
                })

    out_dir = PROJECT_ROOT / "experiments" / "llama_outputs_1b"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"sequential_rollouts.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
