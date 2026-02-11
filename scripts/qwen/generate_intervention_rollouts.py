import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from refusal_representations.ablation.residual_ablation import intervention_generation, create_hook
from refusal_representations.utils import tokenize_instructions_qwen_chat
from refusal_representations.paths import PROJECT_ROOT


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    max_tokens = 250
    temperature = 1.0
    model_id = "Qwen/Qwen-1_8B-chat"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<|extra_0|>"
    eos_token_id = tokenizer.encode(tokenizer.eos_token)[0]

    model = HookedTransformer.from_pretrained_no_processing(model_id, trust_remote_code=True).to(device)
    model.eval()

    harmful_test_path = PROJECT_ROOT / "data" / "harmful_100.json"
    with open(harmful_test_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # A json list of string prompts

    rv_dict_paths = {
        "sequential": PROJECT_ROOT / "models" / "qwen" / "refusal_analysis" / "sequential_rv_dict.pt",
        "batched": PROJECT_ROOT / "models" / "qwen" / "refusal_analysis" / "batched_rv_dict.pt",
    }

    test_method = input("Test method ('sequential', 'batched'): ")
    assert test_method in ['sequential', 'batched']

    result = []
    rv_dict = torch.load(rv_dict_paths[test_method], map_location=device)
    for layer_num, refusal_vector in rv_dict.items():
        print(f"{layer_num=}")
        for alpha in [0.75, 1.00, 1.25]:
            hook = create_hook(refusal_vec=refusal_vector, alpha=alpha)
            fwd_hooks = [(f"blocks.{layer}.hook_resid_pre", hook) for layer in rv_dict.keys()]
            for prompt_idx in tqdm(range(0, len(data), batch_size)):
                prompt_list = data[prompt_idx: prompt_idx + batch_size]
                input_tokens = tokenize_instructions_qwen_chat(prompt_list, tokenizer).to(device)
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
                        "alpha": alpha,
                        "prompt": prompt_list[batch_idx],
                        "response": decoded_str,
                    })

    out_dir = PROJECT_ROOT / "experiments" / "qwen_seq_vs_batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{test_method}_rollouts.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
