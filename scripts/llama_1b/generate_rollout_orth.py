import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from refusal_representations.utils import tokenize_prompt
from refusal_representations.paths import PROJECT_ROOT


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_tokens = 250
    temperature = 1.0
    model_id = str(PROJECT_ROOT / "models" / "llama_1b" / "orthogonalized" / "sequential_alpha_100_tier_2_7_10")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    eos_token_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    harmful_test_path = PROJECT_ROOT / "data" / "harmful_100.json"
    with open(harmful_test_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # A json list of string prompts

    result = []
    for prompt in tqdm(data):
        cache = []
        input_tokens = tokenize_prompt([prompt], tokenizer).to(device)

        for _ in range(max_tokens):
            logits = model(input_tokens).logits
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.argmax(probs, dim=-1)

            input_tokens = torch.cat((input_tokens, next_token.unsqueeze(1)), dim=-1)

            if next_token.item() == eos_token_id:
                break

            cache.append(tokenizer.decode([next_token.item()]))

        result.append({
            "layer": -1,
            "alpha": 1.0,
            "prompt": prompt,
            "response": "".join(cache),
        })

    out_dir = PROJECT_ROOT / "experiments" / "llama_outputs_1b"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"sequential_alpha_100_tier_2_7_10_rollouts.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
