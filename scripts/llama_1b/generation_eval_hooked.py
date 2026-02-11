import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from refusal_representations.ablation.residual_ablation import intervention_generation, create_hook
from refusal_representations.utils import tokenize_prompt
from refusal_representations.paths import PROJECT_ROOT


def main():
    refusal_dict_path = PROJECT_ROOT / "models" / "llama_1b" / "refusal_analysis" / "sequential_rv_dict.pt"
    refusal_dict = torch.load(refusal_dict_path, map_location=device)
    refusal_vec = refusal_dict[refusal_layer] / refusal_dict[refusal_layer].norm()

    print(f"Using {refusal_dict_path=}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    eos_token = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]
    model = HookedTransformer.from_pretrained_no_processing(model_id).to(device)
    model.eval()

    hook = create_hook(refusal_vec=refusal_vec, alpha=alpha)
    while True:
        prompt = input("\n\n>>> Prompt: ")
        input_tokens = tokenize_prompt([prompt], tokenizer).to(device)
        for i in range(2):
            fwd_hooks = [(f"blocks.{layer}.hook_resid_pre", hook) for layer in refusal_dict.keys()] if i == 1 else []
            print(f"\n[{'ABLATION' if i == 1 else 'BASELINE'}]")
            print("===========================")
            output_tokens = intervention_generation(model=model, input_tokens=input_tokens, max_tokens=max_tokens,
                                                    temperature=temperature, fwd_hooks=fwd_hooks,
                                                    eos_token=eos_token)
            print(tokenizer.decode(output_tokens[0].tolist()))  # Only have 1 batch
            print("\n===========================")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    alpha = 1.0
    max_tokens = 200
    temperature = 1.0
    refusal_layer = int(input("Enter layer: "))
    assert 0 <= refusal_layer <= 15  # 16 total layers in llama-3.2-1b-instruct

    main()
