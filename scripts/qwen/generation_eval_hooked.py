import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from refusal_representations.ablation.residual_ablation import intervention_generation, create_hook
from refusal_representations.utils import tokenize_instructions_qwen_chat
from refusal_representations.paths import PROJECT_ROOT


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen-1_8B-chat"

    alpha = 1.0
    max_tokens = 200
    temperature = 1.0
    refusal_layer = 14

    refusal_dict_path = PROJECT_ROOT / "models" / "qwen" / "refusal_analysis" / "sequential_rv_dict.pt"
    refusal_dict = torch.load(refusal_dict_path, map_location=device)
    refusal_vec = refusal_dict[refusal_layer] / refusal_dict[refusal_layer].norm()

    print(f"Using {refusal_dict_path=}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<|extra_0|>"

    model = HookedTransformer.from_pretrained_no_processing(model_id, trust_remote_code=True).to(device)
    model.eval()

    hook = create_hook(refusal_vec=refusal_vec, alpha=alpha)
    while True:
        prompt = input("\n\n>>> Prompt: ")
        input_tokens = tokenize_instructions_qwen_chat([prompt], tokenizer).to(device)
        for i in range(2):
            fwd_hooks = [(f"blocks.{layer}.hook_resid_pre", hook) for layer in refusal_dict.keys()] if i == 1 else []
            print(f"\n[{'ABLATION' if i == 1 else 'BASELINE'}]")
            print("===========================")
            output_tokens = intervention_generation(model=model, input_tokens=input_tokens, max_tokens=max_tokens,
                                                    temperature=temperature, fwd_hooks=fwd_hooks,
                                                    eos_token=tokenizer.encode(tokenizer.eos_token)[0])
            print(tokenizer.decode(output_tokens[0].tolist()))  # Only have 1 batch
            print("\n===========================")


if __name__ == "__main__":
    main()
