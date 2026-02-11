import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from refusal_representations.utils import tokenize_instructions_qwen_chat
from refusal_representations.paths import PROJECT_ROOT


@torch.no_grad()
def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<|extra_0|>"
    eos_token_id = tokenizer.encode(tokenizer.eos_token)[0]

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()

    while True:
        try:
            prompt = input("\n\n>>> Prompt: ").strip()
            if not prompt:
                continue

            input_tokens = tokenize_instructions_qwen_chat([prompt], tokenizer).to(device)

            for _ in range(max_tokens):
                logits = model(input_tokens).logits
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.argmax(probs, dim=-1)

                input_tokens = torch.cat((input_tokens, next_token.unsqueeze(1)), dim=-1)

                if next_token.item() == eos_token_id:
                    break

                print(tokenizer.decode([next_token.item()]), end="", flush=True)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_tokens = 200
    temperature = 1.0

    model_dir = PROJECT_ROOT / "models" / "qwen" / "orthogonalized"

    subdirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])
    assert len(subdirs) > 0, "No orthogonalized models found"

    print("Available orthogonalized models:\n")
    for i, d in enumerate(subdirs):
        print(f"[{i}] {d.name}")

    idx = int(input("\nSelect model index: ").strip())
    model_id = subdirs[idx]
    # model_id = "Qwen/Qwen-1_8B-chat"

    main()
