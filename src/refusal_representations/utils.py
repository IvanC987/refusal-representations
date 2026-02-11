

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""


def tokenize_instructions_qwen_chat(instructions: list[str], tokenizer):
    prompts = [QWEN_CHAT_TEMPLATE.format(instruction=instruction) for instruction in instructions]
    if len(prompts) == 1:
        return tokenizer(prompts, return_tensors="pt").input_ids

    return tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").input_ids


def tokenize_prompt(prompt: list[str], tokenizer):
    message = [[{"role": "user", "content": p}] for p in prompt]
    return tokenizer.apply_chat_template(message, return_tensors="pt", padding=True, truncation=True, add_generation_prompt=True)


