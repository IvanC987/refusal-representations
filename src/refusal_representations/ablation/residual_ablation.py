import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer


def create_hook(refusal_vec: torch.Tensor, alpha: float):
    refusal_vec = refusal_vec / refusal_vec.norm()

    def modify_stream(stream, hook):
        # (batch, seq_len, d_model) @ (d_model) -> (batch, seq_len) scalars
        # Then (batch, seq_len, 1) * (1, 1, d_model) -> (batch, seq_len, d_model)
        proj = (stream @ refusal_vec).unsqueeze(2) * refusal_vec.reshape(1, 1, -1)
        stream = stream - (alpha * proj)
        return stream

    return modify_stream


@torch.no_grad()
def intervention_generation(model: HookedTransformer, input_tokens: torch.Tensor, max_tokens: int, temperature: float, fwd_hooks: list, eos_token: int):
    assert len(input_tokens.shape) == 2  # Should be (batch, seq_len)
    finished = torch.zeros(input_tokens.shape[0], dtype=torch.bool, device=input_tokens.device)

    tokens = input_tokens.clone()
    for _ in range(max_tokens):
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.argmax(probs, dim=-1)

        next_token = torch.where(finished, torch.full_like(next_token, eos_token), next_token)
        tokens = torch.cat((tokens, next_token.unsqueeze(1)), dim=-1)

        finished |= next_token.eq(eos_token)
        if finished.all():
            break

    return tokens

