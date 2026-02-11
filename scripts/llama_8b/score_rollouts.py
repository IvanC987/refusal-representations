import openai
import os
import json
from tqdm import tqdm

from refusal_representations.evaluation.llm_judge import get_evaluations
from refusal_representations.paths import PROJECT_ROOT


def main():
    rollout_path = PROJECT_ROOT / "experiments" / "llama_outputs_8b" / f"sequential_rollouts.json"
    with open(rollout_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = []
    for idx in tqdm(range(0, len(data), BATCH_SIZE)):
        data_slice = data[idx: idx + BATCH_SIZE]
        responses = [e["response"] for e in data_slice]
        evaluations = get_evaluations(client=CLIENT, model=MODEL, responses=responses)

        for i in range(len(data_slice)):
            dict_ = data_slice[i]
            dict_["judge_metrics"] = {
                "compliance": evaluations[i][0],
                "coherence": evaluations[i][1]
            }
            result.append(dict_)

    output_path = PROJECT_ROOT / "experiments" / "llama_outputs_8b" / f"sequential_rollouts_evaluated.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    BATCH_SIZE = 5
    MODEL = "deepseek-chat"
    CLIENT = openai.Client(base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))

    main()
