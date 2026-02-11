import matplotlib.pyplot as plt
import numpy as np
import torch

from refusal_representations.paths import PROJECT_ROOT


path1 = PROJECT_ROOT / "models" / "llama_8b" / "refusal_analysis" / "sequential_rv_dict_tier_2.pt"

dict1 = torch.load(path1, map_location="cpu")
dict2 = torch.load(path1, map_location="cpu")


layers = sorted(dict1.keys())
N = len(layers)
sim_matrix = np.zeros((N, N))

# Compute cosine similarities
for i, li in enumerate(layers):
    v1 = dict1[li].flatten()
    for j, lj in enumerate(layers):
        v2 = dict2[lj].flatten()
        sim_matrix[i, j] = torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(sim_matrix)
plt.colorbar(label="Cosine Similarity")
plt.xticks(range(N), layers)
plt.yticks(range(N), layers)
plt.title("Layer-wise Cosine Similarity Heatmap")
plt.tight_layout()
plt.show()