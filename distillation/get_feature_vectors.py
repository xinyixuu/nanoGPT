import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Disable gradient calculation to reduce memory overhead
torch.set_grad_enabled(False)

# # Load model
# model = HookedTransformer.from_pretrained("gemma-2b-it", device="cuda")

# Load SAE
sae, cfg, sparsity = SAE.from_pretrained(
    "gemma-2b-it-res-jb",
    "blocks.12.hook_resid_post",
)

# Get all decoded vectors (W_dec)
all_vectors = sae.W_dec.cpu().numpy()  # Move to CPU and convert to numpy array

# Save to .npy
np.save("all_W_dec_vectors.npy", all_vectors)

print(f"Saved W_dec matrix with shape {all_vectors.shape} to all_W_dec_vectors.npy")

