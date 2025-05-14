import torch
import numpy as np

def direct_init(config):
    """
    Minimal example of 'direct' hypercube initialization.
    For demonstration, we'll just create 2^n_embd corners
    and then pick the first `vocab_size`.
    """
    # extract config variables
    vocab_size=config.vocab_size
    n_embd=config.n_embd
    scale=config.init_scale

    n_corners = 2 ** n_embd
    n_corners = min(n_corners, vocab_size)

    if vocab_size > n_corners:
        raise ValueError(
            f"Not enough corners (2^{n_embd}={n_corners}) for vocab_size={vocab_size} in 'direct' mode."
        )
    corners = torch.zeros((n_corners, n_embd))
    for i in range(n_corners):
        for d in range(n_embd):
            corners[i, d] = (i >> d) & 1
    return corners[:vocab_size, :] * scale

def one_hot_init(config):
    """
    Create a one-hot embedding matrix of shape [vocab_size, n_embd].
    We assert n_embd >= vocab_size so that each row can have exactly one 1
    in a distinct column.
    """

    # extract config variables
    vocab_size=config.vocab_size
    n_embd=config.n_embd
    scale=config.init_scale

    # init
    if n_embd < vocab_size:
        raise ValueError("For 'one-hot' init, n_embd must be >= vocab_size.")
    weight = torch.zeros((vocab_size, n_embd))
    for i in range(vocab_size):
        weight[i, i] = scale
    return weight

def numpy_import_init(config):
    """
    Loads a pre-trained embedding matrix from a NumPy file.
    The file_path will be handled by the GPT class.
    """

    # extract config variables
    vocab_size=config.vocab_size
    n_embd=config.n_embd
    scale=config.init_scale
    file_path = config.init_wte_npy

    try:
        embedding_data = np.load(file_path)
        embedding_tensor = torch.from_numpy(embedding_data).float()
        if embedding_tensor.shape != (vocab_size, n_embd):
            raise ValueError(f"Numpy embedding shape {embedding_tensor.shape} does not match expected shape ({vocab_size}, {n_embd}).")
        return embedding_tensor * scale
    except FileNotFoundError:
        raise FileNotFoundError(f"NumPy embedding file not found at {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading NumPy embedding file: {e}")

init_dictionary = {
    "gaussian": None,    # fall back to the default Gaussian in model.py
    "hypercube": direct_init,
    "onehot": one_hot_init,
    "numpy_import": numpy_import_init, # New entry
}
