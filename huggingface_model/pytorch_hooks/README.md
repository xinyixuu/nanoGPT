# Swapping PaLI/Gemma Vision Embeddings

This repository demonstrates how to **capture** and **inject (swap)** the final vision embeddings of [PaLI/Gemma](https://huggingface.co/google/paligemma-3b-pt-224) in order to experiment with image-driven prompts. Specifically, we:

- **Capture** the final embeddings for one image and save them into a NumPy file (`.npy`).
- **Load** and inject those embeddings for a *different* image during text generation. This effectively makes the model treat the second image as though it has the representation of the first.

We use images of a **cat** and a **dog** from [nickmuchi/vit-finetuned-cats-dogs](https://huggingface.co/nickmuchi/vit-finetuned-cats-dogs) for our example.

---

## Files

1. **`script.py`**  
   A Python script that:
   - Loads a PaLI/Gemma model (`google/paligemma-3b-pt-224` by default).
   - Captures or injects the **final vision embeddings** via forward hooks at `model.multi_modal_projector.linear`.
   - Can save those embeddings to `.npy` or load them from `.npy`.
   - Finally, generates text describing the (real or swapped) image.

2. **`demo.sh`**  
   A shell script that:
   - Runs `script.py` in two scenarios:
     1. Saves the cat embeddings → loads them into the dog image.  
     2. Saves the dog embeddings → loads them into the cat image.  
   - Demonstrates how to call `script.py` with `--save_npy` and `--load_npy`.

3. **`view_embeddings.py`** (example for heatmap visualization, see below):
   - A minimal script that loads a `.npy` file of embeddings and visualizes them as a Seaborn heatmap.

---

## Quick Start

### 1. Clone this repository (or copy the files)

```bash
git clone <your-repo-url>  # or just download script.py and demo.sh
cd your-repo
```

### 2. Create and activate a Python environment

```bash
conda create -n paligemma_test python=3.10
conda activate paligemma_test
pip install torch transformers pillow requests seaborn matplotlib
```

### 3. Run the demo script

We provide [`demo.sh`](./demo.sh) as a convenience:

```bash
chmod +x demo.sh
./demo.sh
```

This will:

1. **Capture** the cat image embeddings into `cat_emb.npy` and show the model’s caption for the cat.  
2. **Inject** the cat embeddings during the dog image’s forward pass (thus the dog image is “seen” as if it were a cat).  
3. **Capture** the dog image embeddings into `dog_emb.npy` and show the model’s caption for the dog.  
4. **Inject** the dog embeddings into the cat image’s forward pass (thus the cat image is “seen” as if it were a dog).

You’ll see console outputs with the model’s generated text.

---

## How It Works

- **`script.py`** has two main modes, controlled by **command-line arguments**:
  
  | Argument        | Description                                                                                     |
  |-----------------|-------------------------------------------------------------------------------------------------|
  | `--image_path`  | Path to the image (local file or URL).                                                         |
  | `--save_npy`    | If provided, captures the final vision embedding into the specified `.npy` file.               |
  | `--load_npy`    | If provided, loads embeddings from this file and **injects** them in place of the real image.  |
  | `--model_id`    | Hugging Face model identifier (defaults to `google/paligemma-3b-pt-224`).                      |
  | `--prompt`      | The text prompt for generation. (Defaults to `"caption en\n"`)                                 |
  | `--max_new_tokens` | Maximum new tokens to generate in the text output.                                          |

- Internally, `script.py`:
  1. Hooks onto **`model.multi_modal_projector.linear`**—the final linear projection that transforms vision features (1152-d) to language features (2048-d).  
  2. If **`--save_npy`** is given, it captures this tensor and saves it to `.npy`.  
  3. If **`--load_npy`** is given, it replaces the real output with the loaded embeddings.  
  4. Finally, it calls `model.generate(...)` to produce a caption.

---

## The `.npy` Files (Stored Embeddings)

- The **shape** of the stored embeddings typically will be `[1, 2048]` if the image is processed as a single feature vector (or `[1, N, 2048]` if the model keeps multiple tokens for the image).
- `script.py` verifies the shape at runtime to ensure you don’t load mismatching embeddings.
- Each `.npy` file is simply a NumPy array containing the floating-point data.

---

### Usage

```bash
python view_embeddings.py --npy_file cat_emb.npy --save_png cat_emb_heatmap.png
```

This script will load your `.npy` file, then either display or save a heatmap. For a `[1, 2048]` shape, you’ll see a **1×2048** matrix. If you have `[1, 256, 2048]`, you’ll see a **256×2048** matrix (one row for each spatial token).

---

## Credits

- **Images** (cat.jpg and dog.jpg) courtesy of [nickmuchi/vit-finetuned-cats-dogs](https://huggingface.co/nickmuchi/vit-finetuned-cats-dogs).  
- **Model**: [PaLI/Gemma (3B)](https://huggingface.co/google/paligemma-3b-pt-224).  

