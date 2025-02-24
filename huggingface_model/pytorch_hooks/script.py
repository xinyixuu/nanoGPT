import argparse
import os
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration
)
from PIL import Image
import requests

def parse_args():
    parser = argparse.ArgumentParser(description="Capture or swap PaLI/Gemma vision embeddings.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path (local or URL) to the image file.")
    parser.add_argument("--save_npy", type=str, default=None,
                        help="Where to save the captured embeddings as an .npy file.")
    parser.add_argument("--load_npy", type=str, default=None,
                        help="Path to .npy file of previously saved embeddings to inject.")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Optional Hugging Face token if needed to download the model.")
    parser.add_argument("--model_id", type=str, default="google/paligemma-3b-pt-224",
                        help="Which PaLI/Gemma model to use.")
    parser.add_argument("--prompt", type=str, default="caption en\n",
                        help="Text prompt to pass to the processor.")
    parser.add_argument("--max_new_tokens", type=int, default=40,
                        help="Max new tokens to generate.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Decide if image_path is a URL or local file
    if args.image_path.startswith("http://") or args.image_path.startswith("https://"):
        # Download from URL
        import requests
        image = Image.open(requests.get(args.image_path, stream=True).raw).convert("RGB")
    else:
        # Load from local path
        image = Image.open(args.image_path).convert("RGB")

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and processor
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id,
        token=args.hf_token  # or use_auth_token if older Transformers version
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)

    # We'll hook onto the final linear projection for multi-modal features
    chosen_submodule = model.multi_modal_projector.linear

    # We'll have global storage for captured embeddings
    captured_embeddings = []

    def capture_hook(mod, module_in, module_out):
        """
        If user wants to save embeddings, we capture them here.
        We assume shape [B, D] or [B, N, D].
        """
        # Move to CPU numpy for saving
        emb = module_out.detach().cpu().numpy()
        captured_embeddings.append(emb)
        return module_out  # pass through unmodified

    def load_hook(mod, module_in, module_out):
        """
        If user wants to load embeddings, we replace the output entirely.
        We must ensure shape matches our loaded .npy data.
        """
        # loaded_embeddings is a numpy array or a torch tensor
        # We'll do a direct replacement with the loaded data.
        # Return a torch tensor that matches module_out dtype/device/shape
        new_emb = torch.from_numpy(loaded_embeddings).to(module_out.device)
        # If the model processes a batch, we assume it's the same batch shape.
        # If shape is [B, D], we just return [B, D]. If [B, N, D], likewise.
        # So we check if we must broadcast or if the shapes match exactly.
        if new_emb.shape != module_out.shape:
            raise ValueError(
                f"Shape mismatch! new_emb {new_emb.shape} vs module_out {module_out.shape}"
            )
        return new_emb

    # Decide which hooks to apply
    capture_handle = None
    load_handle = None

    # If saving, register a capture hook
    if args.save_npy is not None:
        capture_handle = chosen_submodule.register_forward_hook(capture_hook)

    # If loading, load .npy file and register a load hook
    if args.load_npy is not None:
        global loaded_embeddings
        loaded_embeddings = np.load(args.load_npy)
        load_handle = chosen_submodule.register_forward_hook(load_hook)

    # Prepare inputs
    inputs = processor(
        text=args.prompt,
        images=image,
        return_tensors="pt",
    ).to(device)

    # Generate
    with torch.no_grad():
        prefix_len = inputs["input_ids"].shape[-1]
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False
        )
    new_tokens = generated_ids[0, prefix_len:]
    text_out = processor.decode(new_tokens, skip_special_tokens=True)

    # If saving, write out the embeddings
    if args.save_npy is not None:
        # We might have multiple captures if the model chunked the image internally,
        # typically you'll see just 1 forward pass for 1 image. We'll combine them:
        # We'll just assume it's the last item (or the only item).
        # Or you can store them all if you suspect more than one chunk.
        if len(captured_embeddings) == 0:
            print("Warning: no embeddings were captured. Possibly the hook wasn't triggered.")
        else:
            # For single image, typically captured_embeddings has length=1
            # But if the model does something fancy, it could be multiple.
            # We'll pick the final one (or merge them). 
            final_array = captured_embeddings[-1]
            np.save(args.save_npy, final_array)
            print(f"Saved embeddings to {args.save_npy} with shape {final_array.shape}.")

    # Remove hooks
    if capture_handle is not None:
        capture_handle.remove()
    if load_handle is not None:
        load_handle.remove()

    # Print generation result
    print(f"\n=== PROMPT ===\n{args.prompt}")
    print(f"=== IMAGE ===\n{args.image_path}")
    if args.load_npy:
        print(f"(Embeddings replaced by {args.load_npy})")
    print(f"=== MODEL OUTPUT ===\n{text_out}")

if __name__ == "__main__":
    main()

