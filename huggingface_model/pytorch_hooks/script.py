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
    parser = argparse.ArgumentParser(description="Capture or swap PaLI/Gemma vision embeddings with optional quantization.")
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
    parser.add_argument("--quantize", type=str, choices=["int16", "int8", "int4", "int2", None],
                        default="int16", help="Quantization type for embeddings (default: None).")
    return parser.parse_args()

def fake_quantize(embeddings, quant_type):
    """
    Apply fake quantization to embeddings.
    """
    if quant_type is None:
        return embeddings

    min_val, max_val = embeddings.min(), embeddings.max()
    if quant_type == "int16":
        qmin, qmax = -32768, 32767
    elif quant_type == "int8":
        qmin, qmax = -128, 127
    elif quant_type == "int4":
        qmin, qmax = -8, 7
    elif quant_type == "int2":
        qmin, qmax = -2, 1
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")

    # Scale and shift to fit within quantization range
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    quantized = np.round(embeddings / scale + zero_point).clip(qmin, qmax).astype(np.int32)

    return quantized, scale, zero_point

def fake_dequantize(quantized, scale, zero_point):
    """
    Convert fake quantized values back to float.
    """
    return ((quantized - zero_point) * scale).astype(np.float32)  # Ensure float32

def main():
    args = parse_args()

    # Load image (URL or local)
    if args.image_path.startswith("http://") or args.image_path.startswith("https://"):
        image = Image.open(requests.get(args.image_path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(args.image_path).convert("RGB")

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and processor
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id,
        token=args.hf_token
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)

    # Identify the target layer
    chosen_submodule = model.multi_modal_projector.linear
    captured_embeddings = []

    def capture_hook(mod, module_in, module_out):
        """
        Capture the INPUT to the linear layer.
        """
        emb = module_in[0].detach().cpu().numpy()  # Capture first input tensor
        captured_embeddings.append(emb)
        return module_out  # Pass through unmodified


    def load_hook(mod, module_in):
        """
        Inject saved embeddings into the input of the linear layer.
        """
        global loaded_embeddings

        # If loaded_embeddings is already a tensor, avoid re-conversion
        if isinstance(loaded_embeddings, torch.Tensor):
            new_emb = loaded_embeddings.to(module_in[0].device)
        else:
            # Convert NumPy array to torch tensor (if it’s not already)
            new_emb = torch.tensor(loaded_embeddings, dtype=torch.float32, device=module_in[0].device)

        # Ensure shape consistency
        if new_emb.shape != module_in[0].shape:
            raise ValueError(
                f"Shape mismatch! new_emb {new_emb.shape} vs expected {module_in[0].shape}"
            )

        return (new_emb,)  # Must return a tuple to correctly replace input


    # Decide which hooks to apply
    capture_handle = None
    load_handle = None

    # If saving, register a capture hook on the INPUT of the linear layer
    if args.save_npy is not None:
        capture_handle = chosen_submodule.register_forward_hook(capture_hook)

    # If loading, load .npy file and register a load hook to replace input
    if args.load_npy is not None:
        global loaded_embeddings
        loaded_embeddings_data = np.load(args.load_npy, allow_pickle=True).item()
        loaded_embeddings = loaded_embeddings_data["embeddings"]

        # If quantized, apply dequantization and convert to float32
        if "scale" in loaded_embeddings_data and "zero_point" in loaded_embeddings_data:
            loaded_embeddings = fake_dequantize(
                loaded_embeddings,
                loaded_embeddings_data["scale"],
                loaded_embeddings_data["zero_point"]
            )

        # Ensure tensor is of correct dtype
        # loaded_embeddings = torch.tensor(loaded_embeddings, dtype=torch.float32)

        load_handle = chosen_submodule.register_forward_pre_hook(load_hook)


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
        if len(captured_embeddings) == 0:
            print("Warning: no embeddings were captured. Possibly the hook wasn't triggered.")
        else:
            final_array = captured_embeddings[-1]

            # Apply quantization if requested
            if args.quantize is not None:
                final_array, scale, zero_point = fake_quantize(final_array, args.quantize)
                save_data = {
                    "embeddings": final_array,
                    "scale": scale,
                    "zero_point": zero_point,
                }
            else:
                save_data = {"embeddings": final_array}

            np.save(args.save_npy, save_data)
            print(f"Saved embeddings to {args.save_npy} with shape {final_array.shape}. Quantization: {args.quantize}")

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

