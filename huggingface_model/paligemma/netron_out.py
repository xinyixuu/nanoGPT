import torch
import os
from transformers import AutoProcessor, AutoModel
from rich.console import Console
from torchinfo import summary
from torchviz import make_dot
import hiddenlayer as hl

console = Console()

# Load model and processor
console.print("[bold blue]Loading model and processor...[/]")
model_name = "google/siglip-so400m-patch14-384"  # Or your specific SigLIP model
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Determine device (GPU or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    console.print(f"[green]Using CUDA device: {torch.cuda.get_device_name(0)}[/]")
else:
    device = torch.device("cpu")
    console.print("[yellow]Using CPU[/]")

# Move the model to the device
model = model.to(device)

# Export Model Details (Textual)
console.print("[bold blue]Model Details (Textual):[/]")
console.print(model)

# torchinfo summary
summary_path = "siglip_summary.txt"
with open(summary_path, "w") as f:
    console.print("[bold blue]Generating torchinfo summary...[/]")
    summary_str = str(summary(model.vision_model, input_size=(1, 3, 384, 384)))
    f.write(summary_str)
console.print(f"[green]Torchinfo summary saved to {summary_path}[/]")


# Wrap the vision part of the model 
class SigLipModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.vision_model

    def forward(self, pixel_values):
        pixel_values = pixel_values.to(device)  # Move input to device
        vision_outputs = self.model(pixel_values=pixel_values)
        return vision_outputs

model_wrapper = SigLipModelWrapper(model)

# ONNX Export
onnx_path = "siglip_model.onnx"
if not os.path.exists(onnx_path):
    console.print(f"[bold blue]Exporting model to ONNX for Netron (at {onnx_path})...[/]")
    dummy_image = torch.randn(1, 3, 384, 384).to(device)
    torch.onnx.export(
        model_wrapper,
        dummy_image,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image_input"],
        output_names=["output"],
        dynamic_axes={"image_input": {0: "batch_size"}},
    )
    console.print(f"[green]Model exported to {onnx_path}[/]")
else:
    console.print(f"[yellow]ONNX model already exists at {onnx_path}. Skipping export.[/]")

