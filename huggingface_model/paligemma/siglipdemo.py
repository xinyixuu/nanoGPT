from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import track, Progress
from tqdm import tqdm
import cv2
import numpy as np
from rich.style import Style

console = Console()

# Load model and processor
console.print("[bold blue]Loading model and processor...[/]")
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# Image URL
url = "http://images.cocodataset.org/val2017/000000039769.jpg"

# Candidate labels
candidate_labels = ["a photo of cats", "a photo of dogs", "grayscale photo of cats", "grayscale photo of dogs"]
texts = [f'This is a photo of {label}.' for label in candidate_labels]

def process_image_and_get_logits(image, texts, desc):
    """Processes an image with different configurations and returns logits."""
    inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits_per_image

# Original image
console.print(f"[bold blue]Downloading and processing original image...[/]")
original_image = Image.open(requests.get(url, stream=True).raw)

# Different image scales and types
image_configs = {
    f"{original_image.width}Wx{original_image.height}H": original_image,
    f"{0.5*original_image.width}Wx{0.5*original_image.height}H": original_image.resize(
        (original_image.width // 2, original_image.height // 2)
    ),
    f"{2*original_image.width}Wx{2*original_image.height}H": original_image.resize(
        (original_image.width * 2, original_image.height * 2)
    ),
    f"Gray{original_image.width}Wx{original_image.height}H": Image.merge("RGB", [original_image.convert("L")] * 3)
}

# Define a style for highlighting
highlight_style = Style(color="light_sky_blue1")

# Create tables
logits_table = Table(title="Image Classification Logits Comparison", show_header=True, header_style="bold magenta")
sigmoid_table = Table(title="Image Classification Sigmoid Comparison", show_header=True, header_style="bold magenta")
softmax_table = Table(title="Image Classification Softmax Comparison", show_header=True, header_style="bold magenta")

# Add columns to tables
for table in [logits_table, sigmoid_table, softmax_table]:
    table.add_column("Image Type", justify="right", style="dim", width=20) # increase width here

    for label in candidate_labels:
        if table == logits_table:
            table.add_column(f"{label} (Logits)")
        elif table == sigmoid_table:
            table.add_column(f"{label} (Sigmoid)")
        else:
            table.add_column(f"{label} (Softmax)")

# Process images and add to tables
with Progress() as progress:
    task = progress.add_task("Processing image variations...", total=len(image_configs))
    for desc, image in image_configs.items():
        progress.update(task, advance=1, description=f"Processing: {desc}")
        logits = process_image_and_get_logits(image, texts, desc)
        softmax_probs = torch.softmax(logits, dim=1)
        sigmoid_probs = torch.sigmoid(logits)

        # Collect values for highlighting
        logits_values = [f"{logit:.4f}" for logit in logits[0]]
        sigmoid_values = [f"{prob:.4f}" for prob in sigmoid_probs[0]]
        softmax_values = [f"{prob:.4f}" for prob in softmax_probs[0]]

        # Find indices of max values
        max_logit_index = logits_values.index(max(logits_values, key=lambda x: float(x)))
        max_sigmoid_index = sigmoid_values.index(max(sigmoid_values, key=lambda x: float(x)))
        max_softmax_index = softmax_values.index(max(softmax_values, key=lambda x: float(x)))

        # Add rows to tables with highlighting
        logits_row = [desc] + [
            f"[{highlight_style}]{val}[/]" if i == max_logit_index else val
            for i, val in enumerate(logits_values)
        ]
        sigmoid_row = [desc] + [
            f"[{highlight_style}]{val}[/]" if i == max_sigmoid_index else val
            for i, val in enumerate(sigmoid_values)
        ]
        softmax_row = [desc] + [
            f"[{highlight_style}]{val}[/]" if i == max_softmax_index else val
            for i, val in enumerate(softmax_values)
        ]

        logits_table.add_row(*logits_row)
        sigmoid_table.add_row(*sigmoid_row)
        softmax_table.add_row(*softmax_row)

# Display tables
console.print(logits_table)
console.print(sigmoid_table)
console.print(softmax_table)
