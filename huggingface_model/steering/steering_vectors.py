import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from rich import print
from rich.console import Console
from prompts import *

# Initialize console for rich formatting
console = Console()

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Print model details
console.print("[bold cyan]Model Details:[/bold cyan]")
console.print(model)

# Standard steering vectors for GPT-2 Small (124M parameters) at block index 5
block_index = 5
hidden_size = model.config.n_embd  # 768 for GPT-2 Small

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to capture and average residual states
def capture_residual_hook(module, module_input, module_output, storage_list):
    hidden_states = module_output[0]  # Extract hidden states
    storage_list.append(hidden_states.detach().mean(dim=1))  # Average across tokens
    return module_output

# Define longer sentiment-steering prompts
positive_prompt = prompt_es
negative_prompt = prompt_en

# Storage for residuals
residuals_pos, residuals_neg = [], []

console.print("[bold green]Capturing positive sentiment residuals...[/bold green]")
# Register hooks to capture residual states
hook_handle_pos = model.transformer.h[block_index].register_forward_hook(
    lambda module, module_input, module_output: capture_residual_hook(module, module_input, module_output, residuals_pos)
)
inputs_pos = tokenizer(positive_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    model(**inputs_pos)

# Remove positive hook immediately to prevent corruption
hook_handle_pos.remove()

console.print("[bold red]Capturing negative sentiment residuals...[/bold red]")
torch.cuda.empty_cache()

hook_handle_neg = model.transformer.h[block_index].register_forward_hook(
    lambda module, module_input, module_output: capture_residual_hook(module, module_input, module_output, residuals_neg)
)
inputs_neg = tokenizer(negative_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    model(**inputs_neg)

# Compute average sentiment steering vector difference
vector_pos = torch.mean(torch.stack(residuals_pos), dim=0)
vector_neg = torch.mean(torch.stack(residuals_neg), dim=0)
steering_vector = vector_pos - vector_neg

# Cleanup negative hook
hook_handle_neg.remove()

# Hook function to modify residual stream
def modify_residual_hook(module, module_input, module_output, steering_vector):
    hidden_states = module_output[0]  # Extract hidden states
    modified_hidden = hidden_states + steering_vector
    return (modified_hidden,) + module_output[1:]  # Return as tuple

# Iterate over different steering vector multipliers
steering_multipliers = np.linspace(-5, 5, 41)  # Steps from -1 to 1 in increments of 0.1

console.print("[bold cyan]Generating text with sentiment steering...[/bold cyan]")
# Run generation for each setting
for multiplier in steering_multipliers:
    scaled_vector = multiplier * steering_vector.to(device)

    # Register hook with scaled vector
    target_block = model.transformer.h[block_index]
    hook_handle = target_block.register_forward_hook(
        lambda module, module_input, module_output: modify_residual_hook(module, module_input, module_output, scaled_vector)
    )

    console.print(f"\n[bold yellow]=== Steering Multiplier: {multiplier:.2f} ===[/bold yellow]")

    # Use input text to elicit different responses while preventing repetition
    input_text = "\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            repetition_penalty=1.2,  # Discourages direct repetition
            no_repeat_ngram_size=3,  # Prevents repeating trigrams
            temperature=0.8,  # Balances randomness
            top_k=50,  # Limits sampling to top 50 words
            top_p=0.9  # Nucleus sampling for diversity
        )
    console.print(f"[bold blue]Generated Output:[/bold blue] {tokenizer.decode(outputs[0])}")

    # Remove hook
    hook_handle.remove()


