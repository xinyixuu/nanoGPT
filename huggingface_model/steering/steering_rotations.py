import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from rich import print
from rich.console import Console
from rich.table import Table
from prompts import *
import re

# Initialize console
console = Console()

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# --- Model and Steering Setup ---
block_index = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to capture residual states
def capture_residual_hook(module, input, output, storage_list):
    hidden_states = output[0]
    storage_list.append(hidden_states.detach().mean(dim=1))
    return output

positive_prompt = prompt_ko
positive_prompt_label = "Korean"
negative_prompt = prompt_ja
negative_prompt_label = "Japanese"

# --- Character Counting Functions ---
def count_korean_characters(text):
    """Counts Korean characters (Hangul) in the text."""
    return len(re.findall(r'[\uAC00-\uD7A3]', text))

def count_japanese_characters(text):
    """Counts Japanese characters (Hiragana, Katakana, and Kanji) in the text."""
    return len(re.findall(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9faf]', text))

# --- Residual Capture ---
residuals_pos, residuals_neg = [], []

console.print(f"[bold green]Capturing {positive_prompt_label} residuals...[/bold green]")
hook_handle_pos = model.transformer.h[block_index].register_forward_hook(
    lambda m, i, o: capture_residual_hook(m, i, o, residuals_pos)
)
inputs_pos = tokenizer(positive_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    model(**inputs_pos)
hook_handle_pos.remove()

console.print(f"[bold red]Capturing {negative_prompt_label} residuals...[/bold red]")
torch.cuda.empty_cache()
hook_handle_neg = model.transformer.h[block_index].register_forward_hook(
    lambda m, i, o: capture_residual_hook(m, i, o, residuals_neg)
)
inputs_neg = tokenizer(negative_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    model(**inputs_neg)
vector_pos = torch.mean(torch.stack(residuals_pos), dim=0).to(device)
vector_neg = torch.mean(torch.stack(residuals_neg), dim=0).to(device)
hook_handle_neg.remove()

# --- SLERP and Hook Modification ---
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1)).unsqueeze(1)
    so = torch.sin(omega)
    if so.item() == 0:
        return (1.0-val)*low + val*high  # L'Hopital's rule/LERP
    return (torch.sin((1.0-val)*omega)/so) * low + (torch.sin(val*omega)/so) * high

def modify_residual_hook(module, input, output, steering_vector):
    hidden_states = output[0]
    modified_hidden = hidden_states + steering_vector.float()
    return (modified_hidden,) + output[1:]

# --- Text Generation and Analysis ---
slerp_values = np.linspace(-16, 16, 33)
console.print("[bold cyan]Generating text with language steering using SLERP...[/bold cyan]")

angle_rad = torch.acos((vector_pos * vector_neg).sum() / (torch.norm(vector_pos) * torch.norm(vector_neg))).item()
angle_deg = np.degrees(angle_rad)
console.print(f"[bold green]Angle between {positive_prompt_label} and {negative_prompt_label} vectors: {angle_rad:.4f} radians ({angle_deg:.2f} degrees)[/bold green]")

results = []
for val in slerp_values:
    interpolated_vector = slerp(torch.tensor([val]).to(device), vector_neg, vector_pos)
    target_block = model.transformer.h[block_index]
    hook_handle = target_block.register_forward_hook(
        lambda m, i, o: modify_residual_hook(m, i, o, interpolated_vector)
    )

    table = Table(title=f"[bold yellow]=== SLERP Value: {val:.2f} ===", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold cyan", width=30)
    table.add_column("Value", justify="left")

    input_text = "\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    table.add_row("Generated Output", f"[bold magenta]{generated_text}[/bold magenta]")

    korean_count = count_korean_characters(generated_text)
    japanese_count = count_japanese_characters(generated_text)
    rotation_angle_rad = val * angle_rad
    rotation_angle_deg = np.degrees(rotation_angle_rad)
    magnitude = torch.norm(interpolated_vector).item()

    table.add_row("Korean Character Count", str(korean_count))
    table.add_row("Japanese Character Count", str(japanese_count))
    table.add_row("Magnitude of Steering Vector", f"{magnitude:.4f}")
    table.add_row("Rotation Angle", f"{rotation_angle_rad:.4f} radians ({rotation_angle_deg:.2f} degrees)")

    console.print(table)
    console.print("-" * 50)

    results.append({
        'slerp_value': val,
        'korean_count': korean_count,
        'japanese_count': japanese_count,
        'rotation_angle_rad': rotation_angle_rad, #use radians for polar
        'rotation_angle_deg': rotation_angle_deg,
        'magnitude': magnitude
    })

    hook_handle.remove()


# --- Seaborn Scatter Plot (Cartesian) ---
console.print("\n[bold cyan]Generating Seaborn Scatter Plot (Cartesian)...[/bold cyan]")
df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='rotation_angle_deg', y='korean_count', data=df, label='Korean', color='green', s=50)
sns.scatterplot(x='rotation_angle_deg', y='japanese_count', data=df, label='Japanese', color='red', s=50)
plt.xlabel("Rotation Angle (degrees)", fontsize=12)
plt.ylabel("Character Count", fontsize=12)
plt.title("Language Steering: Character Counts vs. Rotation Angle", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True)
plt.xlim(df['rotation_angle_deg'].min() - 5, df['rotation_angle_deg'].max() + 5)
plt.ylim(-5, max(df['korean_count'].max(), df['japanese_count'].max()) + 10)
plt.tight_layout()
plt.savefig("language_steering_plot_cartesian.png")
console.print("[bold green]Cartesian scatter plot saved to language_steering_plot_cartesian.png[/bold green]")

# --- Seaborn Polar Plot ---
console.print("\n[bold cyan]Generating Seaborn Polar Plot...[/bold cyan]")

# Set up the polar plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='polar')

# Plot Korean data
sns.scatterplot(x='rotation_angle_rad', y='korean_count', data=df, label='Korean', color='green', s=50, ax=ax)
# Plot Japanese Data
sns.scatterplot(x='rotation_angle_rad', y='japanese_count', data=df, label='Japanese', color='red', s=50, ax=ax)

# Customize the plot
ax.set_theta_zero_location("N")  # Set 0 degrees to North
ax.set_theta_direction(-1)  # Clockwise direction
ax.set_rlabel_position(90) # Place radial labels nicely
ax.set_xlabel("Rotation Angle (radians)", labelpad=20, fontsize=12) #labelpad moves label away from plot
ax.set_ylabel("Character Count", labelpad=30, fontsize=12) #y label is radial
ax.set_title("Language Steering: Character Counts vs. Rotation Angle (Polar)", fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))  # Adjust legend position

plt.tight_layout()
plt.savefig("language_steering_plot_polar.png")
console.print("[bold green]Polar scatter plot saved to language_steering_plot_polar.png[/bold green]")


# Convert results to DataFrame and export to CSV
df_csv = pd.DataFrame(results)
df_csv.to_csv("language_steering_results.csv", index=False)
console.print("[bold green]Language steering results saved to language_steering_results.csv[/bold green]")
