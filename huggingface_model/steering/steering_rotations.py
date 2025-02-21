import torch
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from rich import print
from rich.console import Console
from spellchecker import SpellChecker  # Fallback
try:
    import enchant
except ImportError:
    console.print("[bold yellow]Enchant not installed. Falling back to pyspellchecker.[/bold yellow]")
    enchant = None

# Initialize console
console = Console()

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Print model details
console.print("[bold cyan]Model Details:[/bold cyan]")
console.print(model)

# --- Spell Checking Setup ---
if enchant:
    try:
        spell_fr = enchant.Dict("fr_FR")  # French dictionary
        spell_es = enchant.Dict("es")  # Spanish Dictionary
        enchant_available = True
    except Exception as e:
        console.print(f"[bold yellow]Enchant error: {e}. Falling back to pyspellchecker.[/bold yellow]")
        spell_fr = SpellChecker(language='fr')  # French (pyspellchecker)
        spell_es = SpellChecker(language='es')  # Spanish (pyspellchecker)
        enchant_available = False
else:
    spell_fr = SpellChecker(language='fr')
    spell_es = SpellChecker(language='es')
    enchant_available = False


def calculate_spelling_scores(text):
    """Calculates separate French and Spanish spelling scores."""
    words = text.split()
    correct_count_fr = 0
    correct_count_es = 0
    total_words = 0

    for word in words:
        cleaned_word = "".join(c for c in word.lower() if c.isalpha())
        if cleaned_word:
            total_words += 1
            if enchant_available and enchant:
                if spell_fr.check(cleaned_word):
                    correct_count_fr += 1
                if spell_es.check(cleaned_word):
                    correct_count_es += 1
            else:  # Use pyspellchecker
                if cleaned_word in spell_fr:
                    correct_count_fr += 1
                if cleaned_word in spell_es:
                    correct_count_es += 1

    score_fr = correct_count_fr / total_words if total_words else 0.0
    score_es = correct_count_es / total_words if total_words else 0.0
    return score_fr, score_es



# --- Model and Steering Setup ---
block_index = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to capture residual states
def capture_residual_hook(module, input, output, storage_list):
    hidden_states = output[0]
    storage_list.append(hidden_states.detach().mean(dim=1))
    return output

# Define prompts (Spanish and French)
prompt_es  = (
    "Hoy ha sido un día productivo. Me levanté temprano y comencé mi día organizando la casa. "
    "Barrí el suelo, lavé los platos y ordené la sala. Después, preparé una comida deliciosa y saludable. "
    "Finalmente, dediqué tiempo a planificar la semana, asegurándome de que todo estuviera en orden."
    "Terminé mi jornada sintiéndome satisfecho con todo lo que logré. Nada se compara con la sensación de tener un hogar limpio y organizado. "
    "Ahora puedo relajarme sabiendo que todo está bajo control."
)

prompt_en = (
    "Today was a productive day. I woke up early and started my day by organizing the house. "
    "I swept the floor, washed the dishes, and tidied up the living room. Afterwards, I prepared a delicious and healthy meal. "
    "Finally, I dedicated some time to planning the week, making sure everything was in order. "
    "I finished my day feeling satisfied with everything I accomplished. Nothing compares to the feeling of having a clean and organized home. "
    "Now I can relax knowing that everything is under control."
)

prompt_fr = (
    "Aujourd'hui a été une journée productive. Je me suis levé tôt et j'ai commencé ma journée en rangeant la maison. "
    "J'ai balayé le sol, lavé la vaisselle et rangé le salon. Ensuite, j'ai préparé un repas délicieux et sain. "
    "Enfin, j'ai consacré du temps à planifier la semaine, en m'assurant que tout était en ordre. "
    "J'ai terminé ma journée en me sentant satisfait de tout ce que j'ai accompli. Rien n'est comparable à la sensation d'avoir une maison propre et organisée. "
    "Maintenant, je peux me détendre en sachant que tout est sous contrôle."
)

prompt_de = (
    "Heute war ein produktiver Tag. Ich bin früh aufgestanden und habe meinen Tag damit begonnen, das Haus zu organisieren. "
    "Ich habe den Boden gefegt, das Geschirr gespült und das Wohnzimmer aufgeräumt. Danach habe ich eine köstliche und gesunde Mahlzeit zubereitet. "
    "Schließlich habe ich mir Zeit genommen, die Woche zu planen und sicherzustellen, dass alles in Ordnung ist. "
    "Ich beendete meinen Tag mit dem Gefühl, zufrieden mit allem zu sein, was ich erreicht habe. Nichts ist vergleichbar mit dem Gefühl, ein sauberes und organisiertes Zuhause zu haben. "
    "Jetzt kann ich mich entspannen, da ich weiß, dass alles unter Kontrolle ist."
)

prompt_pt = (
    "Hoje foi um dia produtivo. Levantei-me cedo e comecei o meu dia organizando a casa. "
    "Varri o chão, lavei a louça e arrumei a sala de estar. Depois, preparei uma refeição deliciosa e saudável. "
    "Finalmente, dediquei algum tempo a planejar a semana, garantindo que tudo estivesse em ordem. "
    "Terminei o meu dia sentindo-me satisfeito com tudo o que consegui. Nada se compara à sensação de ter um lar limpo e organizado. "
    "Agora posso relaxar sabendo que tudo está sob controle."
)

positive_prompt = prompt_pt
negative_prompt = prompt_fr


# Storage for residuals
residuals_pos, residuals_neg = [], []
console.print("[bold green]Capturing positive sentiment residuals...[/bold green]")
hook_handle_pos = model.transformer.h[block_index].register_forward_hook(
    lambda m, i, o: capture_residual_hook(m, i, o, residuals_pos)
)
inputs_pos = tokenizer(positive_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    model(**inputs_pos)
hook_handle_pos.remove()

console.print("[bold red]Capturing negative sentiment residuals...[/bold red]")
torch.cuda.empty_cache()
hook_handle_neg = model.transformer.h[block_index].register_forward_hook(
    lambda m, i, o: capture_residual_hook(m, i, o, residuals_neg)
)
inputs_neg = tokenizer(negative_prompt, return_tensors="pt").to(device)  # Use French prompt
with torch.no_grad():
    model(**inputs_neg)
vector_pos = torch.mean(torch.stack(residuals_pos), dim=0).to(device)
vector_neg = torch.mean(torch.stack(residuals_neg), dim=0).to(device)
hook_handle_neg.remove()

def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1)).unsqueeze(1)
    so = torch.sin(omega)
    if so.item() == 0:
        return (1.0-val)*low + val*high
    return (torch.sin((1.0-val)*omega)/so) * low + (torch.sin(val*omega)/so) * high

def modify_residual_hook(module, input, output, steering_vector):
    hidden_states = output[0]
    modified_hidden = hidden_states + steering_vector.float()
    return (modified_hidden,) + output[1:]

# --- Text Generation, Scoring, and CSV Export ---
slerp_values = np.linspace(-30, 30, 31)
console.print("[bold cyan]Generating text with sentiment steering using SLERP...[/bold cyan]")

# Calculate the angle between the vectors
angle = torch.acos((vector_pos * vector_neg).sum() / (torch.norm(vector_pos) * torch.norm(vector_neg))).item()
console.print(f"[bold green]Angle between positive and negative vectors: {angle:.4f} radians[/bold green]")


results = []  # List to store results

for val in slerp_values:
    interpolated_vector = slerp(torch.tensor([val]).to(device), vector_neg, vector_pos)
    target_block = model.transformer.h[block_index]
    hook_handle = target_block.register_forward_hook(
        lambda m, i, o: modify_residual_hook(m, i, o, interpolated_vector)
    )

    console.print(f"\n[bold yellow]=== SLERP Value: {val:.2f} ===[/bold yellow]")
    input_text = "\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
    generated_text = tokenizer.decode(outputs[0])
    console.print(f"[bold blue]Generated Output:[/bold blue] [bold magenta] {generated_text} [/bold magenta]")

    # Calculate spelling scores
    score_fr, score_es = calculate_spelling_scores(generated_text)
    console.print(f"[bold magenta]French Spelling Score: {score_fr:.4f}[/bold magenta]")
    console.print(f"[bold cyan]Spanish Spelling Score: {score_es:.4f}[/bold cyan]")

    # Store results
    results.append({
        'slerp_value': val,
        'french_score': f"{score_fr:.2f}",
        'spanish_score': f"{score_es:.2f}",
        'generated_text': generated_text
    })

    hook_handle.remove()

# Convert results to DataFrame and export to CSV
df = pd.DataFrame(results)
df.to_csv("spelling_scores.csv", index=False)
console.print("[bold green]Spelling scores saved to spelling_scores.csv[/bold green]")
