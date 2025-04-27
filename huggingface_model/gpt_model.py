import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Config

# Custom Config
class CustomGPT2Config(GPT2Config):
    def __init__(self, 
                 attention_softmax_variant="standard",
                 layernorm_variant="standard",
                 activation_variant="gelu",
                 ffn_hidden_multiplier=4,
                 **kwargs):
        super().__init__(**kwargs)
        self.attention_softmax_variant = attention_softmax_variant
        self.layernorm_variant = layernorm_variant
        self.activation_variant = activation_variant
        self.ffn_hidden_multiplier = ffn_hidden_multiplier

class CustomMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.n_embd * config.ffn_hidden_multiplier
        self.fc_in = nn.Linear(config.n_embd, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, config.n_embd)
        self.activation_variant = config.activation_variant

    def forward(self, x):
        x = self.fc_in(x)
        if self.activation_variant == "gelu":
            x = nn.functional.gelu(x)
        elif self.activation_variant == "relu":
            x = nn.functional.relu(x)
        elif self.activation_variant == "silu":
            x = nn.functional.silu(x)
        else:
            raise ValueError(f"Unknown activation variant: {self.activation_variant}")
        x = self.fc_out(x)
        return x

class CustomAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.scale_attn_weights = True
        self.attention_softmax_variant = config.attention_softmax_variant

    def forward(self, hidden_states, attention_mask=None):
        query, key, value = self.c_attn(hidden_states).split(hidden_states.size(-1), dim=2)

        query = query.view(hidden_states.size(0), -1, self.n_head, hidden_states.size(-1) // self.n_head).transpose(1,2)
        key = key.view(hidden_states.size(0), -1, self.n_head, hidden_states.size(-1) // self.n_head).transpose(1,2)
        value = value.view(hidden_states.size(0), -1, self.n_head, hidden_states.size(-1) // self.n_head).transpose(1,2)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (key.size(-1) ** 0.5)

        if self.attention_softmax_variant == "standard":
            attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        elif self.attention_softmax_variant == "relu":
            attn_probs = nn.functional.relu(attn_weights)
        elif self.attention_softmax_variant == "squared_relu":
            attn_probs = nn.functional.relu(attn_weights) ** 2
        else:
            raise ValueError(f"Unknown softmax variant: {self.attention_softmax_variant}")

        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1,2).contiguous().view(hidden_states.size(0), -1, hidden_states.size(-1))
        attn_output = self.c_proj(attn_output)
        return attn_output

class CustomGPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = self.build_layernorm(config)
        self.attn = CustomAttention(config)
        self.ln_2 = self.build_layernorm(config)
        self.mlp = CustomMLP(config)

    def build_layernorm(self, config):
        if config.layernorm_variant == "standard":
            return nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        elif config.layernorm_variant == "rmsnorm":
            return RMSNorm(config.n_embd)

    def forward(self, hidden_states, attention_mask=None):
        attn_output = self.attn(self.ln_1(hidden_states), attention_mask)
        hidden_states = hidden_states + attn_output
        mlp_output = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + mlp_output
        return hidden_states

class CustomGPT2Model(GPT2PreTrainedModel):
    config_class = CustomGPT2Config  # <- Use your custom config!

    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        self.h = nn.ModuleList([
            CustomGPT2Block(config) for _ in range(config.n_layer)
        ])
        self.ln_f = self.build_layernorm(config)
        
        self.init_weights()

    def build_layernorm(self, config):
        if config.layernorm_variant == "standard":
            return nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        elif config.layernorm_variant == "rmsnorm":
            return RMSNorm(config.n_embd)  # (You'd implement RMSNorm separately)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(torch.arange(0, input_shape[-1], device=input_ids.device))
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states
    
config = CustomGPT2Config(
    n_embd=768,
    n_layer=12,
    n_head=12,
    activation_variant="relu",
    attention_softmax_variant="squared_relu",
    layernorm_variant="rmsnorm",
    ffn_hidden_multiplier=2,  # Half-size FFN
)
model = CustomGPT2Model(config)

# Load a pre-trained GPT-2 model and replace its transformer with the custom model
pretrained_model = GPT2LMHeadModel.from_pretrained("gpt2")
pretrained_model.transformer = CustomGPT2Model.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")