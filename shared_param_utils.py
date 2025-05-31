# shared_param_utils.py

import sys
import torch
import torch.nn as nn

import copy
import sys
from typing import get_origin, get_args
from string import ascii_uppercase
from rich.console import Console

_console = Console()

from variations.attention_variations import attention_dictionary
from variations.mlp_variations import get_mlp_instance
from variations.moe_variations import MoELayer
from variations.position_encoding_variations import FIRE

class SharedParamGroupCreator:
    """
    A helper class to create shared parameter groups (either MLP or Attn).
    It supports:
      - Reuse of parameter blocks every 'shared_size' layers
      - Optional symmetry if 'shared_sym' is True
      - Multiple attention variants if config.attention_list is provided
      - MoE layers for MLP if config.use_moe is True
    """

    def __init__(self, config):
        self.config = config

        # For attention variants, either use the single config.attention_variant
        # or cycle through a list if config.attention_list is given.
        self.attention_list = []
        if hasattr(config, 'attention_list') and config.attention_list:
            self.attention_list = config.attention_list
        else:
            self.attention_list = [config.attention_variant]

        # Pre-instantiate a single FIRE module to share, if needed
        self.fire_pos_enc = None
        if config.shared_fire_embeddings:
            self.fire_pos_enc = FIRE(config, num_heads=config.n_head)


    def create_shared_param_group(self, layer_type):
        """
        Creates a shared list of layer blocks (either MLP or Attn), optionally
        reusing blocks every 'shared_size' layers and reflecting them symmetrically
        if 'shared_sym' is True.

        For attention layers, we can cycle through multiple attention variants
        if config.attention_list is not empty.

        Args:
            layer_type (str): "mlp" or "attn"

        Returns:
            list of layer_blocks
        """

        if layer_type == "mlp":
            shared_size = self.config.shared_mlp_size
            shared_sym  = self.config.shared_mlp_sym
            seq_len     = self.config.shared_mlp_seq

        elif layer_type == "attn":
            shared_size = self.config.shared_attn_size
            shared_sym = self.config.shared_attn_sym
            seq_len     = self.config.shared_attn_seq
        else:
            sys.exit(f"{layer_type} not supported. Use 'mlp' or 'attn' only.")

        shared_group = []
        layer_block = None

        # For cycling multiple attention variants
        attn_variant_index = 0

        # ────────────────────────────────
        # Cyclic-sequence sharing pool
        # ────────────────────────────────
        if seq_len < 1:
            raise ValueError("shared_*_seq must be ≥ 1")
        seq_pool: list[nn.Module | None] = [None] * seq_len

        # If we'll mirror, build only the *first half plus centre* for odd n.
        if shared_sym:
            num_physical = (self.config.n_layer + 1) // 2   # ceil(n/2)
        else:
            num_physical = self.config.n_layer

        for i in range(num_physical):


            # ------------------------------------------------------------------
            # Build a per-layer clone of the config and apply any *layerlist
            # overrides.  Example:  --mlp_size_layerlist 100 200 300
            # → layer 0→100, 1→200, 2→300, 3→100, 4→200, ...
            # ------------------------------------------------------------------
            layer_config = copy.deepcopy(self.config)
            layer_config.layer_idx = i

            for attr in dir(self.config):
                if attr.endswith("_layerlist"):
                    lst = getattr(self.config, attr)
                    if not lst:          # [], None, or empty → ignore
                        continue
                    core_attr = attr[:-10]         # strip "_layerlist"
                    raw_val   = lst[i % len(lst)]  # cyclic selection

                    if hasattr(self.config, core_attr):
                        ref_val = getattr(self.config, core_attr)


                        _SENTINEL_NONE = {"", "none", "null"}

                        def _is_none(txt) -> bool:
                            return str(txt).strip().lower() in _SENTINEL_NONE

                        def _as_bool(txt):
                            if _is_none(txt):
                                return None
                            truthy_values = {"1", "true", "yes", "y", "on"}
                            falsy_values = {"0", "false", "no", "n", "off"}
                            txt_lower = str(txt).strip().lower()
                            if txt_lower in truthy_values:
                                return True
                            elif txt_lower in falsy_values:
                                return False
                            else:
                                raise ValueError(f"Invalid boolean value: {txt}")
                        # a) If the runtime value is *not* None we can
                        #    rely on its actual Python type.
                        if ref_val is not None:
                            if isinstance(ref_val, bool):
                                raw_val = _as_bool(raw_val)
                            elif isinstance(ref_val, int):
                                raw_val = int(raw_val)
                            elif isinstance(ref_val, float):
                                raw_val = float(raw_val)
                        # b) Otherwise, fall back to the dataclass annotation
                        #    to guess the intended type (handles `T | None`).
                        else:
                            anno = type(self.config).__annotations__.get(core_attr)
                            hinted = str  # default: leave string as-is
                            if anno:
                                origin = get_origin(anno)
                                args   = get_args(anno)
                                if origin is None:
                                    hinted = anno
                                elif len(args) == 2 and type(None) in args:
                                    hinted = next(a for a in args if a is not type(None))

                            if _is_none(raw_val):
                                raw_val = None
                            elif hinted is bool:
                                raw_val = _as_bool(raw_val)
                            elif hinted is int:
                                raw_val = int(raw_val)
                            elif hinted is float:
                                raw_val = float(raw_val)

                    setattr(layer_config, core_attr, raw_val)

            # Decide which sharing mode we are in
            if seq_len > 1:
                seq_idx = i % seq_len
                layer_block = seq_pool[seq_idx]
                if layer_block is None:
                    layer_block = _build_block(layer_type, layer_config, self.fire_pos_enc)

                    seq_pool[seq_idx] = layer_block
            else:
                # create a new block only every k layers,
                # otherwise keep re-using the last one
                if i % shared_size == 0 or layer_block is None:
                    layer_block = _build_block(layer_type, layer_config, self.fire_pos_enc)

            # Add this (possibly reused) block for *every* logical layer
            shared_group.append(layer_block)

        # ────────────────────────────────
        # Optional symmetry mirroring
        # ────────────────────────────────
        if shared_sym:
            # exclude centre element for odd n to avoid duplication
            mirror = list(reversed(shared_group[:-1])) if self.config.n_layer % 2 else list(reversed(shared_group))
            shared_group.extend(mirror)

        # ── pretty debug print ──────────────────────────────
        msg = f"[bold cyan]{layer_type.upper()}[/] sharing: {_label_sequence(shared_group)}"
        _console.print(msg)

        return shared_group

# ─────────────────────────────────────────────────────────────
# Helper to move logic out of the main loop
# ─────────────────────────────────────────────────────────────

def _build_block(layer_type: str, layer_config, fire_pos_enc):
    """Factory wrapper so we don’t repeat the same if/else everywhere."""
    if layer_type == "mlp":
        if layer_config.use_moe and layer_config.layer_idx % layer_config.moe_layer_freq == 0:
            return MoELayer(layer_config)
        return get_mlp_instance(layer_config)
    # Attention
    variant = layer_config.attention_variant
    if hasattr(layer_config, "attention_list") and layer_config.attention_list:
        variant_list = layer_config.attention_list
        idx = layer_config.layer_idx % len(variant_list)
        variant = variant_list[idx]
    attn_cls = attention_dictionary[variant]
    return attn_cls(layer_config, fire_pos_enc=fire_pos_enc)

# ─────────────────────────────────────────────────────────────
#  Debug printer
# ─────────────────────────────────────────────────────────────

def _label_sequence(blocks) -> str:
    """
    Convert a list of blocks into a string like 'A B C A B C'.
    Equal objects (by identity) get the same letter.
    """
    mapping = {}
    label_iter = iter(ascii_uppercase)
    letters = []
    for blk in blocks:
        if blk not in mapping:
            mapping[blk] = next(label_iter, '?')
        letters.append(mapping[blk])
    return " ".join(letters)
