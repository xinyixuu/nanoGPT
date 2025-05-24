# shared_param_utils.py

import sys
import torch
import torch.nn as nn

import copy

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
            shared_sym = self.config.shared_mlp_sym
        elif layer_type == "attn":
            shared_size = self.config.shared_attn_size
            shared_sym = self.config.shared_attn_sym
        else:
            sys.exit(f"{layer_type} not supported. Use 'mlp' or 'attn' only.")

        shared_group = []
        layer_block = None

        # For cycling multiple attention variants
        attn_variant_index = 0

        for i in range(self.config.n_layer):

            # ------------------------------------------------------------------
            # Build a per-layer clone of the config and apply any *layerlist
            # overrides.  Example:  --mlp_size_layerlist 100 200 300
            # → layer 0→100, 1→200, 2→300, 3→100, 4→200, ...
            # ------------------------------------------------------------------
            layer_config = copy.deepcopy(self.config)

            for attr in dir(self.config):
                if attr.endswith("_layerlist"):
                    lst = getattr(self.config, attr)
                    if not lst:          # [], None, or empty → ignore
                        continue
                    core_attr = attr[:-10]         # strip "_layerlist"
                    raw_val   = lst[i % len(lst)]  # cyclic selection

                    # Cast to the same type as the original value if it exists
                    if hasattr(self.config, core_attr):
                        ref_val = getattr(self.config, core_attr)
                        # --------------------------------------------------
                        # 1.  If the runtime value is NOT None, keep
                        #     the old behaviour (inspect the value itself).
                        # 2.  If it IS None, consult the dataclass
                        #     annotation to discover the intended type.
                        # --------------------------------------------------
                        if ref_val is None:
                            from typing import get_origin, get_args
                            anno = type(self.config).__annotations__.get(
                                        core_attr, None)
                            hinted = str          # fallback
                            if anno is not None:
                                origin = get_origin(anno)  # Optional → Union
                                args   = get_args(anno)
                                if origin is None:         # plain type
                                    hinted = anno
                                elif (len(args) == 2
                                      and type(None) in args):  # T | None
                                    hinted = (args[0] if args[1] is type(None)
                                              else args[1])

                            # Cast according to *hinted*
                            if hinted is bool:
                                raw_val = str(raw_val).lower() in ("1", "true", "yes", "0", "false", "no")
                            elif hinted is int:
                                raw_val = int(raw_val)
                            elif hinted is float:
                                raw_val = float(raw_val)
                        else:
                            # old behaviour
                            if isinstance(ref_val, bool):
                                raw_val = str(raw_val).lower() in (
                                    "1", "true", "yes")
                            elif isinstance(ref_val, int):
                                raw_val = int(raw_val)
                            elif isinstance(ref_val, float):
                                raw_val = float(raw_val)
                    setattr(layer_config, core_attr, raw_val)

            # Create a new layer block every "shared_size"
            if i % shared_size == 0:
                if layer_type == "mlp":
                    # Possibly handle MoE
                    if self.config.use_moe and i % self.config.moe_layer_freq == 0:
                        layer_block = MoELayer(layer_config)
                    else:
                        layer_block = get_mlp_instance(layer_config)
                else:
                    # Determine which attention variant to use
                    variant = None
                    if len(self.attention_list) == 1:
                        variant = self.attention_list[0]
                    else:
                        # Cycle through multiple variants
                        variant = self.attention_list[attn_variant_index % len(self.attention_list)]
                        attn_variant_index += 1

                    attn_cls = attention_dictionary[variant]
                    layer_block = attn_cls(layer_config, fire_pos_enc=self.fire_pos_enc)

            # Add this (possibly reused) block to the list
            shared_group.append(layer_block)

            # If symmetrical sharing is requested
            if shared_sym:
                # Even number of layers
                if self.config.n_layer % 2 == 0:
                    # Once we reach halfway-1, we mirror
                    if i == (self.config.n_layer // 2 - 1):
                        for j in range(i + 1):
                            shared_group.append(shared_group[i - j])
                        return shared_group
                else:
                    # Odd number of layers
                    if i == (self.config.n_layer // 2):
                        for j in range(i):
                            shared_group.append(shared_group[i - j])
                        return shared_group

        return shared_group

