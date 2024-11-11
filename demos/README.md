# Demos

This folder will hold repeatable demonstrations of features and results.

## Shifted GELU

This shows that the GELU wants to shift:


Call this from either repo root dir, or from demos dir, but ensure the relative
path for ckpt_path is from your present directory.

Example for calling from repo root directory:

```bash
python3 demos/check_ckpt_for_gelu_shift.py \
        --ckpt_path out/ckpt.pt
```