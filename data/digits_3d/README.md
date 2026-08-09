# Sequential digits with held-out vocabulary characters

This synthetic dataset repeats `0123456789`. Its character vocabulary is fixed
to `0123456789abcd`, but `a`, `b`, `c`, and `d` occur in neither split. This
makes the letters useful controls when inspecting how training moves token
vectors in a model whose embedding width is exactly three.

The data is generated locally and is released under the repository's license;
there is no external source or additional dataset license.

```bash
python3 data/digits_3d/prepare.py
```

The command creates the standard nanoGPT `train.bin`, `val.bin`, and `meta.pkl`
files. Generated binaries are intentionally ignored by git.
