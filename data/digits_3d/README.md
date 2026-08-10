# Sequential digits with held-out vocabulary characters

This synthetic dataset repeats a configurable sequence of digit-like symbols.
The first ten are `0`–`9`; larger `--num-digits` values add renderable
punctuation. A configurable number of letters are included in the vocabulary
but occur in neither split, making them useful controls when inspecting how
training moves token vectors. `EMBEDDING_DIM` selects the model width; widths
above three are projected into the viewer with a shared three-component PCA.

The demo enables fixed-norm embeddings by default: initialization and every
optimizer update project each vector to radius `sqrt(EMBEDDING_DIM)`. Set
`WTE_FIXED_NORM=false` to retain the original unconstrained training mode, or
set `WTE_FIXED_NORM_VALUE` to choose another radius.

To compare unconstrained, square-root-dimension-radius, and unit-radius
initialization over several vocabulary sizes and widths `3`, `8`, `16`, and
`64`, run `demos/digits_3d_trajectory_sweep.sh`. Its `EMBEDDING_DIMS`,
`DIGIT_COUNTS`, and `LETTER_COUNTS` variables accept whitespace-separated
values. Sweep runs train for 10,000 iterations by default; override
`SWEEP_MAX_ITERS` for shorter smoke tests or longer studies.

The data is generated locally and is released under the repository's license;
there is no external source or additional dataset license.

```bash
python3 data/digits_3d/prepare.py --num-digits 10 --num-letters 4
```

The command creates the standard nanoGPT `train.bin`, `val.bin`, and `meta.pkl`
files. Generated binaries are intentionally ignored by git.
