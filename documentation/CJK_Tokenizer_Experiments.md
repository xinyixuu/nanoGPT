# CJK char / char-BPE / byte-fallback experiments

Both entry points share normalization, train/validation splits, IPA conversion,
model settings, BPB reporting and the three validation per-token metrics from
[PR #917](https://github.com/ReaLLMASIC/ReaLLM-Forge/pull/917).

| Entry point | Comparison | Default data/output/log suffix |
| --- | --- | --- |
| `demos/cjk_ipa_charbpe_compare.sh` | char vs char-BPE (8192 vocabulary) | `cjk_ipa_charbpe` |
| `demos/cjk_ipa_char_byte_fallback_compare.sh` | char vs character whitelist + UTF-8 byte fallback | `cjk_ipa_char_byte_fallback` |

Each experiment has 12 runs: Japanese / Korean / Chinese × original / IPA ×
two tokenizers. Defaults match the existing experiment: 6 layers, 6 heads,
embedding dimension 384, block size 256, batch size 64, 3500 iterations,
learning rate 0.001, seed 1337, bfloat16, evaluation every 500 iterations
using 50 sampled batches per split. Training runs sequentially on `cuda:0`;
`CUDA_VISIBLE_DEVICES` chooses the physical GPU.

## Byte-fallback vocabulary

The new demo defaults to `FALLBACK_CHAR_LIMIT=256`. For each language and
representation independently, the whitelist contains the K most frequent
non-whitespace Unicode characters in the training split. Frequency ties are
broken by Unicode order. Validation text is never used to select this whitelist.
Whitespace is encoded as bytes because the existing line-based custom tokenizer
strips whitespace from its vocabulary file.

The demo calls `prepare.py --method custom_char_byte_fallback`. IDs 0–255
represent raw UTF-8 bytes; IDs 256 onward represent whitelist characters.
Unknown characters are preserved through their bytes. Actual vocabulary size
is `256 + min(K, available non-whitespace training characters)`. The baseline
char tokenizer retains its existing behavior of covering train and validation
characters. It is not a train-only/OOV baseline.

If K covers every training character, this comparison primarily measures the
additional byte vocabulary, changed IDs, and validation-only characters.
Several IPA streams have fewer than 256 distinct characters. Use a lower K
when you want fallback to handle more of their observed characters.

To use a fixed whitelist, set `FALLBACK_CHARS_FILE=/absolute/path/characters.txt`.
It must contain one Unicode character per line, with no duplicates. Multi-character
phonemes/tokens are rejected so this experiment remains character-level. The
same file applies to every selected stream; use `RUN_FILTER` to run streams with
different whitelist files separately. A lower-level alternative is
`COMPARISON_TOKENIZER=byte`, which compares char against pure bytes with a fixed
256-token vocabulary.

## On the other machine

Clone the feature branch (replace the directory name if desired):

```bash
git clone --branch feat/cjk-char-byte-fallback-metrics --single-branch \
  https://github.com/xinyixuu/nanoGPT.git
cd nanoGPT
```

Create/activate your Python environment, install CUDA-compatible `torch` and
`torchaudio` for that machine, then install repository dependencies:

```bash
python3 -m pip install -r requirements_cpu.txt
python3 -m pip install pytest
# Ubuntu/Debian: Korean IPA conversion needs espeak-ng.
sudo apt-get install espeak-ng
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())'
```

`requirements_cpu.txt` contains the general Python dependencies and IPA
converters; GPU-specific optional packages are not required for this demo.
Use `DTYPE=float32` or `float16` if needed for your GPU. Supply your three
original UTF-8 text corpora explicitly; they are not uploaded to GitHub:

```bash
export PYTHON_BIN="$(command -v python3)"
export JA_TEXT=/path/to/ja.txt
export KO_TEXT=/path/to/ko.txt
export ZH_CN_TEXT=/path/to/zh_cn.txt
export FALLBACK_CHAR_LIMIT=256

# Optionally select one physical GPU.
export CUDA_VISIBLE_DEVICES=0

bash demos/cjk_ipa_char_byte_fallback_compare.sh prepare
bash demos/cjk_ipa_char_byte_fallback_compare.sh train
# Or run both stages: bash demos/cjk_ipa_char_byte_fallback_compare.sh all
```

For pure byte comparison:

```bash
COMPARISON_TOKENIZER=byte bash demos/cjk_ipa_char_byte_fallback_compare.sh all
```

Use distinct artifact roots for different K values, whitelists, seeds or model
settings. For example:

```bash
FALLBACK_CHAR_LIMIT=64 \
CJK_DATA_ROOT="$PWD/data/cjk_ipa_fallback_k64" \
CJK_OUT_ROOT="$PWD/out/cjk_ipa_fallback_k64" \
CJK_LOG_ROOT="$PWD/logs/cjk_ipa_fallback_k64" \
bash demos/cjk_ipa_char_byte_fallback_compare.sh all
```

Preparation fingerprints include the source files, tokenizer code, K and fixed
whitelist contents. Cached datasets are rebuilt when those inputs change.
Training does not automatically overwrite or resume partial runs. Use a fresh
output directory to collect a complete metric history.

## Retrain the existing char-BPE experiment with new metrics

Existing datasets can be reused on the current machine. Keep the old outputs
and point training at new output/log directories:

```bash
CJK_OUT_ROOT="$PWD/out/cjk_ipa_charbpe_target_metrics" \
CJK_LOG_ROOT="$PWD/logs/cjk_ipa_charbpe_target_metrics" \
bash demos/cjk_ipa_charbpe_compare.sh train
```

On a fresh machine, supply the three corpus paths and run `all` first. The new
metrics cannot be recovered for earlier evaluation snapshots from old CSV
losses alone; full metric histories require retraining.

## Outputs and interpretation

For a detached run with a status file, use the launcher (it invokes `nohup`):

```bash
# Current machine: reuse prepared char-BPE datasets and retrain with new metrics.
bash demos/run_cjk_nohup.sh charbpe train

# Fresh machine: provide the three corpus paths and prepare + train byte fallback.
JA_TEXT=/path/to/ja.txt KO_TEXT=/path/to/ko.txt ZH_CN_TEXT=/path/to/zh_cn.txt \
bash demos/run_cjk_nohup.sh char_byte_fallback all
```

The default char-BPE log root is `logs/cjk_ipa_charbpe_target_metrics`.
The byte-fallback launch uses `logs/cjk_ipa_char_byte_fallback_target_metrics`.
Monitor a char-BPE run with:

```bash
tail -f logs/cjk_ipa_charbpe_target_metrics/nohup.log
cat logs/cjk_ipa_charbpe_target_metrics/status
cat logs/cjk_ipa_charbpe_target_metrics/pid
```

`status` is `RUNNING` while active, `0` when the whole selected pipeline succeeds,
and `1` when it exits with an error. `exit_code` retains the original exit code;
`started_at` and `finished_at` record UTC timestamps. A successful launcher exit
only means the job was started: use the status file to check experiment completion.
Each launch reserves its log root using `.nohup-run`; choose a new log/output root
for another attempt. Normal termination signals are recorded as failure; SIGKILL
or a machine shutdown cannot run the exit handler.

The original `ja.txt`, `ko.txt` and `zh_cn.txt` are external inputs and are not in
GitHub. Copy them separately to the other machine, for example from this machine:

```bash
scp /home/xinyixu/ja.txt /home/xinyixu/ko.txt /home/xinyixu/zh_cn.txt \
  YOUR_USER@YOUR_HOST:/absolute/path/to/corpora/
```

Create the destination directory beforehand, then set the three input variables
to those paths. Run `all` to regenerate IPA and tokenized datasets on that machine.
Alternatively, transfer the prepared dataset tree and use `train` with its
`CJK_DATA_ROOT`; training-only execution does not read the three original files.

Each run writes `full_config.json`, `ckpt.pt`, `best_val_loss_and_iter.txt` and
`per_token_metrics/`. The report directory includes detailed and summary CSVs,
six added probability/rank overview/history HTML pages, and eight static PNG
sort orders per evaluation. The three new CSV columns are:

- `avg_target_probability`: target softmax probability, higher is better.
- `avg_target_rank`: one-based rank, lower is better; ties share the best rank.
- `avg_left_probability`: probability mass strictly ahead of the target,
  lower is better; the target and tied competitors are excluded.

All three are validation-only means over positions with that target token.
Tokens not sampled in validation have `NaN`. Padding targets (`-1`) are ignored.
`per_token_summary.csv` describes the distribution across populated token IDs.

`comparison_summary.tsv` adds `val_avg_target_probability`, `val_avg_target_rank`
and `val_avg_left_probability` at each run's **best validation-loss iteration**.
These three aggregates are weighted by `val_eval_count`, so they represent
sampled validation positions rather than an unweighted average of token IDs.
They should not be interpreted as tokenizer-independent measures: vocabularies,
token units and context spans differ. BPB is the primary cross-tokenizer score
within the same representation. Original and IPA BPB use different byte streams
and cannot directly establish which representation is better. The existing
Chinese IPA converter has known mixed-number/Hanzi issues.

Model depth/width and processed token counts match across runs. Parameter counts,
processed raw bytes and FLOPs are not matched. This is a single-seed experiment.
Reports use Plotly CDN for interactive views; CSVs and static PNGs work offline.

## Git branch and preservation

The feature branch contains both demos, core metric code, tests and this guide.
The old pipeline and its three new metrics remain available. Commit code paths
explicitly; do not use `git add .` to include generated data accidentally.
Datasets, checkpoints, logs and the local results index are kept on the original
machine and are not needed to clone the code branch.

For subsequent code changes:

```bash
git switch feat/cjk-char-byte-fallback-metrics
git add demos/cjk_ipa_charbpe_compare.sh \
  demos/cjk_ipa_char_byte_fallback_compare.sh \
  data/template/utils/build_char_fallback_vocab.py \
  utils/cjk_comparison_metrics.py tests/test_cjk_fallback_pipeline.py \
  documentation/CJK_Tokenizer_Experiments.md \
  README.md utils/per_token_metrics.py utils/per_token_html.py \
  utils/per_token_static.py tests/test_per_token_metrics.py
git commit -m 'feat(experiments): add CJK char byte-fallback comparison with metrics'
git push -u origin feat/cjk-char-byte-fallback-metrics
```
