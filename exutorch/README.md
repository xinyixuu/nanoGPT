# 🚀 Exutorch Setup Script

This folder contains a scripts (and configuration files) automating the setup up
of PyTorch's Exutorch.

These steps and files are adapted from the Exutorch documentation [ExecuTorch Getting Started
Guide](https://pytorch.org/executorch/stable/llm/getting-started.html),
specifically showing how to utilize a nanoGPT model (essentially a custom
pytorch language model) and export to hardware targets.

The binaries and files created allow for inference and profiling directly on cpu
and specific hardware targets like Android and other mobile platforms.

---

## 📁 Description of Files

### Setup & Build Files
- **`setup_exutorch.sh`**
  🛠️ *Setup Script*: Runs all necessary setup steps (including dependency installation via `install_requirements.sh`) to prepare your environment.
- **`CMakeLists.txt`**
  📝 *CMake Configuration*: Defines the build settings for the basic NanoGPT runner.

### Core Export & Runtime Files
- **`export_nanogpt.py`**
  📦 *Model Exporter*: Exports the GPT model to an ExecuTorch‑compatible `.pte` file **without** backend delegation or quantization.
- **`main.cpp`**
  🎯 *Model Runner*: Implements a simple loop to load the exported model, tokenize the prompt, generate tokens, and display the output.
- **`basic_sampler.h`**
  🎲 *Sampler*: Provides a bare‑bones sampler that selects the next token from model logits (e.g., using greedy sampling).
- **`basic_tokenizer.h`**
  🔤 *Tokenizer*: Implements minimal encode/decode methods for converting between text and token IDs using a fixed vocabulary JSON.
- **`model.py`**
  🤖 *NanoGPT Model*: Contains Karpathy’s NanoGPT model definition, adapted as a minimal GPT‑2 architecture for demonstration.
- **`vocab.json`**
  📚 *Vocabulary File*: Maps subwords (or tokens) to integer token IDs for the tokenizer.

### Generated Artifacts
- **`nanogpt.pte`**
  ⚙️ *ExecuTorch Program*: The compiled ExecuTorch program file generated from the PyTorch model, ready to be loaded and executed at runtime.

---

## 📱 Files for XNNPACK Mode (Android Targets)

To optimize for Android devices with XNNPACK acceleration, the following additional files are provided:

- **`xnnpack_mode.sh`**
  🔄 *XNNPACK Mode Switcher*: Cleans and rebuilds the `nanogpt_runner` binary for XNNPACK mode.
- **`CMakeLists_XNNPACK.txt`**
  🏗️ *XNNPACK CMake Configuration*: The CMake file configured for Android targets using XNNPACK.
- **`export_nanogpt_xnnpack.py`**
  🛠️ *XNNPACK Exporter*: Exports the GPT model with PyTorch PT2E quantization steps before delegating to the XNNPACK backend for further optimization.

---

## 🚀 Getting Started

Follow these steps to set up and run your NanoGPT model:

0. **Initial Setup**
   Open your terminal and run:
   ```bash
   bash setup_exutorch.sh
   ```
   This script sets up your environment and installs required dependencies.

1. **Run the Basic Mode**
   Execute the NanoGPT runner:
   ```bash
   ./nanogpt_runner
   ```
   Watch as the model loads and processes your prompt.

2. **Switch to XNNPACK Mode**
   To enable XNNPACK optimizations (recommended for Android targets), run:
   ```bash
   bash xnnpack_mode.sh
   ```
   This script swaps key files and rebuilds the binary for XNNPACK mode.
3. **Run the XNNPACK-Optimized Mode**
   Execute the runner again to benefit from XNNPACK acceleration:
   ```bash
   ./nanogpt_runner
   ```

---

## 🛠️ Troubleshooting

**Issue:** Difficulties during the initial `install_requirements.sh` step in `setup_exutorch.sh` (e.g., hanging due to buck2d processes).

**Solution:**
Kill any lingering `buck2d` processes before re-running the setup script:
```bash
ps aux | grep buck2d
```

Then killing appropriate process id's for the lingering `buck2d` processes.
```bash
kill <process-id>
```

Then clean and re-run:
```bash
rm -rf et-nanogpt/
bash setup_exutorch.sh
```

---

## 🔗 References

These scripts are built around instructions from [ExecuTorch Getting Started
Guide](https://pytorch.org/executorch/stable/llm/getting-started.html), allowing
for smoother installation.

