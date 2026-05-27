# Alpamayo 1 runbook and LLaVA comparison

This note records the path that was verified in this repository and explains,
from the code, how Alpamayo differs from a typical LLaVA-style VLM.

## 1. What was verified locally

The Jetson inference path has been verified with the current repository:

```bash
scripts/run_inference_jetson.sh
```

Observed result:

```text
torch 2.8.0 cuda True 12.6
device Orin
pred_xyz (1, 1, 1, 64, 3) torch.float32 cuda:0
pred_rot (1, 1, 1, 64, 3, 3) torch.float32 cuda:0
minADE: 1.1008784770965576 meters
elapsed_sec 67.8
```

The log is written to:

```bash
.codex/run_inference_jetson_repo.log
```

Important local context:

- `.codex/alpamayo_sample.pt` already exists and is used by the Jetson script.
- Model cache is expected under `/home/zhanghongyu/fsas/models/huggingface`.
- The repository's `.venv` is Python 3.12, but currently has CPU-only PyTorch.
- `jetson_venv` is the CUDA-capable Jetson environment.

## 2. Prerequisites

Hardware:

- NVIDIA GPU with enough memory. This repository was verified on Jetson Orin with about 61 GiB total CUDA memory.
- For the default non-Jetson path, the README expects a CUDA environment that can install and use `flash-attn`.

HuggingFace access:

- Request access to `nvidia/PhysicalAI-Autonomous-Vehicles`.
- Request access to `nvidia/Alpamayo-R1-10B`.
- Log in with a HuggingFace token before downloading data or weights.

Recommended environment variables:

```bash
export HF_HOME="$HOME/fsas/models/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
mkdir -p "$HF_HOME"
```

Then log in:

```bash
.venv/bin/hf auth login
```

If `hf` is not available in the environment:

```bash
uv add huggingface_hub
uv run --active hf auth login
```

## 3. Fast path: run inference with the prepared sample

Use this when `.codex/alpamayo_sample.pt` already exists.

```bash
ls -lh .codex/alpamayo_sample.pt
scripts/run_inference_jetson.sh
```

The script does the following:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/zhanghongyu/fsas/models/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

jetson_venv/bin/python src/alpamayo_r1/test_inference.py \
  --sample-path .codex/alpamayo_sample.pt \
  --attn-implementation sdpa \
  --jetson-compat \
  --low-cpu-mem-usage \
  --device-map-cuda
```

Expected successful signs:

- It prints `torch ... cuda True`.
- It loads `nvidia/Alpamayo-R1-10B`.
- It prints `Running rollout`.
- It prints `pred_xyz`, `pred_rot`, a Chain-of-Causation text, and `minADE`.

## 4. Prepare a sample from the gated dataset

Use this when `.codex/alpamayo_sample.pt` is missing or stale.

Make sure HuggingFace authentication works and the current shell has `HF_HOME`:

```bash
export HF_HOME="$HOME/fsas/models/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
.venv/bin/hf auth whoami
```

Generate only the sample, without loading the 10B model:

```bash
.venv/bin/python src/alpamayo_r1/test_inference.py \
  --save-sample-path .codex/alpamayo_sample.pt \
  --prepare-sample-only
```

This path uses `physical_ai_av.PhysicalAIAVDatasetInterface`, loads the configured
clip, extracts 4 cameras x 4 frames, and prepares:

- `image_frames`: `(N_cameras, num_frames, 3, H, W)`
- `ego_history_xyz`: `(1, 1, 16, 3)`
- `ego_history_rot`: `(1, 1, 16, 3, 3)`
- `ego_future_xyz`: `(1, 1, 64, 3)`
- `ego_future_rot`: `(1, 1, 64, 3, 3)`

If this fails with `401 Unauthorized`, the machine is not authenticated for the
gated dataset, or the token does not have access.

## 5. Full Jetson run after sample preparation

After the sample exists:

```bash
scripts/run_inference_jetson.sh
```

For a direct command:

```bash
PYTHONPATH="$PWD/src" HF_HOME="$HOME/fsas/models/huggingface" \
HF_ENDPOINT="https://hf-mirror.com" \
jetson_venv/bin/python src/alpamayo_r1/test_inference.py \
  --sample-path .codex/alpamayo_sample.pt \
  --attn-implementation sdpa \
  --jetson-compat \
  --low-cpu-mem-usage \
  --device-map-cuda \
  --log-path .codex/run_inference_jetson_repo.log
```

Why these flags matter on Jetson:

- `--attn-implementation sdpa` avoids the FlashAttention path.
- `--jetson-compat` applies local compatibility patches.
- `--low-cpu-mem-usage` reduces model loading memory pressure.
- `--device-map-cuda` loads the model directly onto CUDA.

## 6. Common failures

### `uv sync --frozen` fails on `flash-attn`

The default dependency list includes:

```toml
flash-attn>=2.8.3
```

On the checked machine, `.venv` had `torch 2.8.0+cpu`, so building `flash-attn`
failed. The error also reported that `CUDA_HOME` was not set.

For Jetson inference, use the verified `scripts/run_inference_jetson.sh` path,
which selects SDPA instead of FlashAttention.

For a clean non-Jetson CUDA setup, make sure:

- PyTorch is a CUDA build, not `+cpu`.
- `nvcc` is available.
- `CUDA_HOME` points to the CUDA installation, for example:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

### Dataset download fails with `401 Unauthorized`

The sample preparation path accesses:

```text
nvidia/PhysicalAI-Autonomous-Vehicles
```

This is gated. Request access and log in with a token that has permission.

### Model download is slow or fails

The model is large. Set:

```bash
export HF_HOME="$HOME/fsas/models/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
```

Then retry the Jetson script. The first successful run should populate the cache.

## 7. How Alpamayo works in this repository

The inference script is `src/alpamayo_r1/test_inference.py`.

High-level flow:

1. Load a sample from `.pt` or from the PhysicalAI AV dataset.
2. Build a multi-image chat message with `helper.create_message`.
3. Load `AlpamayoR1` from `nvidia/Alpamayo-R1-10B`.
4. Tokenize images and text with a Qwen3-VL processor.
5. Replace placeholder trajectory-history tokens with encoded ego-history tokens.
6. Let the VLM generate Chain-of-Causation reasoning until `<|traj_future_start|>`.
7. Use the generated VLM KV cache as context for an action expert.
8. Run flow-matching diffusion in action space.
9. Convert sampled actions into future xyz positions and rotations.

Key code locations:

- `src/alpamayo_r1/helper.py`: creates the multi-image driving prompt.
- `src/alpamayo_r1/models/base_model.py`: builds the Qwen3-VL backbone and adds trajectory tokens.
- `src/alpamayo_r1/models/alpamayo_r1.py`: combines VLM rollout, action expert, and diffusion sampling.
- `src/alpamayo_r1/action_space/unicycle_accel_curvature.py`: maps actions to physical trajectories.
- `src/alpamayo_r1/diffusion/flow_matching.py`: samples actions with Euler flow matching.
- `src/alpamayo_r1/load_physical_aiavdataset.py`: loads cameras and ego motion from the dataset.

## 8. Alpamayo vs. LLaVA

LLaVA is usually a vision-language assistant: it maps image features into an LLM
and autoregressively produces text. Alpamayo uses a VLM too, but the code here
turns it into a vision-language-action system for driving.

| Aspect | Typical LLaVA | Alpamayo in this repo |
|---|---|---|
| Main task | Image understanding and text response | Driving reasoning plus future trajectory prediction |
| Backbone style | Vision encoder + projector + LLM | Qwen3-VL conditional generation model plus action expert |
| Inputs | Usually one or a few images plus text | Multi-camera, multi-frame video tensors plus ego-motion history |
| Prompt | General visual instruction | Driving prompt with history-trajectory placeholders and CoC request |
| Extra tokens | Image tokens and normal text tokens | Image/text tokens plus trajectory special tokens and discrete trajectory vocabulary |
| Output type | Text tokens | Reasoning text plus continuous future trajectory |
| Action modeling | Not part of the standard design | Explicit action space: acceleration and curvature over 64 waypoints |
| Generation | Autoregressive text decoding | Autoregressive reasoning first, then diffusion sampling for actions |
| Physics/geometry | Usually none | Converts actions to xyz and rotation with a unicycle kinematic model |
| Dataset assumptions | General image-text instruction data | Physical AI autonomous driving clips, cameras, ego history, future ego motion |

The biggest architectural difference is the split after reasoning. Alpamayo first
uses the VLM to read visual context and produce Chain-of-Causation text. Then it
does not simply keep decoding future coordinates as plain text. Instead, it uses
the VLM's cached context to condition an expert transformer over future action
tokens. A flow-matching sampler repeatedly calls that expert to denoise an action
sequence.

In code, this happens in `AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout`:

- `self.vlm.generate(...)` produces reasoning tokens until `<|traj_future_start|>`.
- `prompt_cache = vlm_outputs.past_key_values` saves the VLM context.
- `self.action_in_proj(x, t)` embeds noisy action samples and diffusion time.
- `self.expert(...)` predicts hidden states for future action tokens using the VLM cache.
- `self.action_out_proj(...)` predicts the vector field/noise update.
- `self.diffusion.sample(...)` integrates the flow.
- `self.action_space.action_to_traj(...)` converts acceleration/curvature actions into trajectory tensors.

So, compared with LLaVA, Alpamayo is not just "VLM plus a different prompt". It
adds a driving-specific action representation, a trajectory tokenizer, a second
expert model, and a diffusion/flow-matching trajectory generator. The result is a
model that can still explain what it is doing in language, but its final product
is a physically structured driving trajectory rather than only text.
