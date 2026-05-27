# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end example script for the inference pipeline."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from alpamayo_r1 import helper
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo_r1.jetson_compat import apply_jetson_inference_compat, patch_typing_self
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--clip-id", default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--sample-path", default=os.environ.get("ALPAMAYO_SAMPLE_PATH"))
    parser.add_argument("--save-sample-path", default=os.environ.get("ALPAMAYO_SAVE_SAMPLE_PATH"))
    parser.add_argument("--prepare-sample-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("ALPAMAYO_ATTN_IMPLEMENTATION", "flash_attention_2"),
    )
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--device-map-cuda", action="store_true")
    parser.add_argument(
        "--jetson-compat",
        action="store_true",
        default=os.environ.get("ALPAMAYO_JETSON_COMPAT") == "1",
    )
    parser.add_argument("--log-path", default=os.environ.get("ALPAMAYO_LOG_PATH"))
    return parser.parse_args()


def setup_logging(log_path: str | None) -> None:
    if not log_path:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_file = path.open("w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print("log_path", path.resolve())


def get_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def cuda_mem(prefix: str) -> None:
    if not torch.cuda.is_available():
        print(prefix, "cuda unavailable")
        return

    free, total = torch.cuda.mem_get_info()
    print(
        prefix,
        "cuda_mem_free_gib",
        round(free / 1024**3, 2),
        "cuda_mem_total_gib",
        round(total / 1024**3, 2),
        "allocated_gib",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
    )


def load_data(args: argparse.Namespace) -> dict:
    if args.sample_path:
        print("Loading sample:", args.sample_path)
        return torch.load(args.sample_path, map_location="cpu", weights_only=False)

    patch_typing_self()
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    print(f"Loading dataset for clip_id: {args.clip_id}...")
    data = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)
    print("Dataset loaded.")

    if args.save_sample_path:
        save_path = Path(args.save_sample_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, save_path)
        print("Saved sample:", save_path)

    return data


def load_model(args: argparse.Namespace, dtype: torch.dtype) -> AlpamayoR1:
    config = AlpamayoR1Config.from_pretrained(args.model_id)
    config.attn_implementation = args.attn_implementation
    config.model_dtype = args.dtype
    print("attn_implementation", config.attn_implementation)

    kwargs = {
        "config": config,
        "dtype": dtype,
    }
    if args.low_cpu_mem_usage:
        kwargs["low_cpu_mem_usage"] = True
    if args.device_map_cuda:
        kwargs["device_map"] = {"": args.device}

    model = AlpamayoR1.from_pretrained(args.model_id, **kwargs)
    if not args.device_map_cuda:
        model = model.to(args.device)
    return model.eval()


def main() -> None:
    started = time.time()
    args = parse_args()
    setup_logging(args.log_path)

    if args.jetson_compat:
        apply_jetson_inference_compat()

    dtype = get_dtype(args.dtype)
    print("HF_HOME", os.environ.get("HF_HOME"))
    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
    print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    cuda_mem("start")

    data = load_data(args)
    print("sample keys", sorted(data.keys()))
    print("image_frames", tuple(data["image_frames"].shape), data["image_frames"].dtype)
    if args.prepare_sample_only:
        print("Prepared sample only; skipping model inference.")
        return

    messages = helper.create_message(data["image_frames"].flatten(0, 1))

    print("Loading model:", args.model_id)
    model = load_model(args, dtype)
    cuda_mem("after_model")

    print("Loading processor")
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, args.device)
    cuda_mem("after_inputs")

    print("Running rollout")
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(42)
    autocast_device = torch.device(args.device).type
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=dtype):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length,
            return_extra=True,
        )
    cuda_mem("after_rollout")

    print("pred_xyz", tuple(pred_xyz.shape), pred_xyz.dtype, pred_xyz.device)
    print("pred_rot", tuple(pred_rot.shape), pred_rot.dtype, pred_rot.device)
    print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    print("minADE:", float(min_ade), "meters")
    print("elapsed_sec", round(time.time() - started, 1))


if __name__ == "__main__":
    main()
