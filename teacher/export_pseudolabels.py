#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor, InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "universal-image-restoration"))

import options as option  # noqa: E402
from data import create_dataset  # noqa: E402

import open_clip  # noqa: E402

from teacher import DegradationCoder, MaskHead  # noqa: E402


def build_clip_transform(resolution: int = 224) -> Compose:
    return Compose(
        [
            Resize(resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(resolution),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def load_prompts(prompt_file: Path) -> List[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        prompts = list(data.values())
    elif isinstance(data, Sequence):
        prompts = list(data)
    else:
        raise ValueError("Prompt file must contain a JSON list or dictionary of prompts.")
    return prompts


def preprocess_clip_batch(batch: torch.Tensor, transform: Compose) -> torch.Tensor:
    images = []
    for img in batch:
        img = img.clamp(0, 1)
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        images.append(transform(pil))
    return torch.stack(images, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DACLiP pseudo-labels (masks + z_d).")
    parser.add_argument("--config", type=str, required=True, help="Path to options YAML file.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset key to use from config (train/val/test).")
    parser.add_argument("--output", type=str, required=True, help="Directory to store pseudo-label outputs.")
    parser.add_argument("--prompts", type=str, default=str(ROOT / "prompts/degradation_prompts.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--confidence-delta", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip-model", type=str, default="daclip_ViT-B-32")
    parser.add_argument("--use-crf", action="store_true", help="Enable differentiable guided CRF refinement.")
    parser.add_argument("--crf-iters", type=int, default=5)
    parser.add_argument("--crf-kernel", type=int, default=5)
    parser.add_argument("--crf-sigma", type=float, default=3.0)
    parser.add_argument("--crf-edge", type=float, default=5.0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    opt = option.parse(args.config, is_train=False)
    opt = option.dict_to_nonedict(opt)
    opt["dist"] = False

    dataset_key = args.dataset
    if dataset_key is None:
        dataset_key = "train" if "train" in opt["datasets"] else "test"
    if dataset_key not in opt["datasets"]:
        raise KeyError(f"Dataset key '{dataset_key}' not found in config options.")

    dataset_opt = opt["datasets"][dataset_key]
    dataset = create_dataset(dataset_opt)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    clip_model, preprocess = open_clip.create_model_from_pretrained(
        args.clip_model, pretrained=opt["path"]["daclip"]
    )
    clip_model = clip_model.to(args.device).eval()
    transform = preprocess or build_clip_transform()

    prompts = load_prompts(Path(args.prompts))
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    text_tokens = tokenizer(prompts).to(args.device)

    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_tokens, normalize=True)

    mask_head = MaskHead(
        temperature=args.temperature,
        alpha=args.alpha,
        default_threshold=args.threshold,
        confidence_delta=args.confidence_delta,
        use_crf=args.use_crf,
        crf_iterations=args.crf_iters,
        crf_spatial_kernel=args.crf_kernel,
        crf_spatial_sigma=args.crf_sigma,
        crf_edge_weight=args.crf_edge,
    )
    coder = DegradationCoder()
    grid_size = tuple(clip_model.visual.grid_size)

    for batch in dataloader:
        lq = batch["LQ"].to(args.device)
        clip_inputs = preprocess_clip_batch(lq, transform).to(args.device)

        with torch.no_grad():
            _, hiddens = clip_model.visual_control(clip_inputs, output_hiddens=True)

        tokens = hiddens[-1]
        if tokens.dim() == 3 and tokens.shape[1] == lq.shape[0]:
            tokens = tokens.permute(1, 0, 2)
        tokens = tokens[:, 1:, :]

        mask_out = mask_head(
            tokens,
            text_embeddings,
            grid_size=grid_size,
            image=lq,
        )
        code = coder(mask_out, spatial_shape=grid_size)

        paths = batch["LQ_path"]
        if isinstance(paths, (str, Path)):
            paths = [paths]
        for idx, path in enumerate(paths):
            sample = {
                "masks": mask_out.masks[idx].cpu(),
                "scores": mask_out.scores[idx].cpu(),
                "probs": mask_out.probs[idx].cpu(),
                "raw_probs": mask_out.raw_probs[idx].cpu(),
                "intensity": mask_out.intensity[idx].cpu(),
                "avg_intensity": mask_out.avg_intensity[idx].cpu(),
                "global_prob": mask_out.global_prob[idx].cpu(),
                "confidence": mask_out.confidence[idx].cpu(),
                "used_crf": mask_out.used_crf,
                "z_d": code.code[idx].cpu(),
                "logits": code.logits[idx].cpu(),
                "confidence_scale": code.confidence_scale[idx].cpu(),
                "metadata": {
                    "source_path": path,
                    "prompts": prompts,
                    "temperature": args.temperature,
                    "threshold": args.threshold,
                    "alpha": args.alpha,
                    "confidence_delta": args.confidence_delta,
                },
            }
            rel_path = Path(path)
            dataset_root = Path(dataset_opt["dataroot_LQ"]).resolve()
            try:
                rel_path = rel_path.resolve().relative_to(dataset_root)
            except Exception:
                rel_path = rel_path.name
            rel_path = Path(rel_path).with_suffix(".pt")
            save_path = output_dir / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sample, save_path)


if __name__ == "__main__":
    main()
