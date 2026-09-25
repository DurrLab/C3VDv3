#!/usr/bin/env python3
"""Precompute Gemma + embeddings-processor outputs for a small prompt set.

Writes one ``.pt`` per positive prompt to ``--output-dir/{slug(prompt)}.pt`` with
keys matching what ``inference_edge_ic.load_cached_condition`` expects:

  - ``video_prompt_embeds``, ``audio_prompt_embeds`` — positive embeddings
  - ``video_neg_embeds``,   ``audio_neg_embeds``   — negative embeddings
  - ``prompt``, ``negative_prompt`` — strings (audit)

All tensors are saved without a batch dimension (the loader re-adds it with
``unsqueeze(0)``).

Usage::

    python sim2real/scripts/precompute_validation_prompts.py \\
        --checkpoint /path/to/base_model.safetensors \\
        --text-encoder-path /path/to/gemma \\
        --output-dir /tmp/prompt_cache \\
        --negative-prompt "worst quality, ..." \\
        --prompts "Real Colonoscopy Image, White Light Imaging" \\
                  "Real Colonoscopy Image, Narrow Band Imaging"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _squeeze_batch(t: torch.Tensor) -> torch.Tensor:
    if t.ndim >= 1 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-encoder-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=str, nargs="+", required=True)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-text-encoder-in-8bit", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Gemma from {args.text_encoder_path} "
          f"(8bit={args.load_text_encoder_in_8bit})...")
    text_encoder = load_text_encoder(
        gemma_model_path=str(args.text_encoder_path),
        device="cpu",
        dtype=torch.bfloat16,
        load_in_8bit=args.load_text_encoder_in_8bit,
    )
    print(f"Loading embeddings processor from {args.checkpoint}...")
    embeddings_processor = load_embeddings_processor(
        checkpoint_path=str(args.checkpoint),
        device="cpu",
        dtype=torch.bfloat16,
    )

    device = torch.device(args.device)
    text_encoder.to(device)
    embeddings_processor.to(device)

    def encode(prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        hs, mask = text_encoder.encode(prompt)
        out = embeddings_processor.process_hidden_states(hs, mask)
        return out.video_encoding.cpu(), out.audio_encoding.cpu()

    with torch.inference_mode():
        print(f"Encoding negative prompt: {args.negative_prompt!r}")
        neg_video, neg_audio = encode(args.negative_prompt)

        unique_prompts = list(dict.fromkeys(args.prompts))  # preserve order, dedup
        for prompt in unique_prompts:
            print(f"Encoding positive prompt: {prompt!r}")
            pos_video, pos_audio = encode(prompt)
            payload = {
                "video_prompt_embeds": _squeeze_batch(pos_video),
                "audio_prompt_embeds": _squeeze_batch(pos_audio),
                "video_neg_embeds": _squeeze_batch(neg_video),
                "audio_neg_embeds": _squeeze_batch(neg_audio),
                "prompt": prompt,
                "negative_prompt": args.negative_prompt,
            }
            out_path = args.output_dir / f"{_slug(prompt)}.pt"
            torch.save(payload, out_path)
            print(f"  saved -> {out_path}")

    print(f"\nDone. Wrote {len(unique_prompts)} cache file(s) to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
