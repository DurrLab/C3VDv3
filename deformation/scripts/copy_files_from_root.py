#!/usr/bin/env python3

import argparse
import random
import shutil
from pathlib import Path
import os

import yaml


REQUIRED_REFERENCE_FILES = ("pose.txt",)
MASK_SOURCE = Path(__file__).resolve().parents[1] / "utils" / "mask.png"
CONFIG_INI_SOURCE = Path(__file__).resolve().parents[1] / "utils" / "config.ini"

MATERIAL_PRESETS = [
	{
		"Ka": (0.18, 0.12, 0.11),
		"Kd": (0.82, 0.48, 0.42),
		"Ks": (0.45, 0.40, 0.36),
		"Ns": 45.0,
	},
	{
		"Ka": (0.20, 0.13, 0.11),
		"Kd": (0.88, 0.55, 0.46),
		"Ks": (0.35, 0.32, 0.30),
		"Ns": 60.0,
	},
	{
		"Ka": (0.22, 0.16, 0.15),
		"Kd": (0.92, 0.66, 0.60),
		"Ks": (0.30, 0.28, 0.26),
		"Ns": 70.0,
	},
	{
		"Ka": (0.14, 0.07, 0.06),
		"Kd": (0.62, 0.24, 0.20),
		"Ks": (0.50, 0.42, 0.36),
		"Ns": 35.0,
	},
	{
		"Ka": (0.20, 0.12, 0.09),
		"Kd": (0.86, 0.43, 0.32),
		"Ks": (0.38, 0.32, 0.28),
		"Ns": 50.0,
	},
]


def parse_args():
	parser = argparse.ArgumentParser(
		description="Copy fixed reference files from reference_dir to c3vd_input_path/<config_stem>."
	)
	parser.add_argument("--config", required=True, help="Path to YAML config")
	parser.add_argument(
		"--reference-base",
		help="Root directory containing reference_dir",
	)
	return parser.parse_args()


def load_config(config_path: Path) -> dict:
	with config_path.open("r", encoding="utf-8") as f:
		config = yaml.safe_load(f)

	if not isinstance(config, dict):
		raise ValueError(f"Config must be a YAML mapping: {config_path}")

	for key in ("geometry", "reference_dir", "c3vd_input_path"):
		if config.get(key) is None:
			raise ValueError(f"Missing required config key: {key}")
	return config


def write_material_preset(dst: Path):
	preset = random.choice(MATERIAL_PRESETS)
	with dst.open("w", encoding="utf-8") as f:
		f.write("newmtl material_0\n")
		ka = preset["Ka"]
		kd = preset["Kd"]
		ks = preset["Ks"]
		ns = preset["Ns"]
		f.write(f"Ka {ka[0]:.2f} {ka[1]:.2f} {ka[2]:.2f}\n")
		f.write(f"Kd {kd[0]:.2f} {kd[1]:.2f} {kd[2]:.2f}\n")
		f.write(f"Ks {ks[0]:.2f} {ks[1]:.2f} {ks[2]:.2f}\n")
		f.write(f"Ns {ns:.1f}\n")
		f.write("illum 2\n")
	print(f"Wrote {dst} with random material preset (Ns={preset['Ns']:.1f})")


def copy_required_files(source_dir: Path, target_dir: Path):
	if not source_dir.is_dir():
		raise FileNotFoundError(f"Missing source reference directory: {source_dir}")
	if not MASK_SOURCE.is_file():
		raise FileNotFoundError(f"Missing mask source file: {MASK_SOURCE}")
	if not CONFIG_INI_SOURCE.is_file():
		raise FileNotFoundError(f"Missing config.ini source file: {CONFIG_INI_SOURCE}")

	target_dir.mkdir(parents=True, exist_ok=True)

	for filename in REQUIRED_REFERENCE_FILES:
		src = source_dir / filename
		dst = target_dir / filename
		if not src.is_file():
			raise FileNotFoundError(f"Missing source file: {src}")
		shutil.copy2(src, dst)
		print(f"Copied {src} -> {dst}")

	config_dst = target_dir / "config.ini"
	shutil.copy2(CONFIG_INI_SOURCE, config_dst)
	print(f"Wrote {config_dst}")

	mask_dst = target_dir / "mask.png"
	shutil.copy2(MASK_SOURCE, mask_dst)
	print(f"Copied {MASK_SOURCE} -> {mask_dst}")

	write_material_preset(target_dir / "model.mtl")

def main():
	args = parse_args()
	config_path = Path(args.config)
	if not config_path.is_file():
		raise FileNotFoundError(f"Missing config file: {config_path}")

	config = load_config(config_path)

	config_stem = config_path.stem
	reference_dir = Path(str(config["reference_dir"]))
	c3vd_input_path = Path(str(config["c3vd_input_path"]))

	if reference_dir.is_absolute():
		source_dir = Path(reference_dir)
	else:	
		source_dir = Path(args.reference_base) / reference_dir
	target_dir = Path(c3vd_input_path) / config_stem

	copy_required_files(source_dir, target_dir)
	render_dir = target_dir / "render"
	if not os.path.exists(render_dir):
		render_dir.mkdir(parents=True, exist_ok=True)
		print(f"Created render directory: {render_dir}")


if __name__ == "__main__":
	main()
