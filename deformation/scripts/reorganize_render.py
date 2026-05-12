import argparse
from pathlib import Path
from typing import Dict


DEST_DIRS = {
	"*_depth.tiff": "depth",
	"*_diffuse.png": "diffuse",
	"*_normals.tiff": "normals",
	"*_occlusion.png": "occlusion",
	"*_flow.tiff": "optical_flow",
}


def ensure_structure(render_dir: Path) -> None:
	for subdir in ["depth", "diffuse", "normals", "occlusion", "optical_flow"]:
		(render_dir / subdir).mkdir(parents=True, exist_ok=True)


def move_pattern_files(render_dir: Path) -> Dict[str, int]:
	moved_counts: Dict[str, int] = {}

	for pattern, subdir in DEST_DIRS.items():
		dst = render_dir / subdir
		moved = 0
		for src in sorted(render_dir.glob(pattern)):
			target = dst / src.name
			src.replace(target)
			moved += 1
		moved_counts[subdir] = moved

	return moved_counts


def delete_top_level_files(render_dir: Path) -> int:
	deleted = 0
	for entry in render_dir.iterdir():
		if entry.is_file():
			if entry == render_dir / "coverage_mesh.obj" or entry == render_dir / "pose.txt" or str(entry.name).endswith(".bin"):
				continue
			entry.unlink()
			deleted += 1
	return deleted


def count_files(folder: Path) -> int:
	return sum(1 for p in folder.iterdir() if p.is_file())


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Reorganize renders for one geometry.")
	parser.add_argument("path", help="Path to the render directory")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	render_dir = Path(args.path) / "render"

	if not render_dir.exists() or not render_dir.is_dir():
		raise RuntimeError(f"Render directory not found: {render_dir.resolve()}")

	ensure_structure(render_dir)
	moved_counts = move_pattern_files(render_dir)
	deleted_top_level = delete_top_level_files(render_dir)

	print("Moved files:")
	for subdir in ["depth", "diffuse", "normals", "occlusion", "optical_flow"]:
		print(f"  {subdir}: {moved_counts.get(subdir, 0)}")

	print(f"Deleted remaining top-level files: {deleted_top_level}")

	print("Final counts:")
	for subdir in ["depth", "diffuse", "normals", "occlusion", "optical_flow"]:
		print(f"  {subdir}: {count_files(render_dir / subdir)}")


if __name__ == "__main__":
	main()
