from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import sudoku_utils as sutils


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(f"No Sudoku images found in: {input_dir}")

    saved = 0
    failures = 0

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skipped unreadable image: {image_path}")
            failures += 1
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = sutils.resize_and_maintain_aspect_ratio(
            input_image=image_rgb,
            new_width=1000,
        )

        try:
            cells, _, _ = sutils.get_valid_cells_from_image(image_rgb)
        except Exception as exc:
            print(f"Extraction failed for {image_path.name}: {exc}")
            failures += 1
            continue

        image_saved = 0
        for cell_index, cell in enumerate(cells):
            if not cell["contains_digit"]:
                filename = (
                    f"{image_path.stem}_cell_{cell_index:02d}_{saved:06d}.png"
                )
                cv2.imwrite(str(output_dir / filename), cell["img"])
                saved += 1
                image_saved += 1

        print(f"{image_path.name}: saved {image_saved} blank cells")

    print("\nCollection complete")
    print(f"Blank cells saved: {saved}")
    print(f"Failed images: {failures}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect real blank Sudoku cells for the ten-class model."
    )
    parser.add_argument(
        "--input_dir",
        default="data/sudoku_images",
        type=Path,
    )
    parser.add_argument(
        "--output_dir",
        default="data/blank_cells",
        type=Path,
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
