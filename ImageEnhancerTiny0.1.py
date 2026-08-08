import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from super_image import EdsrModel, ImageLoader
from nafnetlib import DeblurProcessor, DenoiseProcessor

# -----------------------------
# Helpers
# -----------------------------

def is_image_file(path: Path):
    return path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]


def ensure_rgb(img):
    return img.convert("RGB")


# -----------------------------
# Processing Pipeline
# -----------------------------

def process_image(input_path: Path, output_path: Path, scale: int,
                  deblur: DeblurProcessor, denoise: DenoiseProcessor, edsr: EdsrModel):

    # Load
    img = Image.open(input_path)

    # Deblur
    deblurred = deblur.process(img)
    deblurred = ensure_rgb(deblurred)

    # Denoise
    denoised = denoise.process(deblurred)
    denoised = ensure_rgb(denoised)

    # Upscale
    inputs = ImageLoader.load_image(denoised)
    preds = edsr(inputs)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ImageLoader.save_image(preds, str(output_path))


# -----------------------------
# Batch Runner
# -----------------------------

def run_batch(input_dir: Path, output_dir: Path, scale: int):

    # Initialize processors
    print("Loading models...")
    deblur = DeblurProcessor(model_id="gopro_width64", model_dir="./models/", device="cuda")
    denoise = DenoiseProcessor(model_id="sidd_width64", model_dir="./models/", device="cuda")
    edsr = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale)

    # Collect files
    files = [p for p in input_dir.rglob("*") if is_image_file(p)]
    total = len(files)

    if total == 0:
        print("No images found.")
        return

    print(f"Found {total} images.\n")

    # Process
    for idx, file in enumerate(files, start=1):
        rel = file.relative_to(input_dir)
        out_path = output_dir / rel

        print(f"[{idx}/{total}] Processing: {rel}")

        try:
            process_image(file, out_path, scale, deblur, denoise, edsr)
        except Exception as e:
            print(f"   ERROR: {e}")
            continue

        print(f"   Saved → {out_path}\n")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch Image Enhancer")
    parser.add_argument("input", type=str, help="Input folder")
    parser.add_argument("output", type=str, help="Output folder")
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor (default: 2)")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    run_batch(input_dir, output_dir, args.scale)


if __name__ == "__main__":
    main()
