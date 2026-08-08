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
                  deblur, denoise, edsr,
                  use_deblur: bool, use_denoise: bool, use_scale: bool,
                  deblur_factor: float, denoise_factor: float):

    img = Image.open(input_path)

    # Denoise
    if use_denoise and denoise is not None:
        denoised = denoise.process(img)
        img = ensure_rgb(denoised)

    # Deblur
    if use_deblur and deblur is not None:
        deblurred = deblur.process(img)
        img = ensure_rgb(deblurred)

    # Upscale
    if use_scale and edsr is not None:
        inputs = ImageLoader.load_image(img)
        preds = edsr(inputs)
        img = preds  # tensor

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_scale and edsr is not None:
        # EDSR output → tensor
        ImageLoader.save_image(img, str(output_path))
    else:
        # PIL Image
        img.save(str(output_path))


# -----------------------------
# Batch Runner
# -----------------------------

def run_batch(input_dir: Path, output_dir: Path, scale: int,
              use_deblur: bool, use_denoise: bool, use_scale: bool,
              deblur_factor: float, denoise_factor: float):

    print("Loading models...")

    # Conditional model loading
    deblur = DeblurProcessor(model_id="gopro_width64", model_dir="./models/", device="cuda") if use_deblur else None
    denoise = DenoiseProcessor(model_id="sidd_width64", model_dir="./models/", device="cuda") if use_denoise else None
    edsr = EdsrModel.from_pretrained("eugenesiow/edsr-light", scale=scale) if use_scale else None

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
            process_image(
                file, out_path, scale,
                deblur, denoise, edsr,
                use_deblur, use_denoise, use_scale,
                deblur_factor, denoise_factor
            )
        except Exception as e:
            print(f"   ERROR: {e}")
            continue

        print(f"   Saved → {out_path}\n")

    # GPU cleanup
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch Image Enhancer")
    parser.add_argument("input", type=str, help="Input folder")
    parser.add_argument("output", type=str, help="Output folder")
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor (default: 2)")

    # Toggle flags
    parser.add_argument("--no-deblur", action="store_true", help="Disable deblurring")
    parser.add_argument("--no-denoise", action="store_true", help="Disable denoising")
    parser.add_argument("--no-scale", action="store_true", help="Disable upscaling")

    # Factors
    parser.add_argument("--deblur-factor", type=float, default=1.0, help="Deblur strength factor")
    parser.add_argument("--denoise-factor", type=float, default=1.0, help="Denoise strength factor")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    run_batch(
        input_dir, output_dir, args.scale,
        use_deblur=not args.no_deblur,
        use_denoise=not args.no_denoise,
        use_scale=not args.no_scale,
        deblur_factor=args.deblur_factor,
        denoise_factor=args.denoise_factor
    )


if __name__ == "__main__":
    main()
