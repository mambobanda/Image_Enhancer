import argparse
import gc
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
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
# Sharpen (DaVinci Resolve-style)
# -----------------------------

def sharpen_resolve_style(img: Image, amount: float = 1.0, radius: float = 1.0):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    blurred = cv2.GaussianBlur(img_cv, (0, 0), radius)
    high_pass = img_cv - blurred
    sharpened = img_cv + amount * high_pass

    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))


# -----------------------------
# Processing Pipeline
# -----------------------------

def process_image(
    input_path: Path,
    output_path: Path,
    scale: int,
    deblur,
    denoise,
    edsr,
    use_deblur: bool,
    use_denoise: bool,
    use_scale: bool,
    use_sharpen: bool,
    deblur_factor: float,
    denoise_factor: float,
    sharpen_amount: float,
    sharpen_radius: float
):

    img = Image.open(input_path)

    # Deblur
    if use_deblur and deblur is not None:
        deblurred = deblur.process(img)
        img = ensure_rgb(deblurred)
        del deblurred
        torch.cuda.empty_cache()

    # Denoise
    if use_denoise and denoise is not None:
        denoised = denoise.process(img)
        img = ensure_rgb(denoised)
        del denoised
        torch.cuda.empty_cache()

    # Sharpen (works on PIL only)
    if use_sharpen:
        if isinstance(img, Image.Image):
            img = sharpen_resolve_style(img, amount=sharpen_amount, radius=sharpen_radius)
        else:
            # Convert tensor → PIL
            pil_img = ImageLoader.tensor_to_image(img)
            img = sharpen_resolve_style(pil_img, amount=sharpen_amount, radius=sharpen_radius)

    # Upscale (AMP for VRAM reduction)
    if use_scale and edsr is not None:
        with torch.cuda.amp.autocast():
            inputs = ImageLoader.load_image(img)
            preds = edsr(inputs)

        img = preds
        del inputs
        torch.cuda.empty_cache()

   

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_scale and edsr is not None:
        ImageLoader.save_image(img, str(output_path))
    else:
        img.save(str(output_path))

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# -----------------------------
# Batch Runner
# -----------------------------

def run_batch(
    input_dir: Path,
    output_dir: Path,
    scale: int,
    use_deblur: bool,
    use_denoise: bool,
    use_scale: bool,
    use_sharpen: bool,
    deblur_factor: float,
    denoise_factor: float,
    sharpen_amount: float,
    sharpen_radius: float
):

    print("Loading models...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    deblur = DeblurProcessor(model_id="gopro_width64", model_dir="./models/", device=device) if use_deblur else None
    denoise = DenoiseProcessor(model_id="sidd_width64", model_dir="./models/", device=device) if use_denoise else None
    edsr = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale) if use_scale else None
    if edsr:
        edsr.to(device)

    files = [p for p in input_dir.rglob("*") if is_image_file(p)]
    total = len(files)

    if total == 0:
        print("No images found.")
        return

    print(f"Found {total} images.\n")

    for idx, file in enumerate(files, start=1):
        rel = file.relative_to(input_dir)
        out_path = output_dir / rel

        print(f"[{idx}/{total}] Processing: {rel}")

        try:
            process_image(
                file,
                out_path,
                scale,
                deblur,
                denoise,
                edsr,
                use_deblur,
                use_denoise,
                use_scale,
                use_sharpen,
                deblur_factor,
                denoise_factor,
                sharpen_amount,
                sharpen_radius
            )
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
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor")

    # Toggles
    parser.add_argument("--no-deblur", action="store_true", help="Disable deblurring")
    parser.add_argument("--no-denoise", action="store_true", help="Disable denoising")
    parser.add_argument("--no-scale", action="store_true", help="Disable upscaling")
    parser.add_argument("--sharpen", action="store_true", help="Enable sharpening")

    # Factors
    parser.add_argument("--deblur-factor", type=float, default=1.0, help="Deblur strength")
    parser.add_argument("--denoise-factor", type=float, default=1.0, help="Denoise strength")
    parser.add_argument("--sharpen-amount", type=float, default=1.0, help="Sharpen intensity")
    parser.add_argument("--sharpen-radius", type=float, default=1.0, help="Sharpen radius")

    args = parser.parse_args()

    run_batch(
        Path(args.input),
        Path(args.output),
        args.scale,
        use_deblur=not args.no_deblur,
        use_denoise=not args.no_denoise,
        use_scale=not args.no_scale,
        use_sharpen=args.sharpen,
        deblur_factor=args.deblur_factor,
        denoise_factor=args.denoise_factor,
        sharpen_amount=args.sharpen_amount,
        sharpen_radius=args.sharpen_radius
    )


if __name__ == "__main__":
    main()
