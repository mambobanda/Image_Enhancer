import argparse
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from diffusers import StableDiffusionXLImg2ImgPipeline, IPAdapterXL


# -----------------------------
# Load Diffusers Pipeline
# -----------------------------
def load_pipeline(device):
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    ip_adapter = IPAdapterXL.from_pretrained(
        "h94/IP-Adapter",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe.load_ip_adapter(ip_adapter)

    return pipe


# -----------------------------
# Image Helpers
# -----------------------------
def is_image(path: Path):
    return path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


def load_image(path: Path):
    return Image.open(path).convert("RGB")


# -----------------------------
# Style Transfer Function
# -----------------------------
def stylize(pipe, content_img, style_img, strength):
    return pipe(
        prompt="stylize this image",
        image=content_img,
        ip_adapter_image=style_img,
        strength=strength,
        guidance_scale=1.0,
        num_inference_steps=8
    ).images[0]


# -----------------------------
# Batch Runner
# -----------------------------
def run_batch(input_dir: Path, style_path: Path, output_dir: Path, strength: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipe = load_pipeline(device)
    style_img = load_image(style_path)

    files = [p for p in input_dir.rglob("*") if is_image(p)]
    if not files:
        print("No images found.")
        return

    print(f"Found {len(files)} images.\n")

    for p in tqdm(files, desc="Stylizing"):
        content_img = load_image(p)
        result = stylize(pipe, content_img, style_img, strength)

        out_path = output_dir / p.relative_to(input_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Diffusers Style Transfer (Batch + Subfolders)")
    parser.add_argument("input", type=str, help="Input folder")
    parser.add_argument("style", type=str, help="Style image")
    parser.add_argument("output", type=str, help="Output folder")
    parser.add_argument("--strength", type=float, default=0.8, help="Style strength (0–1)")

    args = parser.parse_args()

    run_batch(Path(args.input), Path(args.style), Path(args.output), args.strength)


if __name__ == "__main__":
    main()
