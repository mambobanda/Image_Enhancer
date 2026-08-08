import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------
# Sharpening Functions
# ---------------------------------------------------------

def unsharp_mask(image, radius=1.0, amount=1.0, threshold=0):
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()

    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    mask = img - blurred

    if threshold > 0:
        mask = np.where(np.abs(mask) < threshold, 0, mask)

    sharpened = img + amount * mask
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))


def frequency_sharpen(image, fine_amount=1.0, medium_amount=0.5, large_amount=0.2):
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()

    img_float = img.astype(np.float32) / 255.0

    fine_blur = cv2.GaussianBlur(img_float, (0, 0), 0.5)
    medium_blur = cv2.GaussianBlur(img_float, (0, 0), 2.0)
    large_blur = cv2.GaussianBlur(img_float, (0, 0), 5.0)

    fine_detail = img_float - fine_blur
    medium_detail = fine_blur - medium_blur
    large_detail = medium_blur - large_blur

    sharpened = (
        img_float +
        fine_amount * fine_detail +
        medium_amount * medium_detail +
        large_amount * large_detail
    )

    sharpened = np.clip(sharpened, 0, 1) * 255
    sharpened = sharpened.astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))


def edge_aware_sharpen(image, radius=1.0, amount=1.0, edge_threshold=30):
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_threshold, edge_threshold * 3)

    edge_mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    edge_mask = cv2.GaussianBlur(edge_mask.astype(np.float32), (5, 5), 1)
    edge_mask = edge_mask / 255.0

    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    mask = img - blurred

    sharpened = img + amount * mask * edge_mask[:, :, np.newaxis]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))


def selective_sharpen(image, radius=1.0, amount=1.0,
                      sharpen_highlights=True, sharpen_shadows=True,
                      threshold_low=50, threshold_high=200):

    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)

    if not sharpen_highlights:
        mask_high = l_channel < threshold_high
    else:
        mask_high = np.ones_like(l_channel, dtype=bool)

    if not sharpen_shadows:
        mask_low = l_channel > threshold_low
    else:
        mask_low = np.ones_like(l_channel, dtype=bool)

    mask = (mask_high & mask_low).astype(np.float32)

    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    detail = img - blurred

    sharpened = img + amount * detail * mask[:, :, np.newaxis]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------

def is_image_file(path: Path):
    return path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]


def process_image(path: Path, out: Path, mode, args):
    img = Image.open(path)

    if mode == "unsharp":
        img = unsharp_mask(img, radius=args.radius, amount=args.amount, threshold=args.threshold)

    elif mode == "frequency":
        img = frequency_sharpen(
            img,
            fine_amount=args.amount,
            medium_amount=args.amount * 0.5,
            large_amount=args.amount * 0.2
        )

    elif mode == "edges":
        img = edge_aware_sharpen(img, radius=args.radius, amount=args.amount, edge_threshold=args.edge_threshold)

    elif mode == "selective":
        img = selective_sharpen(
            img,
            radius=args.radius,
            amount=args.amount,
            sharpen_highlights=args.sharpen_highlights,
            sharpen_shadows=args.sharpen_shadows,
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def run_batch(input_dir: Path, output_dir: Path, mode, args):
    files = [p for p in input_dir.rglob("*") if is_image_file(p)]
    total = len(files)

    print(f"Found {total} images.\n")

    for i, file in enumerate(files, 1):
        rel = file.relative_to(input_dir)
        out = output_dir / rel

        print(f"[{i}/{total}] {rel}")
        process_image(file, out, mode, args)
        print(f"   Saved → {out}\n")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch Sharpening Tool")

    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)

    parser.add_argument("--mode", type=str, default="unsharp",
                        choices=["unsharp", "frequency", "edges", "selective"],
                        help="Sharpening mode")

    parser.add_argument("--amount", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--threshold", type=int, default=0)

    parser.add_argument("--edge-threshold", type=int, default=30)

    parser.add_argument("--sharpen-highlights", action="store_true")
    parser.add_argument("--sharpen-shadows", action="store_true")
    parser.add_argument("--threshold-low", type=int, default=50)
    parser.add_argument("--threshold-high", type=int, default=200)

    args = parser.parse_args()

    run_batch(Path(args.input), Path(args.output), args.mode, args)


if __name__ == "__main__":
    main()
