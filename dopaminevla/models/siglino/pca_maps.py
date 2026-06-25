# PCA visualization for Falcon Vision standalone model
import argparse
import glob
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

from dopaminevla.models.siglino.siglino import (
    SigLino,
    SigLinoHFModel,
    SigLinoImageProcessor,
    load_siglino_model,
    quantize_cpu_model,
)
from dopaminevla.models.siglino.siglino.model import SigLinoFeatures

matplotlib.use("TkAgg")


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    print(f"Image size: {img.size}")
    return img


def _get_n_storage_tokens(model: SigLino | SigLinoHFModel) -> int:
    """Get n_storage_tokens from model, handling SigLinoHFModel wrapper."""
    if isinstance(model, SigLino):
        return model.n_storage_tokens
    return model.model.n_storage_tokens


@torch.inference_mode()
def extract_patch_features(
    model: SigLino | SigLinoHFModel,
    processor: SigLinoImageProcessor,
    images: list[Image.Image],
    device: torch.device | None = None,
    max_num_patches: int = 256,
) -> list[SigLinoFeatures]:
    if device is None:
        device = next(model.parameters()).device

    dtype = next(model.parameters()).dtype
    features_per_image: list[SigLinoFeatures] = []

    for image in images:
        processed = processor(
            image,
            max_num_patches=max_num_patches,
            n_storage_tokens=_get_n_storage_tokens(model),
            pad=False,
        )
        pixel_values = processed["pixel_values"].to(device, dtype=dtype)
        padding_mask = processed["padding_mask"].to(device)
        spatial_shapes = processed["spatial_shape"].to(device)

        H, W = spatial_shapes[0].tolist()
        print(f"Spatial shapes: H={H}, W={W}, total_patches={H * W}")
        print(f"Pixel values: {pixel_values.shape}")
        out = model(
            pixel_values=pixel_values,
            padding_mask=padding_mask,
            spatial_shapes=spatial_shapes,
        )

        patch_feats = out["patch_features"]

        feats_siglip = patch_feats["siglip2"].squeeze(0)
        feats_dinov3 = patch_feats["dinov3"].squeeze(0)
        feats_siglino = patch_feats["siglino"].squeeze(0)

        features_per_image.append(
            SigLinoFeatures(
                features_siglip=feats_siglip,
                features_dinov3=feats_dinov3,
                features_siglino=feats_siglino,
                grid_hw=(H, W),
            )
        )

    return features_per_image


def fit_and_project_pca(feats_2d: torch.Tensor, n_components: int = 3, whiten: bool = True) -> np.ndarray:
    x = feats_2d.detach().float().cpu().numpy()
    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(x)
    proj = pca.transform(x)
    return proj


def render_pca_image(
    image_rgb: Image.Image,
    projected_L3: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    grid_hw: tuple[int, int],
    save_path: str,
    title: str | None = None,
) -> None:
    projected_siglino, projected_siglip, projected_dinov3 = projected_L3

    H, W = grid_hw

    def create_pca_grid(projected_features: np.ndarray) -> np.ndarray:
        grid_hw3 = projected_features.reshape(H, W, 3).astype(np.float32)
        grid_hw3 = np.nan_to_num(grid_hw3, nan=0.0, posinf=1.0, neginf=0.0)
        grid_hw3 = 1.0 / (1.0 + np.exp(-2.0 * grid_hw3))
        return grid_hw3

    viz_items = []
    if projected_siglip is not None:
        viz_items.append(("SigLIP PCA", create_pca_grid(projected_siglip)))
    if projected_dinov3 is not None:
        viz_items.append(("DINO v3 PCA", create_pca_grid(projected_dinov3)))
    if projected_siglino is not None:
        viz_items.append(("SigLino PCA", create_pca_grid(projected_siglino)))

    n_cols = max(2, len(viz_items))
    plt.figure(figsize=(4 * n_cols, 8), dpi=200)

    top_col = (n_cols + 1) // 2
    for c in range(1, n_cols + 1):
        plt.subplot(2, n_cols, c)
        if c == top_col:
            plt.imshow(image_rgb)
            plt.axis("off")
            plt.title("Original Image")
        else:
            plt.axis("off")

    for idx, (name, grid) in enumerate(viz_items, start=1):
        plt.subplot(2, n_cols, n_cols + idx)
        plt.imshow(grid)
        plt.axis("off")
        plt.title(name)

    if title:
        plt.suptitle(title, fontsize=14)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def load_model_and_processor(
    ckpt_path: str | None = None,
    config_name: str | None = None,
    device: str | None = None,
    min_pixels: int = 128 * 128,
    max_pixels: int = 256 * 256,
) -> tuple[SigLino, SigLinoImageProcessor]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model with config: {config_name or '(auto-detect)'}")

    model, processor = load_siglino_model(
        checkpoint_path=ckpt_path,
        config_name=config_name,
        device=device,
        resolve=True,
        max_pixels=max_pixels,
    )

    # Apply INT8 dynamic quantization for CPU inference
    if device == "cpu" or (isinstance(device, torch.device) and device.type == "cpu"):
        model = quantize_cpu_model(model)

    model = model.eval()
    return model, processor


def sample_jpg_images(input_dir: str, num_samples: int = 10) -> list[str]:
    jpg_pattern = os.path.join(input_dir, "*.jpg")
    jpg_files = glob.glob(jpg_pattern)

    png_pattern = os.path.join(input_dir, "*.png")
    jpg_files.extend(glob.glob(png_pattern))

    if len(jpg_files) == 0:
        raise ValueError(f"No JPG/PNG files found in {input_dir}")

    random.seed(42)
    num_to_sample = min(num_samples, len(jpg_files))
    sampled_files = random.sample(jpg_files, num_to_sample)

    print(f"Found {len(jpg_files)} image files, sampling {num_to_sample} images")
    return sampled_files


def process_single_image(
    image_path: str,
    output_dir: str,
    model: SigLino | SigLinoHFModel,
    processor: SigLinoImageProcessor,
    device: str | torch.device | None = None,
    max_num_patches: int = 256,
) -> None:
    if device is not None and not isinstance(device, torch.device):
        device = torch.device(device)
    image = load_image(image_path)

    features_info = extract_patch_features(
        model=model,
        processor=processor,
        images=[image],
        device=device,
        max_num_patches=max_num_patches,
    )
    info = features_info[0]
    H, W = info.grid_hw
    num_valid = H * W

    feats_LD_siglip = info.features_siglip[:num_valid]
    feats_LD_dinov3 = info.features_dinov3[:num_valid]
    feats_LD_siglino = info.features_siglino[:num_valid]

    msg = f"Feature shapes - siglip: {feats_LD_siglip.shape}, dinov3: {feats_LD_dinov3.shape}, siglino: {feats_LD_siglino.shape}"
    print(msg)

    projected_all_siglip = fit_and_project_pca(feats_LD_siglip)
    projected_all_dinov3 = fit_and_project_pca(feats_LD_dinov3)
    projected_all_siglino = fit_and_project_pca(feats_LD_siglino)

    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{image_basename}_pca_vis.png"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Projected shapes - siglino: {projected_all_siglino.shape}, siglip: {projected_all_siglip.shape}, dinov3: {projected_all_dinov3.shape}")
    render_pca_image(
        image_rgb=image,
        projected_L3=(projected_all_siglino, projected_all_siglip, projected_all_dinov3),
        grid_hw=info.grid_hw,
        save_path=output_path,
        title=os.path.basename(image_path),
    )

    print(f"Saved visualization: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PCA of SigLino patch features")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint, HF hub model ID, or omit to auto-resolve from --config_name",
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_path", type=str, required=True, help="Base output directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of images to sample")
    parser.add_argument("--config_name", type=str, default="dense-30M")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto-detect)")
    parser.add_argument("--max_num_patches", type=int, default=256)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output directory: {args.output_path}")

    sampled_images = sample_jpg_images(args.input_dir, args.num_samples)

    print("Loading model and processor...")
    model, processor = load_model_and_processor(
        ckpt_path=args.ckpt_path,
        config_name=args.config_name,
        device=args.device,
        min_pixels=128 * 128,
        max_pixels=(args.max_num_patches**0.5 * 16) ** 2,
    )

    print(f"Running on: {args.device}")

    print(f"Processing {len(sampled_images)} images...")
    for i, image_path in enumerate(sampled_images, 1):
        print(f"Processing image {i}/{len(sampled_images)}: {os.path.basename(image_path)}")
        process_single_image(
            image_path=image_path,
            output_dir=args.output_path,
            model=model,
            processor=processor,
            device=args.device,
            max_num_patches=args.max_num_patches,
        )

    print(f"Completed! All visualizations saved in: {args.output_path}")


if __name__ == "__main__":
    main()
