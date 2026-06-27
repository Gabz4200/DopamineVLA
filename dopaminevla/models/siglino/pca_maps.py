# PCA visualization for Falcon Vision standalone model
import argparse
import glob
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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


def _compute_pos_interpolation(
    grid_dim: int, default_max_num_patches: int = 256
) -> tuple[int, int]:
    """Convert grid dimension to (max_patches, max_pixels).

    Args:
        grid_dim: Target patch grid dimension (N = NxN grid).
            Values >= 16 enable interpolation; values < 16 return the default.
        default_max_num_patches: Fallback when interpolation is disabled.

    Returns:
        Tuple of (max_patches, max_pixels) for the model/processor.
    """
    if grid_dim >= 16:
        max_patches = grid_dim**2
        max_pixels = (grid_dim * 16) ** 2
    else:
        max_patches = default_max_num_patches
        max_pixels = int((max_patches**0.5 * 16) ** 2)
    return max_patches, max_pixels


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
    blend_layer: int | list[int] | None = None,
) -> list[SigLinoFeatures]:
    if device is None:
        device = next(model.parameters()).device

    dtype = next(model.parameters()).dtype
    features_per_image: list[SigLinoFeatures] = []

    # Normalize single int to list for uniform handling
    if isinstance(blend_layer, int):
        blend_layer = [blend_layer]

    # Unwrap HF wrapper to get the bare SigLino for hook attachment
    base_model = model.model if isinstance(model, SigLinoHFModel) else model

    # Forward hooks: capture hidden states right after target layers,
    # before the final norm.  We apply norm in post-processing below.
    captured_early: dict[int, torch.Tensor] = {}
    hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    if blend_layer is not None:
        n_layers = len(base_model.layers)

        def _make_hook(idx: int):
            def hook(_module: object, _inp: object, out: torch.Tensor) -> None:
                captured_early[idx] = out.detach()

            return hook

        for layer_idx in blend_layer:
            target = layer_idx if layer_idx >= 0 else n_layers + layer_idx
            target = max(0, min(n_layers - 1, target))
            hook_handles.append(
                base_model.layers[str(target)].register_forward_hook(_make_hook(target))
            )

    try:
        for image in images:
            processed = processor(
                image,
                max_num_patches=max_num_patches,
                n_storage_tokens=_get_n_storage_tokens(model),
                pad=True,
            )
            pixel_values = processed["pixel_values"].to(device, dtype=dtype)
            padding_mask = processed["padding_mask"].to(device)
            spatial_shapes = processed["spatial_shape"].to(device)

            H, W = spatial_shapes[0].tolist()
            n_actual = H * W
            n_padded = pixel_values.shape[1]

            # Safety: enforce padding mask correctness.
            # If the processor returned more patches than max_num_patches
            # (which our fix to pad_along_first_dim truncates), the mask
            # may be all-1s with wrong length.  Fix both here.
            if n_padded > n_actual:
                # Padded case: ensure padding tokens are masked out
                padding_mask[0, n_actual:] = 0.0
            elif n_padded < n_actual:
                # Shouldn't happen with pad=True, but guard
                pixel_values = F.pad(pixel_values, (0, 0, 0, n_actual - n_padded))
                padding_mask = F.pad(padding_mask, (0, n_actual - n_padded), value=0.0)

            print(f"Spatial shapes: H={H}, W={W}, total_patches={n_actual}")
            print(f"Pixel values: {pixel_values.shape}")
            out = model(
                pixel_values=pixel_values,
                padding_mask=padding_mask,
                spatial_shapes=spatial_shapes,
            )

            patch_feats = out["patch_features"]

            n_reg = _get_n_storage_tokens(model)
            feats_siglip = patch_feats["siglip2"].squeeze(0)  # (L, siglip_dim)
            feats_dinov3 = patch_feats["dinov3"].squeeze(0)  # (L, dinov3_dim)
            feats_siglino = patch_feats["siglino"].squeeze(0)[n_reg:]  # (L, model_dim)

            # Multi-layer blending: project each early layer through the same
            # teacher-space adapters as the final layer, then concatenate per
            # feature type.  Keeps all dims in the same space so PCA finds
            # meaningful variance instead of being dominated by mismatched scales.
            if blend_layer is not None and captured_early:
                early_siglip: list[torch.Tensor] = []
                early_dinov3: list[torch.Tensor] = []
                early_siglino: list[torch.Tensor] = []
                for layer_idx in sorted(captured_early.keys()):
                    h = captured_early[layer_idx].squeeze(0)  # (S, D)
                    h = base_model.norm(h)  # (S, D) — student-space norm
                    patches = h[1 + n_reg :]  # (L, D) — strip CLS + registers
                    # Project through the same adapters the final layer uses
                    early_siglip.append(base_model.siglip2_adapter(patches))  # (L, 1152)
                    early_dinov3.append(base_model.dinov3_adapter(patches))  # (L, 1280)
                    early_siglino.append(patches)  # (L, D) — raw model dim
                feats_siglip = torch.cat([feats_siglip] + early_siglip, dim=-1)  # (L, 1152*(1+N))
                feats_dinov3 = torch.cat([feats_dinov3] + early_dinov3, dim=-1)  # (L, 1280*(1+N))
                feats_siglino = torch.cat([feats_siglino] + early_siglino, dim=-1)  # (L, D*(1+N))

            features_per_image.append(
                SigLinoFeatures(
                    features_siglip=feats_siglip,
                    features_dinov3=feats_dinov3,
                    features_siglino=feats_siglino,
                    grid_hw=(H, W),
                )
            )
    finally:
        for handle in hook_handles:
            handle.remove()

    return features_per_image


def fit_and_project_pca(
    feats_2d: torch.Tensor, n_components: int = 3, whiten: bool = False
) -> np.ndarray:
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
    use_sigmoid: bool = False,
) -> None:
    projected_siglino, projected_siglip, projected_dinov3 = projected_L3

    H, W = grid_hw

    def create_pca_grid(projected_features: np.ndarray) -> np.ndarray:
        grid_hw3 = projected_features.reshape(H, W, 3).astype(np.float32)
        # Safety: guard NaN/Inf before any further processing
        grid_hw3 = np.nan_to_num(grid_hw3, nan=0.0, posinf=0.0, neginf=0.0)
        if use_sigmoid:
            # Sigmoid: unbounded, saturates at |x| > 3. Tendency to oversaturate
            # and produce black points from extreme negative outliers.
            grid_hw3 = 1.0 / (1.0 + np.exp(-2.0 * grid_hw3))
        else:
            # Robust min-max: clip outliers per channel before scaling to [0, 1].
            # This prevents oversaturation from high-variance features and
            # eliminates black points from extreme negative outliers.
            for c in range(3):
                channel = grid_hw3[:, :, c]
                lo, hi = np.percentile(channel, [1, 99])
                span = hi - lo
                if span > 1e-8:
                    grid_hw3[:, :, c] = np.clip((channel - lo) / span, 0.0, 1.0)
                else:
                    grid_hw3[:, :, c] = 0.5
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


def _smooth_features_2d(
    feats_LD: torch.Tensor,
    H: int,
    W: int,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Apply 2D average pooling to smooth patch features spatially.

    Reshapes (L, D) -> (1, D, H, W), applies avg_pool2d, then flattens
    back to (L, D).  Useful for taming checkerboard noise in SigLIP
    features (spatial incoherence from contrastive pretraining).
    """
    D = feats_LD.shape[-1]
    grid = feats_LD.view(H, W, D).permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
    # Replicate-pad before pooling to avoid zero-boundary distortion (the "ring" artifact).
    # Zero padding dilutes edge features with zeros, creating a distinct PCA cluster
    # at boundaries that renders as a discolored border.
    pad = kernel_size // 2
    if pad > 0:
        grid = F.pad(grid, (pad, pad, pad, pad), mode="replicate")
    smoothed = F.avg_pool2d(grid, kernel_size=kernel_size, stride=1, padding=0)
    return smoothed.squeeze(0).permute(1, 2, 0).reshape(-1, D)


def process_single_image(
    image_path: str,
    output_dir: str,
    model: SigLino | SigLinoHFModel,
    processor: SigLinoImageProcessor,
    device: str | torch.device | None = None,
    max_num_patches: int = 256,
    use_sigmoid: bool = False,
    apply_feature_averaging: int = 0,
    blend_layer: int | list[int] | None = None,
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
        blend_layer=blend_layer,
    )
    info = features_info[0]
    H, W = info.grid_hw
    num_valid = H * W

    feats_LD_siglip = info.features_siglip[:num_valid]
    feats_LD_dinov3 = info.features_dinov3[:num_valid]
    feats_LD_siglino = info.features_siglino[:num_valid]

    # Spatial smoothing tames checkerboard noise (SigLIP adapter) and
    # position-dependent variation (SigLino raw features) at high resolution.
    # DINOv3 features are naturally smooth (teacher self-attention) but mild
    # smoothing doesn't hurt edges — it removes high-freq spatial noise.
    if apply_feature_averaging > 0:
        k = apply_feature_averaging
        feats_LD_siglip = _smooth_features_2d(feats_LD_siglip, H, W, kernel_size=k)
        feats_LD_dinov3 = _smooth_features_2d(feats_LD_dinov3, H, W, kernel_size=k)
        feats_LD_siglino = _smooth_features_2d(feats_LD_siglino, H, W, kernel_size=k)

    msg = (
        f"Feature shapes - siglip: {feats_LD_siglip.shape}, "
        f"dinov3: {feats_LD_dinov3.shape}, siglino: {feats_LD_siglino.shape}"
    )
    print(msg)

    projected_all_siglip = fit_and_project_pca(feats_LD_siglip)
    projected_all_dinov3 = fit_and_project_pca(feats_LD_dinov3)
    projected_all_siglino = fit_and_project_pca(feats_LD_siglino)

    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{image_basename}_pca_vis.png"
    output_path = os.path.join(output_dir, output_filename)

    print(
        f"Projected shapes - siglino: {projected_all_siglino.shape}, "
        f"siglip: {projected_all_siglip.shape}, dinov3: {projected_all_dinov3.shape}"
    )
    render_pca_image(
        image_rgb=image,
        projected_L3=(projected_all_siglino, projected_all_siglip, projected_all_dinov3),
        grid_hw=info.grid_hw,
        save_path=output_path,
        title=os.path.basename(image_path),
        use_sigmoid=use_sigmoid,
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
    parser.add_argument(
        "--pos-interpolation",
        type=int,
        default=0,
        help="Target patch grid dimension N (N >= 16).  Sets max_num_patches = N*N "
        "for denser feature maps.  E.g. 64 = 64x64 = 4096 patches.  "
        "0 = disabled, uses --max_num_patches.",
    )
    parser.add_argument(
        "--feature-averaging",
        type=int,
        default=0,
        help="Kernel size for 2D spatial avg pooling on ALL feature types "
        "(siglip, dinov3, siglino) before PCA (tames checkerboard noise and "
        "position-dependent variation). 0 = disabled. Recommended: 3.",
    )
    parser.add_argument(
        "--blend-layer",
        type=int,
        nargs="+",
        default=None,
        help="Early layer index(es) whose raw (normalized) features are concatenated "
        "to all three PCA representations (siglip, dinov3, siglino) to recover "
        "structural edges lost in deep layers.  Accepts multiple values for multi- "
        "layer blending: --blend-layer -4 -2.  Uses forward hooks in PCA script, "
        "no model changes. Negative indices count from the last layer.",
    )
    parser.add_argument(
        "--use-sigmoid",
        action="store_true",
        help="Use sigmoid scaling (old behavior) instead of robust percentile-clipped min-max "
        "for PCA rendering. Sigmoid oversaturates with high-variance features and "
        "produces black points from extreme outliers.",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Apply positional interpolation: compute effective patch count first
    max_patches, max_pixels_for_processor = _compute_pos_interpolation(
        args.pos_interpolation, args.max_num_patches
    )

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output directory: {args.output_path}")

    sampled_images = sample_jpg_images(args.input_dir, args.num_samples)

    print("Loading model and processor...")

    model, processor = load_model_and_processor(
        ckpt_path=args.ckpt_path,
        config_name=args.config_name,
        device=args.device,
        min_pixels=128 * 128,
        max_pixels=max_pixels_for_processor,
    )

    print(f"Running on: {args.device}")
    if args.pos_interpolation >= 16:
        print(
            f"Positional interpolation enabled: max_num_patches={max_patches} "
            f"({args.pos_interpolation}x{args.pos_interpolation} grid)"
        )
    if args.feature_averaging > 0:
        print(
            f"Feature averaging enabled: "
            f"{args.feature_averaging}x{args.feature_averaging} spatial smoothing before PCA"
        )
    if args.blend_layer is not None:
        print(f"Multi-layer blending enabled: final layer \u2295 layers {args.blend_layer}")

    print(f"Processing {len(sampled_images)} images...")
    for i, image_path in enumerate(sampled_images, 1):
        print(f"Processing image {i}/{len(sampled_images)}: {os.path.basename(image_path)}")
        process_single_image(
            image_path=image_path,
            output_dir=args.output_path,
            model=model,
            processor=processor,
            device=args.device,
            max_num_patches=max_patches,
            use_sigmoid=args.use_sigmoid,
            apply_feature_averaging=args.feature_averaging,
            blend_layer=args.blend_layer,
        )

    print(f"Completed! All visualizations saved in: {args.output_path}")


if __name__ == "__main__":
    main()
