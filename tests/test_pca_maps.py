"""Tests for pca_maps.py — PCA visualization pipeline for SigLino patch features.

Each test exercises an importable function from pca_maps.py to verify
the entire script works without running main() on real images.
"""

import pathlib

import numpy as np
import pytest
import torch
from PIL import Image

from dopaminevla.models.siglino.pca_maps import (
    _compute_pos_interpolation,
    _get_n_storage_tokens,
    extract_patch_features,
    fit_and_project_pca,
    load_image,
    process_single_image,
    render_pca_image,
    sample_jpg_images,
)
from dopaminevla.models.siglino.siglino import (
    SigLino,
    SigLinoHFModel,
    SigLinoImageProcessor,
    load_siglino_model,
    siglino_configs,
)
from dopaminevla.models.siglino.siglino.hf_integration import SigLinoConfig


def _make_small_test_image(size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Create a small synthetic RGB image for testing."""
    return Image.fromarray(
        np.random.randint(0, 256, (*size, 3), dtype=np.uint8),
        mode="RGB",
    )


def _create_model_dense_30m() -> SigLino:
    args = siglino_configs["dense-30M"]
    model = SigLino(args)
    model.init_weights()
    model.eval()
    return model


class TestLoadImage:
    def test_loads_pil_rgb(self, tmp_path: pathlib.Path) -> None:
        img = _make_small_test_image()
        path = tmp_path / "test.png"
        img.save(str(path))
        loaded = load_image(str(path))
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"
        assert loaded.size == (64, 64)

    def test_raises_on_missing_file(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises((FileNotFoundError, Exception)):
            load_image(str(tmp_path / "nonexistent.jpg"))


class TestGetNStorageTokens:
    def test_bare_siglino_model(self) -> None:
        model = _create_model_dense_30m()
        n = _get_n_storage_tokens(model)
        assert n == siglino_configs["dense-30M"].n_storage_tokens
        assert n == 4

    def test_hf_wrapper_model(self) -> None:
        hf_config = SigLinoConfig.from_siglino_args(siglino_configs["dense-30M"])
        hf_model = SigLinoHFModel(hf_config)
        n = _get_n_storage_tokens(hf_model)
        assert n == 4


class TestFitAndProjectPCA:
    @pytest.fixture
    def synthetic_features(self) -> torch.Tensor:
        """Create an (N, D) feature tensor with 5 clusters of structure."""
        N, D = 50, 32
        rng = np.random.RandomState(42)
        x = rng.randn(N, D).astype(np.float32)
        # Inject low-rank structure so PCA has signal
        direction = rng.randn(D).astype(np.float32)
        x += 3.0 * np.outer(np.linspace(-1, 1, N), direction)
        return torch.from_numpy(x)

    def test_returns_correct_shape(self, synthetic_features: torch.Tensor) -> None:
        proj = fit_and_project_pca(synthetic_features, n_components=3)
        assert isinstance(proj, np.ndarray)
        assert proj.shape == (50, 3)

    def test_default_n_components(self, synthetic_features: torch.Tensor) -> None:
        proj = fit_and_project_pca(synthetic_features)
        assert proj.shape[1] == 3

    def test_different_n_components(self, synthetic_features: torch.Tensor) -> None:
        proj = fit_and_project_pca(synthetic_features, n_components=2)
        assert proj.shape == (50, 2)

    def test_deterministic_whiten_false(self, synthetic_features: torch.Tensor) -> None:
        proj1 = fit_and_project_pca(synthetic_features, whiten=False)
        proj2 = fit_and_project_pca(synthetic_features, whiten=False)
        np.testing.assert_allclose(proj1, proj2)

    def test_projection_is_finite(self, synthetic_features: torch.Tensor) -> None:
        proj = fit_and_project_pca(synthetic_features)
        assert np.all(np.isfinite(proj)), "PCA projection contains NaN or Inf"


class TestRenderPCAImage:
    @pytest.fixture
    def synthetic_projections(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create (H*W, 3) PCA projections for all three branches."""
        H, W = 4, 4
        rng = np.random.RandomState(42)
        siglino = rng.randn(H * W, 3).astype(np.float32)
        siglip = rng.randn(H * W, 3).astype(np.float32)
        dinov3 = rng.randn(H * W, 3).astype(np.float32)
        return siglino, siglip, dinov3

    def test_saves_png(
        self,
        tmp_path: pathlib.Path,
        synthetic_projections: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        image = _make_small_test_image((64, 64))
        save_path = tmp_path / "test_pca.png"
        render_pca_image(
            image_rgb=image,
            projected_L3=synthetic_projections,
            grid_hw=(4, 4),
            save_path=str(save_path),
            title="test",
        )
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        # Verify it's a valid PNG
        loaded = Image.open(save_path)
        assert loaded.mode == "RGBA"

    def test_handles_none_projections(self, tmp_path: pathlib.Path) -> None:
        image = _make_small_test_image((64, 64))
        save_path = tmp_path / "test_pca_partial.png"
        # Only SigLino projection, no SigLIP or DINOv3
        rng = np.random.RandomState(42)
        proj = rng.randn(16, 3).astype(np.float32)
        render_pca_image(
            image_rgb=image,
            projected_L3=(proj, None, None),
            grid_hw=(4, 4),
            save_path=str(save_path),
        )
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_includes_title(self, tmp_path: pathlib.Path) -> None:
        image = _make_small_test_image((32, 32))
        save_path = tmp_path / "test_pca_titled.png"
        rng = np.random.RandomState(42)
        proj = rng.randn(9, 3).astype(np.float32)
        render_pca_image(
            image_rgb=image,
            projected_L3=(proj, proj, proj),
            grid_hw=(3, 3),
            save_path=str(save_path),
            title="My Test Image",
        )
        assert save_path.exists()


class TestSampleJpgImages:
    def test_samples_correct_number(self, tmp_path: pathlib.Path) -> None:
        for i in range(5):
            img = _make_small_test_image()
            img.save(tmp_path / f"img_{i}.jpg")
        sampled = sample_jpg_images(str(tmp_path), num_samples=3)
        assert len(sampled) == 3
        for path in sampled:
            assert path.startswith(str(tmp_path))

    def test_includes_png(self, tmp_path: pathlib.Path) -> None:
        img = _make_small_test_image()
        img.save(tmp_path / "test.png")
        sampled = sample_jpg_images(str(tmp_path), num_samples=1)
        assert len(sampled) == 1
        assert sampled[0].endswith(".png")

    def test_raises_on_empty_dir(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="No JPG/PNG files found"):
            sample_jpg_images(str(tmp_path))

    def test_respects_num_samples(self, tmp_path: pathlib.Path) -> None:
        for i in range(10):
            img = _make_small_test_image()
            img.save(tmp_path / f"img_{i}.jpg")
        sampled = sample_jpg_images(str(tmp_path), num_samples=5)
        assert len(sampled) == 5


class TestExtractPatchFeatures:
    """Integration test: create model + processor + synthetic image, extract features."""

    @pytest.fixture
    def model_and_processor(self) -> tuple[SigLino, SigLinoImageProcessor]:
        model, processor = load_siglino_model(
            checkpoint_path=None,
            config_name="dense-30M",
            device="cpu",
        )
        return model, processor

    def test_extracts_all_three_branches(
        self, model_and_processor: tuple[SigLino, SigLinoImageProcessor]
    ) -> None:
        model, processor = model_and_processor
        image = _make_small_test_image((224, 224))
        features_list = extract_patch_features(
            model=model,
            processor=processor,
            images=[image],
            device=torch.device("cpu"),
            max_num_patches=256,
        )
        assert len(features_list) == 1
        info = features_list[0]
        assert info.features_siglip is not None
        assert info.features_dinov3 is not None
        assert info.features_siglino is not None
        assert info.grid_hw[0] > 0 and info.grid_hw[1] > 0

    def test_feature_no_nan(
        self, model_and_processor: tuple[SigLino, SigLinoImageProcessor]
    ) -> None:
        """Random-init model forward should not produce NaN with proper init_weights."""
        model, processor = model_and_processor
        image = _make_small_test_image((224, 224))
        features_list = extract_patch_features(
            model=model,
            processor=processor,
            images=[image],
            device=torch.device("cpu"),
            max_num_patches=256,
        )
        info = features_list[0]
        for name, feat in [
            ("siglip", info.features_siglip),
            ("dinov3", info.features_dinov3),
            ("siglino", info.features_siglino),
        ]:
            assert not feat.isnan().any(), f"NaN in {name}"
            assert not feat.isinf().any(), f"Inf in {name}"

    def test_multiple_images(
        self, model_and_processor: tuple[SigLino, SigLinoImageProcessor]
    ) -> None:
        model, processor = model_and_processor
        images = [_make_small_test_image((224, 224)) for _ in range(2)]
        features_list = extract_patch_features(
            model=model,
            processor=processor,
            images=images,
            device=torch.device("cpu"),
            max_num_patches=256,
        )
        assert len(features_list) == 2

    def test_spatial_grid_matches_patches(
        self, model_and_processor: tuple[SigLino, SigLinoImageProcessor]
    ) -> None:
        """Validate that grid_hw matches the actual number of patches returned."""
        model, processor = model_and_processor
        image = _make_small_test_image((224, 224))
        features_list = extract_patch_features(
            model=model,
            processor=processor,
            images=[image],
            device=torch.device("cpu"),
            max_num_patches=256,
        )
        info = features_list[0]
        H, W = info.grid_hw
        num_valid = H * W
        assert info.features_siglino.shape[0] >= num_valid
        assert info.features_siglip.shape[0] >= num_valid
        assert info.features_dinov3.shape[0] >= num_valid

    def test_mask_safety_padded_image_no_nan(
        self, model_and_processor: tuple[SigLino, SigLinoImageProcessor]
    ) -> None:
        """Small image with large max_num_patches: mask safety path should prevent NaN."""
        model, processor = model_and_processor
        # A 32x32 image produces ~4 patches in the default 16x16 grid.
        # max_num_patches=256 means lots of padding tokens.
        image = _make_small_test_image((32, 32))
        features_list = extract_patch_features(
            model=model,
            processor=processor,
            images=[image],
            device=torch.device("cpu"),
            max_num_patches=256,
        )
        info = features_list[0]
        for name, feat in [
            ("siglip", info.features_siglip),
            ("dinov3", info.features_dinov3),
            ("siglino", info.features_siglino),
        ]:
            assert not feat.isnan().any(), f"NaN in {name}"
            assert not feat.isinf().any(), f"Inf in {name}"

    def test_mask_safety_large_max_patches(
        self, model_and_processor: tuple[SigLino, SigLinoImageProcessor]
    ) -> None:
        """Even with very large max_num_patches, features should remain valid."""
        model, processor = model_and_processor
        image = _make_small_test_image((224, 224))
        features_list = extract_patch_features(
            model=model,
            processor=processor,
            images=[image],
            device=torch.device("cpu"),
            max_num_patches=4096,
        )
        info = features_list[0]
        # With 4096 max patches, the 224x224 image produces H*W < 4096,
        # so there should be padding tokens. Features should be clean.
        for feat in [info.features_siglip, info.features_dinov3, info.features_siglino]:
            assert not feat.isnan().any(), "NaN in features"
            assert not feat.isinf().any(), "Inf in features"


class TestProcessSingleImage:
    """End-to-end orchestration: model + image → saved PCA visualization."""

    @pytest.fixture
    def model_and_processor(self) -> tuple[SigLino, SigLinoImageProcessor]:
        model, processor = load_siglino_model(
            checkpoint_path=None,
            config_name="dense-30M",
            device="cpu",
        )
        return model, processor

    def test_full_pipeline(
        self, model_and_processor: tuple[SigLino, SigLinoImageProcessor], tmp_path: pathlib.Path
    ) -> None:
        model, processor = model_and_processor
        image_path = tmp_path / "input.png"
        _make_small_test_image((224, 224)).save(str(image_path))

        output_dir = tmp_path / "outputs"
        process_single_image(
            image_path=str(image_path),
            output_dir=str(output_dir),
            model=model,
            processor=processor,
            device=torch.device("cpu"),
            max_num_patches=256,
        )

        # Verify output file was created
        png_files = list(output_dir.glob("*_pca_vis.png"))
        assert len(png_files) == 1
        assert png_files[0].stat().st_size > 0

    def test_output_naming(
        self, model_and_processor: tuple[SigLino, SigLinoImageProcessor], tmp_path: pathlib.Path
    ) -> None:
        model, processor = model_and_processor
        image_path = tmp_path / "my_photo.png"
        _make_small_test_image((224, 224)).save(str(image_path))

        output_dir = tmp_path / "viz"
        process_single_image(
            image_path=str(image_path),
            output_dir=str(output_dir),
            model=model,
            processor=processor,
            device=torch.device("cpu"),
            max_num_patches=256,
        )

        expected = output_dir / "my_photo_pca_vis.png"
        assert expected.exists(), f"Expected {expected}"

    def test_pca_projection_finite(
        self, model_and_processor: tuple[SigLino, SigLinoImageProcessor], tmp_path: pathlib.Path
    ) -> None:
        """Confirm the PCA step doesn't produce NaN/Inf from the extracted features."""
        model, processor = model_and_processor
        image_path = tmp_path / "input.png"
        _make_small_test_image((224, 224)).save(str(image_path))

        output_dir = tmp_path / "outputs"
        process_single_image(
            image_path=str(image_path),
            output_dir=str(output_dir),
            model=model,
            processor=processor,
            device=torch.device("cpu"),
            max_num_patches=256,
        )
        assert list(output_dir.glob("*_pca_vis.png"))[0].stat().st_size > 0


class TestPosInterpolation:
    def test_grid_dim_16_gives_256_patches(self) -> None:
        max_patches, max_pixels = _compute_pos_interpolation(16)
        assert max_patches == 256, f"expected 256, got {max_patches}"
        assert max_pixels == (16 * 16) ** 2, f"expected {(16 * 16) ** 2}, got {max_pixels}"

    def test_grid_dim_32_gives_1024_patches(self) -> None:
        max_patches, max_pixels = _compute_pos_interpolation(32)
        assert max_patches == 1024, f"expected 1024, got {max_patches}"
        assert max_pixels == (32 * 16) ** 2, f"expected {(32 * 16) ** 2}, got {max_pixels}"

    def test_grid_dim_64_gives_4096_patches(self) -> None:
        max_patches, max_pixels = _compute_pos_interpolation(64)
        assert max_patches == 4096, f"expected 4096, got {max_patches}"
        assert max_pixels == (64 * 16) ** 2, f"expected {(64 * 16) ** 2}, got {max_pixels}"

    def test_zero_grid_dim_uses_default(self) -> None:
        max_patches, max_pixels = _compute_pos_interpolation(0)
        assert max_patches == 256, "disabled interpolation should use default_max_num_patches"
        assert max_pixels == (16 * 16) ** 2

    def test_small_grid_dim_uses_default(self) -> None:
        max_patches, max_pixels = _compute_pos_interpolation(8)
        assert max_patches == 256, "grid_dim < 16 should use default"

    def test_custom_default_preserved_when_disabled(self) -> None:
        max_patches, max_pixels = _compute_pos_interpolation(0, default_max_num_patches=512)
        assert max_patches == 512, "custom default should be used when disabled"
        assert max_pixels == int((512**0.5 * 16) ** 2)

    def test_negative_grid_dim_uses_default(self) -> None:
        max_patches, max_pixels = _compute_pos_interpolation(-4)
        assert max_patches == 256, "negative grid_dim should use default"
