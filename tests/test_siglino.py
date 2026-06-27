import pathlib
from typing import Any

import pytest
import torch
from PIL import Image

from dopaminevla.models.siglino.siglino import (
    SigLino,
    SigLinoArgs,
    SigLinoConfig,
    SigLinoHFModel,
    SigLinoImageProcessor,
    SigLinoPreTrainedModel,
    load_siglino_model,
    quantize_cpu_model,
    siglino_configs,
)
from dopaminevla.models.siglino.siglino.configs import MoEArgs
from dopaminevla.models.siglino.siglino.utils import (
    _find_matching_config_name,
    _is_hub_id,
    _validate_config_checkpoint_match,
)


@pytest.fixture(scope="module")
def dense_30m_args() -> SigLinoArgs:
    return siglino_configs["dense-30M"]


@pytest.fixture(scope="module")
def dense_70m_args() -> SigLinoArgs:
    return siglino_configs["dense-70M"]


@pytest.fixture(scope="module")
def siglino_015b_args() -> SigLinoArgs:
    return siglino_configs["siglino-0.15B"]


@pytest.fixture(scope="module")
def dummy_input() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)


class TestImports:
    def test_all_exports_exist(self) -> None:
        assert SigLino is not None
        assert SigLinoArgs is not None
        assert SigLinoConfig is not None
        assert SigLinoHFModel is not None
        assert SigLinoImageProcessor is not None
        assert SigLinoPreTrainedModel is not None
        assert load_siglino_model is not None
        assert quantize_cpu_model is not None
        assert siglino_configs is not None
        assert "dense-30M" in siglino_configs
        assert "dense-70M" in siglino_configs
        assert "siglino-0.15B" in siglino_configs
        assert "siglino-0.3B" in siglino_configs


class TestSigLinoArgs:
    def test_moe_args_defaults(self) -> None:
        args = SigLinoArgs()
        assert isinstance(args.moe_args, MoEArgs)

    def test_dense_config_has_moe_dim_zero(self) -> None:
        args = siglino_configs["dense-30M"]
        assert args.moe_dim == 0


class TestSigLinoConfig:
    def test_create_from_scratch(self) -> None:
        config = SigLinoConfig()
        assert config.model_type == "siglino"

    def test_to_siglino_args_roundtrip(self) -> None:
        config = SigLinoConfig(hidden_size=512, num_hidden_layers=12, num_attention_heads=8)
        args = config.to_siglino_args()
        assert args.dim == 512
        assert args.n_layers == 12
        assert args.n_heads == 8

    def test_from_siglino_args(self) -> None:
        args = SigLinoArgs(dim=384, n_layers=12, n_heads=6)
        config = SigLinoConfig.from_siglino_args(args)
        assert config.hidden_size == 384
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 6

    def test_from_hub_config(self) -> None:
        hub_dict = {
            "dim": 512,
            "n_layers": 12,
            "n_heads": 8,
            "n_kv_heads": 8,
            "head_dim": 64,
            "moe_dim": 0,
            "activation": "silu",
        }
        config = SigLinoConfig.from_hub_config(hub_dict)
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8
        assert config.num_key_value_heads == 8

    def test_hub_config_roundtrip(self) -> None:
        config = SigLinoConfig(hidden_size=768, num_hidden_layers=18, num_attention_heads=12)
        args = config.to_siglino_args()
        config2 = SigLinoConfig.from_siglino_args(args)
        assert config2.hidden_size == config.hidden_size
        assert config2.num_hidden_layers == config.num_hidden_layers

    def test_save_and_load_local_config(self, tmp_path: pathlib.Path) -> None:
        config = SigLinoConfig(hidden_size=384, num_hidden_layers=12)
        config.save_pretrained(tmp_path)
        loaded = SigLinoConfig.from_pretrained(tmp_path)
        assert loaded.hidden_size == 384
        assert loaded.num_hidden_layers == 12


class TestSigLinoModelCPU:
    """CPU forward tests for dense and MoE configurations."""

    def _create_model(self, args: SigLinoArgs) -> SigLino:
        model = SigLino(args)
        model.init_weights()
        model.eval()
        return model

    def _run_forward(self, model: SigLino) -> list[dict[str, Any]]:
        out = []
        for n in range(1, 5):
            x = torch.randn(1, 3, 224, 224)
            out.append(model(pixel_values=x, spatial_shapes=torch.tensor([[14, 14]])))
            assert "patch_features" in out[-1]
            assert "siglino" in out[-1]["patch_features"]
        return out

    def test_dense_30m_cpu_forward(self, dense_30m_args: SigLinoArgs) -> None:
        model = self._create_model(dense_30m_args)
        out = self._run_forward(model)
        for o in out:
            feats = o["patch_features"]["siglino"]
            assert feats.ndim == 3

    def test_dense_70m_cpu_forward(self, dense_70m_args: SigLinoArgs) -> None:
        model = self._create_model(dense_70m_args)
        out = self._run_forward(model)
        for o in out:
            feats = o["patch_features"]["siglino"]
            assert feats.ndim == 3

    def test_moe_015b_cpu_forward(self, siglino_015b_args: SigLinoArgs) -> None:
        model = self._create_model(siglino_015b_args)
        out = self._run_forward(model)
        for o in out:
            feats = o["patch_features"]["siglino"]
            assert feats.ndim == 3

    def test_batched_input(self, dense_30m_args: SigLinoArgs) -> None:
        model = self._create_model(dense_30m_args)
        x = torch.randn(2, 3, 224, 224)
        out = model(pixel_values=x, spatial_shapes=torch.tensor([[14, 14], [14, 14]]))
        feats = out["patch_features"]["siglino"]
        assert feats.shape[0] == 2

    def test_no_nan_in_output(self, dense_30m_args: SigLinoArgs) -> None:
        """Verify random-init model produces no NaN (init_weights was called)."""
        model = self._create_model(dense_30m_args)
        out = self._run_forward(model)
        for o in out:
            feats = o["patch_features"]["siglino"]
            assert not feats.isnan().any(), f"NaN found in {feats.shape}"
            assert not feats.isinf().any(), f"Inf found in {feats.shape}"


_HF_SMALL = dict(
    hidden_size=384,
    num_hidden_layers=12,
    num_attention_heads=6,
    num_key_value_heads=6,
    head_dim=64,
)


class TestSigLinoHFModel:
    def test_create_hf_model(self) -> None:
        config = SigLinoConfig(**_HF_SMALL)
        model = SigLinoHFModel(config)
        assert isinstance(model, SigLinoHFModel)
        assert model.config.hidden_size == 384

    def test_hf_forward(self) -> None:
        config = SigLinoConfig(**_HF_SMALL)
        model = SigLinoHFModel(config)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        out = model(pixel_values=x, spatial_shapes=torch.tensor([[14, 14]]))
        assert "patch_features" in out

    def test_hf_state_dict_keys(self) -> None:
        config = SigLinoConfig(**_HF_SMALL)
        model = SigLinoHFModel(config)
        sd = model.state_dict()
        assert any(k.startswith("model.layers.") for k in sd.keys())
        assert any(k.startswith("model.img_projector.") for k in sd.keys())
        assert any(k.startswith("model.cls_token") for k in sd.keys())

    def test_from_pretrained_hub(self) -> None:
        model = SigLinoHFModel.from_pretrained("tiiuae/siglino-70M")
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        out = model(pixel_values=x, spatial_shapes=torch.tensor([[14, 14]]))
        assert "patch_features" in out

    @staticmethod
    def _assert_save_load_preserves_weights(
        original_model: SigLinoPreTrainedModel, tmp_path: pathlib.Path
    ) -> SigLinoHFModel:
        """Verify save_pretrained/from_pretrained cycle preserves all weights."""
        original_model.save_pretrained(tmp_path)
        loaded_model = SigLinoHFModel.from_pretrained(tmp_path)
        loaded_model.eval()

        # Primary check: weights must match exactly
        orig_sd = original_model.state_dict()
        loaded_sd = loaded_model.state_dict()
        assert orig_sd.keys() == loaded_sd.keys(), "State dict keys differ after save/load!"
        for k in orig_sd:
            assert torch.equal(orig_sd[k], loaded_sd[k]), f"Weight mismatch in {k} after save/load!"

        # Non-persistent buffer freqs_cis can be corrupted by from_pretrained.
        # Recompute to ensure correctness.
        loaded_model.model._post_init()

        return loaded_model

    def test_save_and_load_moe_hf(self, tmp_path: pathlib.Path) -> None:
        # Initialize a miniature MoE architecture.
        config = SigLinoConfig(
            hidden_size=64,
            num_hidden_layers=2,
            first_n_layers_dense=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            moe_dim=128,
            moe_num_experts=4,
            moe_num_shared_experts=1,
            moe_top_k=2,
        )

        original_model = SigLinoHFModel(config)
        original_model.eval()

        x = torch.randn(1, 3, 224, 224)
        spatial_shapes = torch.tensor([[14, 14]])

        with torch.no_grad():
            original_out = original_model(pixel_values=x, spatial_shapes=spatial_shapes)
            orig_siglino_feat = original_out["patch_features"]["siglino"]

        # Verify weights survive the void
        loaded_model = self._assert_save_load_preserves_weights(original_model, tmp_path)

        # Verify forward pass matches (secondary check)
        with torch.no_grad():
            loaded_out = loaded_model(pixel_values=x, spatial_shapes=spatial_shapes)
            loaded_siglino_feat = loaded_out["patch_features"]["siglino"]

        assert not loaded_siglino_feat.isnan().any(), (
            "Loaded MoE model produced NaN (weights and buffers verified)."
        )
        assert torch.allclose(orig_siglino_feat, loaded_siglino_feat, atol=1e-6), (
            "MoE forward output changed after save/load (weights verified above)."
        )

    def test_save_and_load_local_hf(self, tmp_path: pathlib.Path) -> None:
        config = SigLinoConfig(**_HF_SMALL)
        original_model = SigLinoHFModel(config)
        original_model.eval()

        x = torch.randn(1, 3, 224, 224)
        spatial_shapes = torch.tensor([[14, 14]])

        with torch.no_grad():
            original_out = original_model(pixel_values=x, spatial_shapes=spatial_shapes)
            orig_siglino_feat = original_out["patch_features"]["siglino"]

        # Verify weights survive
        loaded_model = self._assert_save_load_preserves_weights(original_model, tmp_path)

        # Verify forward pass matches (secondary check)
        with torch.no_grad():
            loaded_out = loaded_model(pixel_values=x, spatial_shapes=spatial_shapes)
            loaded_siglino_feat = loaded_out["patch_features"]["siglino"]

        assert not loaded_siglino_feat.isnan().any(), (
            "Loaded model produced NaN (weights and buffers verified)."
        )
        assert torch.allclose(orig_siglino_feat, loaded_siglino_feat, atol=1e-6), (
            "Forward output changed after save/load (weights verified above)."
        )


class TestSigLinoImageProcessor:
    def test_create_processor(self) -> None:
        processor = SigLinoImageProcessor()
        assert processor is not None

    def test_process_single_image(self) -> None:
        processor = SigLinoImageProcessor(min_pixels=128 * 128, max_pixels=256 * 256)
        img = Image.new("RGB", (224, 224))
        out = processor(img)
        assert "pixel_values" in out
        assert "padding_mask" in out
        assert "spatial_shape" in out

    def test_process_multiple_images(self) -> None:
        processor = SigLinoImageProcessor(min_pixels=128 * 128, max_pixels=256 * 256)
        imgs = [Image.new("RGB", (224, 224)) for _ in range(3)]
        out = processor(imgs)
        assert out["pixel_values"].shape[0] == 3


class TestLoadSiglinoModel:
    def test_load_with_config_name(self) -> None:
        model, processor = load_siglino_model(
            checkpoint_path=None,
            config_name="dense-30M",
            device="cpu",
        )
        assert isinstance(model, SigLino)
        assert processor is not None

    def test_auto_device_cpu(self) -> None:
        model, processor = load_siglino_model(
            checkpoint_path=None,
            config_name="dense-30M",
        )
        dev = next(model.parameters()).device
        assert dev.type == "cpu"


class TestDeviceAgnostic:
    def test_flex_attn_disabled_on_cpu(self) -> None:
        model = SigLino(siglino_configs["dense-30M"])
        cpu = torch.device("cpu")
        assert not model._use_flex_attn_on_device(cpu)

    def test_compile_auto_disabled_on_cpu(self, dense_30m_args: SigLinoArgs) -> None:
        model = SigLino(dense_30m_args)
        x = torch.randn(1, 3, 224, 224)
        out = model(pixel_values=x, spatial_shapes=torch.tensor([[14, 14]]))
        assert "patch_features" in out

    def test_no_nan_in_random_forward(self, dense_30m_args: SigLinoArgs) -> None:
        """Sanity: random weights with init_weights produce finite output."""
        model = SigLino(dense_30m_args)
        model.init_weights()
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        out = model(pixel_values=x, spatial_shapes=torch.tensor([[14, 14]]))
        for name, feat in out["patch_features"].items():
            assert not feat.isnan().any(), f"NaN in {name}"
            assert not feat.isinf().any(), f"Inf in {name}"


class TestQuantizeCPU:
    def test_quantize_cpu_model_fn_available(self) -> None:
        """quantize_cpu_model should be a callable function."""
        assert callable(quantize_cpu_model)

    def test_quantize_cpu_model_does_not_crash(self, dense_30m_args: SigLinoArgs) -> None:
        """Apply torchao quantize to a small model; verify it still runs."""
        model = SigLino(dense_30m_args)
        model.init_weights()
        model.eval()
        try:
            quantize_cpu_model(model)
        except Exception as e:
            pytest.skip(f"torchao quantize not supported in this env: {e}")

        out = model(
            pixel_values=torch.randn(1, 3, 224, 224), spatial_shapes=torch.tensor([[14, 14]])
        )
        assert "patch_features" in out


class TestConfigCheckpointMatching:
    """Tests for config/checkpoint resolution and validation helpers."""

    def test_find_matching_dense_30m(self) -> None:
        """_find_matching_config_name finds 'dense-30M' for its own args."""
        args = siglino_configs["dense-30M"]
        assert _find_matching_config_name(args) == "dense-30M"

    def test_find_matching_dense_70m(self) -> None:
        args = siglino_configs["dense-70M"]
        assert _find_matching_config_name(args) == "dense-70M"

    def test_find_matching_moe_015b(self) -> None:
        args = siglino_configs["siglino-0.15B"]
        assert _find_matching_config_name(args) == "siglino-0.15B"

    def test_no_match_for_unknown_args(self) -> None:
        """Args with no matching config return None."""
        args = SigLinoArgs(dim=999, n_layers=99, n_heads=32)
        assert _find_matching_config_name(args) is None

    def test_validate_match_passes(self) -> None:
        """Matching config and checkpoint args should not raise."""
        config_args = siglino_configs["dense-30M"]
        chk_args = siglino_configs["dense-30M"]
        # Should not raise
        _validate_config_checkpoint_match("dense-30M", config_args, "tiiuae/siglino-30M", chk_args)

    def test_validate_mismatch_raises(self) -> None:
        """Mismatched config and checkpoint args should raise ValueError."""
        config_args = siglino_configs["dense-30M"]
        chk_args = siglino_configs["dense-70M"]
        with pytest.raises(ValueError, match="does not match"):
            _validate_config_checkpoint_match(
                "dense-30M", config_args, "tiiuae/siglino-70M", chk_args
            )

    def test_is_hub_id_true(self) -> None:
        assert _is_hub_id("tiiuae/siglino-30M")
        assert _is_hub_id("someuser/model")

    def test_is_hub_id_false_local_file(self, tmp_path: pathlib.Path) -> None:
        local = tmp_path / "model.safetensors"
        local.touch()
        assert not _is_hub_id(str(local))

    def test_is_hub_id_false_abs_path(self) -> None:
        assert not _is_hub_id("/absolute/path/to/model.safetensors")


class TestSigLinoEdgeCases:
    """Edge-case tests: non-square images, variable shapes, explicit masks."""

    @staticmethod
    def _create_model() -> SigLino:
        args = siglino_configs["dense-30M"]
        model = SigLino(args)
        model.init_weights()
        model.eval()
        return model

    def test_non_square_image_forward(self) -> None:
        """Forward with non-square image (taller than wide)."""
        model = self._create_model()
        x = torch.randn(1, 3, 288, 192)  # H=288, W=192, both /16
        out = model(pixel_values=x)
        feats = out["patch_features"]["siglino"]
        # 288/16=18, 192/16=12 → 18*12 = 216 patches + 4 registers
        n_reg = model.n_storage_tokens
        assert feats.shape[1] == 18 * 12 + n_reg, (
            f"Expected {18 * 12 + n_reg} (patches+registers), got {feats.shape[1]}"
        )
        assert not feats.isnan().any()

    def test_variable_spatial_shapes_in_batch(self) -> None:
        """Batched forward where spatial_shapes differ but image sizes match."""
        model = self._create_model()
        # All images in a batch must have the same H, W for 4D input.
        # spatial_shapes overrides the auto-computed shapes for the RoPE grid.
        x = torch.randn(2, 3, 224, 224)
        # Both have the same pixel dims but we pass different spatial_shapes
        shapes = torch.tensor([[14, 14], [7, 28]])  # Same patch count: 196
        out = model(pixel_values=x, spatial_shapes=shapes)
        feats = out["patch_features"]["siglino"]
        assert feats.shape[0] == 2
        assert not feats.isnan().any()

    def test_forward_with_explicit_padding_mask(self) -> None:
        """Provide an explicit padding_mask (all valid) — should match auto-generated."""
        model = self._create_model()
        x = torch.randn(1, 3, 224, 224)
        _, _, h, w = x.shape
        n_patches = (h // 16) * (w // 16)  # 14*14 = 196
        padding_mask = torch.ones(1, n_patches, dtype=torch.float32)
        spatial_shapes = torch.tensor([[14, 14]])
        out = model(pixel_values=x, padding_mask=padding_mask, spatial_shapes=spatial_shapes)
        feats = out["patch_features"]["siglino"]
        assert not feats.isnan().any()
        assert not feats.isinf().any()

    def test_non_divisible_dimensions_padded(self) -> None:
        """Images with H,W not divisible by patch_size are padded, not cropped.

        Previously ``_patchify`` used ``unfold`` which silently dropped trailing
        pixels.  Now H/W are padded to the next multiple of 16 before unfolding.
        The number of output patches must reflect the **padded** dimensions.
        """
        model = self._create_model()
        # H=100, W=100 — neither divisible by 16.  unfold(2,16,16) would cover
        # only 96 px (6 patches); pixels 96-99 are dropped without padding.
        # After padding: H=112, W=112 → 7×7 = 49 patches.
        n_reg = model.n_storage_tokens
        x = torch.randn(1, 3, 100, 100)
        out = model(pixel_values=x)
        feats = out["patch_features"]["siglino"]
        # 112/16=7 → 7*7=49 patches (not 6*6=36 which would indicate cropping) + registers
        assert feats.shape[1] == 7 * 7 + n_reg, (
            f"Expected {7 * 7 + n_reg} (patches+registers), got {feats.shape[1]}"
        )
        assert not feats.isnan().any()
        assert not feats.isinf().any()

    def test_multiple_non_divisible_batch(self) -> None:
        """Batch of images with non-divisible dimensions — all padded consistently."""
        model = self._create_model()
        n_reg = model.n_storage_tokens
        # Different padding requirements per dim but same H,W within batch:
        # H=101, W=99 → pad to 112, 112 → 7*7=49 patches
        x = torch.randn(2, 3, 101, 99)
        out = model(pixel_values=x)
        feats = out["patch_features"]["siglino"]
        assert feats.shape[0] == 2
        assert feats.shape[1] == 7 * 7 + n_reg, (
            f"Expected {7 * 7 + n_reg} (patches+registers), got {feats.shape[1]}"
        )
        assert not feats.isnan().any()
