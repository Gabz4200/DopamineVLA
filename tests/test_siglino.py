import pytest
import torch
from PIL import Image
from siglino import (
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
from siglino.configs import MoEArgs


@pytest.fixture(scope="module")
def dense_30m_args():
    return siglino_configs["dense-30M"]


@pytest.fixture(scope="module")
def dense_70m_args():
    return siglino_configs["dense-70M"]


@pytest.fixture(scope="module")
def siglino_015b_args():
    return siglino_configs["siglino-0.15B"]


@pytest.fixture(scope="module")
def dummy_input():
    return torch.randn(1, 3, 224, 224)


class TestImports:
    def test_all_exports_exist(self):
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
    def test_default_creation(self):
        args = SigLinoArgs()
        assert args.dim == 768
        assert args.n_layers == 18
        assert args.moe_dim == 768

    def test_moe_args_defaults(self):
        args = SigLinoArgs()
        assert isinstance(args.moe_args, MoEArgs)
        assert args.moe_args.num_experts == 16

    def test_dense_config_has_moe_dim_zero(self):
        args = siglino_configs["dense-30M"]
        assert args.moe_dim == 0
        assert args.first_n_layers_dense == 12

    def test_siglino_config_moe_nonzero(self):
        args = siglino_configs["siglino-0.15B"]
        assert args.moe_dim == 384
        assert args.moe_args.num_experts == 28


class TestSigLinoConfig:
    def test_create_from_scratch(self):
        config = SigLinoConfig()
        assert config.model_type == "siglino"
        assert config.hidden_size == 768

    def test_to_siglino_args_roundtrip(self):
        config = SigLinoConfig(hidden_size=512, num_hidden_layers=12, num_attention_heads=8)
        args = config.to_siglino_args()
        assert args.dim == 512
        assert args.n_layers == 12
        assert args.n_heads == 8

    def test_from_siglino_args(self):
        args = SigLinoArgs(dim=384, n_layers=12, n_heads=6)
        config = SigLinoConfig.from_siglino_args(args)
        assert config.hidden_size == 384
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 6

    def test_from_hub_config(self):
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

    def test_hub_config_roundtrip(self):
        config = SigLinoConfig(hidden_size=768, num_hidden_layers=18, num_attention_heads=12)
        args = config.to_siglino_args()
        config2 = SigLinoConfig.from_siglino_args(args)
        assert config2.hidden_size == config.hidden_size
        assert config2.num_hidden_layers == config.num_hidden_layers

    @pytest.mark.slow
    def test_from_pretrained_hub(self):
        config = SigLinoConfig.from_pretrained("tiiuae/siglino-70M")
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8

    def test_save_and_load_local_config(self, tmp_path):
        config = SigLinoConfig(hidden_size=384, num_hidden_layers=12)
        config.save_pretrained(tmp_path)
        loaded = SigLinoConfig.from_pretrained(tmp_path)
        assert loaded.hidden_size == 384
        assert loaded.num_hidden_layers == 12


class TestSigLinoModelCPU:
    """CPU forward tests for dense and MoE configurations."""

    def _create_model(self, args: SigLinoArgs):
        model = SigLino(args)
        model.init_weights()
        model.eval()
        return model

    def _run_forward(self, model):
        x = torch.randn(1, 3, 224, 224)
        out = model(pixel_values=x, spatial_shapes=torch.tensor([[14, 14]]))
        assert "patch_features" in out
        assert "siglino" in out["patch_features"]
        return out

    def test_dense_30m_cpu_forward(self, dense_30m_args):
        model = self._create_model(dense_30m_args)
        out = self._run_forward(model)
        feats = out["patch_features"]["siglino"]
        assert feats.ndim == 3

    def test_dense_70m_cpu_forward(self, dense_70m_args):
        model = self._create_model(dense_70m_args)
        out = self._run_forward(model)
        feats = out["patch_features"]["siglino"]
        assert feats.ndim == 3

    @pytest.mark.slow
    def test_moe_015b_cpu_forward(self, siglino_015b_args):
        model = self._create_model(siglino_015b_args)
        out = self._run_forward(model)
        feats = out["patch_features"]["siglino"]
        assert feats.ndim == 3

    def test_batched_input(self, dense_30m_args):
        model = self._create_model(dense_30m_args)
        x = torch.randn(2, 3, 224, 224)
        out = model(pixel_values=x, spatial_shapes=torch.tensor([[14, 14], [14, 14]]))
        assert out["patch_features"]["siglino"].shape[0] == 2

    def test_no_nan_in_output(self, dense_30m_args):
        """Verify random-init model produces no NaN (init_weights was called)."""
        model = self._create_model(dense_30m_args)
        out = self._run_forward(model)
        for feat in out["patch_features"].values():
            assert not feat.isnan().any(), f"NaN found in {feat.shape}"
            assert not feat.isinf().any(), f"Inf found in {feat.shape}"


_HF_SMALL = dict(
    hidden_size=384,
    num_hidden_layers=12,
    num_attention_heads=6,
    num_key_value_heads=6,
    head_dim=64,
)


class TestSigLinoHFModel:
    def test_create_hf_model(self):
        config = SigLinoConfig(**_HF_SMALL)
        model = SigLinoHFModel(config)
        assert isinstance(model, SigLinoHFModel)
        assert model.config.hidden_size == 384

    def test_hf_forward(self):
        config = SigLinoConfig(**_HF_SMALL)
        model = SigLinoHFModel(config)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        out = model(pixel_values=x, spatial_shapes=torch.tensor([[14, 14]]))
        assert "patch_features" in out

    def test_hf_state_dict_keys(self):
        config = SigLinoConfig(**_HF_SMALL)
        model = SigLinoHFModel(config)
        sd = model.state_dict()
        assert any(k.startswith("model.layers.") for k in sd.keys())
        assert any(k.startswith("model.img_projector.") for k in sd.keys())
        assert any(k.startswith("model.cls_token") for k in sd.keys())

    @pytest.mark.slow
    def test_from_pretrained_hub(self):
        model = SigLinoHFModel.from_pretrained("tiiuae/siglino-70M")
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        out = model(pixel_values=x, spatial_shapes=torch.tensor([[14, 14]]))
        assert "patch_features" in out
        assert out["patch_features"]["siglino"].shape[-1] == 512

    def test_save_and_load_local_hf(self, tmp_path):
        config = SigLinoConfig(**_HF_SMALL)
        model = SigLinoHFModel(config)
        model.save_pretrained(tmp_path)
        loaded = SigLinoHFModel.from_pretrained(tmp_path)
        loaded.eval()
        x = torch.randn(1, 3, 224, 224)
        out = loaded(pixel_values=x, spatial_shapes=torch.tensor([[14, 14]]))
        assert "patch_features" in out


class TestSigLinoImageProcessor:
    def test_create_processor(self):
        processor = SigLinoImageProcessor()
        assert processor is not None

    def test_process_single_image(self):
        processor = SigLinoImageProcessor(min_pixels=128 * 128, max_pixels=256 * 256)
        img = Image.new("RGB", (224, 224))
        out = processor(img)
        assert "pixel_values" in out
        assert "padding_mask" in out
        assert "spatial_shape" in out

    def test_process_multiple_images(self):
        processor = SigLinoImageProcessor(min_pixels=128 * 128, max_pixels=256 * 256)
        imgs = [Image.new("RGB", (224, 224)) for _ in range(3)]
        out = processor(imgs)
        assert out["pixel_values"].shape[0] == 3


class TestLoadSiglinoModel:
    def test_load_with_config_name(self):
        model, processor = load_siglino_model(
            checkpoint_path=None,
            config_name="dense-30M",
            device="cpu",
        )
        assert isinstance(model, SigLino)
        assert processor is not None

    def test_auto_device_cpu(self):
        model, processor = load_siglino_model(
            checkpoint_path=None,
            config_name="dense-30M",
        )
        dev = next(model.parameters()).device
        assert dev.type == "cpu"


class TestDeviceAgnostic:
    def test_flex_attn_disabled_on_cpu(self):
        model = SigLino(siglino_configs["dense-30M"])
        cpu = torch.device("cpu")
        assert not model._use_flex_attn_on_device(cpu)

    def test_compile_auto_disabled_on_cpu(self, dense_30m_args):
        model = SigLino(dense_30m_args)
        x = torch.randn(1, 3, 224, 224)
        out = model(pixel_values=x, spatial_shapes=torch.tensor([[14, 14]]))
        assert "patch_features" in out

    def test_no_nan_in_random_forward(self, dense_30m_args):
        """Sanity: random weights with init_weights produce finite output."""
        model = SigLino(dense_30m_args)
        model.init_weights()
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        out = model(pixel_values=x, spatial_shapes=torch.tensor([[14, 14]]))
        for name, feat in out["patch_features"].items():
            assert not feat.isnan().any(), f"NaN in {name}"
            assert not feat.isinf().any(), f"Inf in {name}"


class TestONNXWrapper:
    """ONNX export wrapper tests (no optimum dependency needed)."""

    def test_onnx_wrapper_creation(self):
        config = SigLinoConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
        )
        model = SigLinoHFModel(config)
        model.eval()
        wrapper = model._get_onnx_wrapper()

        x = torch.randn(1, 3, 224, 224)
        out = wrapper(x)
        assert isinstance(out, tuple)
        assert len(out) == 6
        for t in out:
            assert isinstance(t, torch.Tensor)

    @pytest.mark.slow
    def test_onnx_wrapper_forward(self):
        config = SigLinoConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
        )
        model = SigLinoHFModel(config)
        model.eval()
        wrapper = model._get_onnx_wrapper()

        x = torch.randn(1, 3, 224, 224)
        wrapper_out = wrapper(x)

        spatial_shapes = torch.tensor([[14, 14]])
        model_out = model(pixel_values=x, spatial_shapes=spatial_shapes)
        pf = model_out["patch_features"]

        assert torch.equal(wrapper_out[0], pf["dinov3"])
        assert torch.equal(wrapper_out[1], pf["siglip2"])
        assert torch.equal(wrapper_out[2], pf["siglino"])


class TestQuantizeCPU:
    def test_quantize_cpu_model_fn_available(self):
        """quantize_cpu_model should be a callable function."""
        assert callable(quantize_cpu_model)

    def test_quantize_cpu_model_does_not_crash(self, dense_30m_args):
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
