"""Tests for DopamineVLA — no SmolVLM dependencies, fully self-contained."""

import pytest
import torch

from dopaminevla.models.dopamine.dopamine import (
    DopamineVLABaseModelOutputWithPast,
    DopamineVLACausalLMOutputWithPast,
    DopamineVLAConfig,
    DopamineVLAConnector,
    DopamineVLAForConditionalGeneration,
    DopamineVLAModel,
    DopamineVLAVisionConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB = 32000
IMG_TOKEN_ID = VOCAB - 1
PAD_TOKEN_ID = 0

TEXT_CFG = {
    "hidden_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 512,
    "vocab_size": VOCAB,
}

VISION_CFG = dict(
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    head_dim=32,
    num_key_value_heads=4,
    spatial_patch_size=16,
    ffn_dim=256,
    max_seq_len=256,
)


def _make_config(**overrides: object) -> DopamineVLAConfig:
    # Separate vision-field overrides (SigLinoConfig parameters)
    vis_fields = {
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "head_dim",
        "num_key_value_heads",
        "spatial_patch_size",
        "ffn_dim",
        "max_seq_len",
    }
    vis_kw = {k: overrides.pop(k) for k in list(overrides) if k in vis_fields}
    vision_config = DopamineVLAVisionConfig(**{**VISION_CFG, **vis_kw})

    # Separate text-field overrides
    text_fields = {
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "vocab_size",
    }
    text_kw = {k: overrides.pop(k) for k in list(overrides) if k in text_fields}
    text_cfg = {**TEXT_CFG, **text_kw}

    return DopamineVLAConfig(
        vision_config=vision_config,
        text_config=text_cfg,
        image_token_id=IMG_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        vision_connector_n_latents=8,
        vision_connector_n_layers=2,
        vision_connector_n_heads=4,
        vision_connector_head_dim=32,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_creation(self) -> None:
        config = DopamineVLAConfig()
        assert config.model_type == "dopaminevla"
        assert config.vision_config is not None
        assert config.text_config is not None
        assert config.use_cache is True
        assert config.vision_connector_n_latents == 64

    def test_custom_values(self) -> None:
        vision = DopamineVLAVisionConfig(**VISION_CFG)
        config = DopamineVLAConfig(
            vision_config=vision,
            text_config=dict(**TEXT_CFG),
            vision_connector_n_latents=16,
            use_cache=False,
        )
        assert config.vision_connector_n_latents == 16
        assert config.use_cache is False

    def test_vision_config_subclass(self) -> None:
        vision = DopamineVLAVisionConfig()
        assert vision.model_type == "dopaminevla_vision"
        assert vision.spatial_patch_size == 16  # inherited from SigLinoConfig

    def test_text_config_resolved(self) -> None:
        config = _make_config()
        assert hasattr(config.text_config, "hidden_size")
        assert config.text_config.hidden_size == 256

    def test_image_token_id(self) -> None:
        vision = DopamineVLAVisionConfig(**VISION_CFG)
        config = DopamineVLAConfig(
            vision_config=vision,
            text_config=dict(**TEXT_CFG),
            image_token_id=42,
        )
        assert config.image_token_id == 42

    def test_serialization_roundtrip(self) -> None:
        config = _make_config()
        config_dict = config.to_dict()
        restored = DopamineVLAConfig.from_dict(config_dict)
        assert restored.image_token_id == config.image_token_id
        assert restored.vision_connector_n_latents == config.vision_connector_n_latents
        assert restored.text_config.hidden_size == config.text_config.hidden_size


# ---------------------------------------------------------------------------
# Output dataclass tests
# ---------------------------------------------------------------------------


class TestOutputs:
    def test_base_model_output_fields(self) -> None:
        out = DopamineVLABaseModelOutputWithPast(
            last_hidden_state=torch.randn(1, 10, 256),
        )
        assert out.last_hidden_state.shape == (1, 10, 256)
        assert out.past_key_values is None
        assert out.image_hidden_states is None

    def test_causal_lm_output_fields(self) -> None:
        out = DopamineVLACausalLMOutputWithPast(
            loss=torch.tensor(1.0),
            logits=torch.randn(1, 10, 32000),
        )
        assert out.loss is not None
        assert out.logits.shape == (1, 10, VOCAB)
        assert out.past_key_values is None


# ---------------------------------------------------------------------------
# Connector shape tests
# ---------------------------------------------------------------------------


class TestConnector:
    def test_output_shape(self) -> None:
        config = _make_config()
        connector = DopamineVLAConnector(config)
        B = 2
        views = (
            torch.randn(B, 50, config.vision_config.hidden_size),
            torch.randn(B, 25, config.vision_config.hidden_size),
            torch.randn(B, 30, config.vision_config.hidden_size),
        )
        out = connector(views)
        expected_latents = config.vision_connector_n_latents
        expected_dim = config.text_config.hidden_size
        assert out.shape == (B, expected_latents, expected_dim), f"Got {out.shape}"

    def test_single_view(self) -> None:
        config = _make_config()
        connector = DopamineVLAConnector(config)
        B = 1
        views = (torch.randn(B, 100, config.vision_config.hidden_size),)
        out = connector(views)
        assert out.shape == (B, config.vision_connector_n_latents, config.text_config.hidden_size)

    def test_with_attention_mask(self) -> None:
        config = _make_config()
        connector = DopamineVLAConnector(config)
        B = 2
        views = (torch.randn(B, 50, config.vision_config.hidden_size),)
        masks = (torch.ones(B, 50, dtype=torch.bool),)
        out = connector(views, attention_masks=masks)
        assert out.shape == (B, config.vision_connector_n_latents, config.text_config.hidden_size)


# ---------------------------------------------------------------------------
# Model forward tests
# ---------------------------------------------------------------------------


class TestDopamineVLAModel:
    def test_create_model(self) -> None:
        config = _make_config()
        model = DopamineVLAModel(config)
        assert isinstance(model.vision_model, DopamineVLAVisionConfig | object)
        assert isinstance(model.connector, DopamineVLAConnector)
        assert hasattr(model.text_model, "forward")

    def test_forward_text_only(self) -> None:
        config = _make_config()
        model = DopamineVLAModel(config)
        model.eval()
        input_ids = torch.tensor([[1, 5, 10, 2]])
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.last_hidden_state.shape == (1, 4, 256)

    def test_forward_with_images(self) -> None:
        config = _make_config()
        model = DopamineVLAModel(config)
        model.eval()
        input_ids = torch.tensor([[1] + [IMG_TOKEN_ID] * 8 + [2]])
        pixel_values = torch.randn(1, 1, 3, 224, 224)
        with torch.no_grad():
            out = model(pixel_values=pixel_values, input_ids=input_ids)
        assert out.last_hidden_state.shape == (1, 10, 256)

    def test_forward_with_precomputed_features(self) -> None:
        config = _make_config()
        model = DopamineVLAModel(config)
        model.eval()
        # Simulate pre-computed image features (connector output)
        image_hidden_states = torch.randn(1, 8, config.text_config.hidden_size)
        input_ids = torch.tensor([[1] + [IMG_TOKEN_ID] * 8 + [2]])
        with torch.no_grad():
            out = model(input_ids=input_ids, image_hidden_states=image_hidden_states)
        assert out.last_hidden_state.shape == (1, 10, 256)

    def test_mutual_exclusion_pixel_and_features(self) -> None:
        config = _make_config()
        model = DopamineVLAModel(config)
        model.eval()
        pixel_values = torch.randn(1, 1, 3, 224, 224)
        image_hidden_states = torch.randn(1, 8, config.text_config.hidden_size)
        input_ids = torch.tensor([[1] + [IMG_TOKEN_ID] * 8 + [2]])
        with pytest.raises(ValueError), torch.no_grad():
            model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_hidden_states=image_hidden_states,
            )

    def test_batched_images(self) -> None:
        config = _make_config()
        model = DopamineVLAModel(config)
        model.eval()
        # Batch of 2, each with 1 image (same number of image tokens)
        pixel_values = torch.randn(2, 1, 3, 224, 224)
        input_ids = torch.tensor(
            [
                [1] + [IMG_TOKEN_ID] * 8 + [2],
                [1] + [IMG_TOKEN_ID] * 8 + [2],
            ]
        )
        with torch.no_grad():
            out = model(pixel_values=pixel_values, input_ids=input_ids)
        assert out.last_hidden_state.shape[0] == 2

    def test_gradient_flows(self) -> None:
        config = _make_config()
        model = DopamineVLAModel(config)
        input_ids = torch.tensor([[1] + [IMG_TOKEN_ID] * 8 + [2]])
        pixel_values = torch.randn(1, 1, 3, 224, 224)
        out = model(pixel_values=pixel_values, input_ids=input_ids)
        loss = out.last_hidden_state.mean()
        loss.backward()
        # Check some gradients exist
        assert model.connector.modality_projection.weight.grad is not None


# ---------------------------------------------------------------------------
# Conditional generation tests
# ---------------------------------------------------------------------------


class TestDopamineVLAForConditionalGeneration:
    def test_create_model(self) -> None:
        config = _make_config()
        model = DopamineVLAForConditionalGeneration(config)
        assert hasattr(model, "lm_head")
        assert model.lm_head.out_features == VOCAB

    def test_forward_no_labels(self) -> None:
        config = _make_config()
        model = DopamineVLAForConditionalGeneration(config)
        model.eval()
        input_ids = torch.tensor([[1, 5, 10, 2]])
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (1, 4, VOCAB)
        assert out.loss is None

    def test_forward_with_labels(self) -> None:
        config = _make_config()
        model = DopamineVLAForConditionalGeneration(config)
        model.eval()
        input_ids = torch.tensor([[1] + [IMG_TOKEN_ID] * 8 + [2]])
        pixel_values = torch.randn(1, 1, 3, 224, 224)
        labels = torch.tensor([[-100] * 9 + [2]])  # predict only the last token
        with torch.no_grad():
            out = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels,
            )
        assert out.loss is not None
        assert out.logits.shape == (1, 10, VOCAB)

    def test_get_image_features_delegates(self) -> None:
        config = _make_config()
        model = DopamineVLAForConditionalGeneration(config)
        model.eval()
        pixel_values = torch.randn(1, 1, 3, 224, 224)
        with torch.no_grad():
            features = model.get_image_features(pixel_values=pixel_values)
        # Connector output shape: (n_real_images, n_latents, text_hidden_size)
        assert features.shape == (1, 8, config.text_config.hidden_size)

    def test_forward_with_precomputed_features(self) -> None:
        config = _make_config()
        model = DopamineVLAForConditionalGeneration(config)
        model.eval()
        image_hidden_states = torch.randn(1, 8, config.text_config.hidden_size)
        input_ids = torch.tensor([[1] + [IMG_TOKEN_ID] * 8 + [2]])
        labels = torch.tensor([[-100] * 9 + [2]])
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                labels=labels,
                image_hidden_states=image_hidden_states,
            )
        assert out.loss is not None

    def test_prepare_inputs_for_generation_first_iter(self) -> None:
        config = _make_config()
        model = DopamineVLAForConditionalGeneration(config)
        model.eval()
        input_ids = torch.tensor([[1] + [IMG_TOKEN_ID] * 8 + [2]])
        pixel_values = torch.randn(1, 1, 3, 224, 224)
        prepared = model.prepare_inputs_for_generation(
            input_ids,
            pixel_values=pixel_values,
            cache_position=torch.tensor([0]),
        )
        # First iteration with pixel_values -> should be forwarded
        assert "pixel_values" in prepared

    def test_prepare_inputs_for_generation_subsequent(self) -> None:
        config = _make_config()
        model = DopamineVLAForConditionalGeneration(config)
        model.eval()
        input_ids = torch.tensor([[5]])
        prepared = model.prepare_inputs_for_generation(
            input_ids,
            cache_position=torch.tensor([10]),
            image_hidden_states=torch.randn(1, 8, config.text_config.hidden_size),
        )
        # Subsequent iteration: pixel_values should be None (already encoded)
        assert prepared.get("pixel_values") is None
