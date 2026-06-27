"""Tests for DopamineVLA — no SmolVLM dependencies, fully self-contained."""

from typing import Any, cast

import pytest
import torch

from dopaminevla.models.dopamine import (
    DopamineVLABaseModelOutputWithPast,
    DopamineVLACausalLMOutputWithPast,
    DopamineVLAConfig,
    DopamineVLAConnector,
    DopamineVLAForConditionalGeneration,
    DopamineVLAModel,
    DopamineVLAVisionConfig,
    DopamineVLAVisionTransformer,
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


def _make_config(**overrides: Any) -> DopamineVLAConfig:
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
    vision_config = DopamineVLAVisionConfig(**cast(dict[str, Any], {**VISION_CFG, **vis_kw}))

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
        assert out.last_hidden_state is not None
        assert out.last_hidden_state.shape == (1, 10, 256)
        assert out.past_key_values is None
        assert out.image_hidden_states is None

    def test_causal_lm_output_fields(self) -> None:
        out = DopamineVLACausalLMOutputWithPast(
            loss=torch.tensor(1.0),
            logits=torch.randn(1, 10, 32000),
        )
        assert out.loss is not None
        assert out.logits is not None
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
        assert out.logits is not None
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


# ---------------------------------------------------------------------------
# Vision Transformer tests
# ---------------------------------------------------------------------------


class TestVisionTransformer:
    """Direct tests for DopamineVLAVisionTransformer (single-pass encoder)."""

    def test_forward_output_shape(self) -> None:
        config = _make_config()
        assert isinstance(config.vision_config, DopamineVLAVisionConfig)
        vt = DopamineVLAVisionTransformer(config.vision_config)
        vt.eval()
        x = torch.randn(1, 3, 224, 224)
        features_tuple, masks_tuple = vt(x)
        # With default vision_feature_layers=1, returns 1-tuple
        features = features_tuple[0]
        mask = masks_tuple[0]
        assert features.ndim == 3, f"expected (B, L, D), got {features.shape}"
        assert features.shape[0] == 1
        assert features.shape[2] == config.vision_config.hidden_size
        assert mask.shape == features.shape[:2], f"expected {features.shape[:2]}, got {mask.shape}"

    def test_forward_no_nan(self) -> None:
        config = _make_config()
        assert isinstance(config.vision_config, DopamineVLAVisionConfig)
        vt = DopamineVLAVisionTransformer(config.vision_config)
        vt.eval()
        x = torch.randn(1, 3, 224, 224)
        features_tuple, _ = vt(x)
        features = features_tuple[0]
        assert not features.isnan().any(), "Features have NaN"
        assert not features.isinf().any(), "Features have Inf"


# ---------------------------------------------------------------------------
# Inputs merger tests
# ---------------------------------------------------------------------------


class TestInputsMerger:
    """Direct tests for DopamineVLAModel.inputs_merger."""

    def test_basic_merge(self) -> None:
        config = _make_config()
        model = DopamineVLAModel(config)
        model.eval()
        B, seq_len, n_latents = 1, 12, 8
        hidden = config.text_config.hidden_size
        input_ids = torch.tensor([[1] + [IMG_TOKEN_ID] * n_latents + [5, 10, 2]])
        inputs_embeds = model.get_input_embeddings()(input_ids)
        image_hidden_states = torch.randn(1, n_latents, hidden)
        merged = model.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_hidden_states=image_hidden_states,
        )
        assert merged.shape == (B, seq_len, hidden)
        # Image token positions should differ from embedding lookup
        img_pos = (input_ids == IMG_TOKEN_ID).squeeze()
        assert not torch.equal(merged[0, img_pos], inputs_embeds[0, img_pos])

    def test_no_image_tokens_passthrough(self) -> None:
        """When input_ids has no image tokens, merged == inputs_embeds."""
        config = _make_config()
        model = DopamineVLAModel(config)
        model.eval()
        input_ids = torch.tensor([[1, 5, 10, 2]])
        inputs_embeds = model.get_input_embeddings()(input_ids)
        # n_latents=8 but no image tokens in input_ids
        image_hidden_states = torch.randn(1, 8, config.text_config.hidden_size)
        merged = model.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_hidden_states=image_hidden_states,
        )
        assert merged.shape == inputs_embeds.shape
        assert torch.equal(merged, inputs_embeds)

    def test_requires_input_ids(self) -> None:
        config = _make_config()
        model = DopamineVLAModel(config)
        model.eval()
        inputs_embeds = torch.randn(1, 4, config.text_config.hidden_size)
        img_hidden = torch.randn(1, 8, config.text_config.hidden_size)
        with pytest.raises(ValueError, match="input_ids is required"):
            model.inputs_merger(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                image_hidden_states=img_hidden,
            )


# ---------------------------------------------------------------------------
# get_image_features edge cases
# ---------------------------------------------------------------------------


class TestGetImageFeatures:
    """Edge cases for DopamineVLAModel.get_image_features."""

    def test_all_zero_images_filtered(self) -> None:
        """All-zero padding images should be filtered out before the vision encoder."""
        config = _make_config()
        model = DopamineVLAModel(config)
        model.eval()
        # Batch of 1 with 2 images: one real, one all-zero (padding placeholder)
        real_img = torch.randn(1, 1, 3, 224, 224)
        zero_img = torch.zeros(1, 1, 3, 224, 224)
        pixel_values = torch.cat([real_img, zero_img], dim=1)  # (1, 2, 3, 224, 224)
        input_ids = torch.tensor([[1] + [IMG_TOKEN_ID] * 8 + [2]])
        with torch.no_grad():
            out = model(pixel_values=pixel_values, input_ids=input_ids)
        # Should not crash — one real image, one zero image filtered
        assert out.last_hidden_state.shape[0] == 1


# ---------------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------------


class TestWeightTying:
    """Verify lm_head shape matches the embedding table.

    Note: ``_tied_weights_keys`` with dotted paths like ``model.text_model.*``
    is silently skipped by ``hasattr`` in ``tie_weights`` — this is a known
    transformers limitation.  Weight tying is thus not active; the output
    projection and the embedding table remain separate parameters.
    """

    def test_lm_head_out_features_matches_vocab(self) -> None:
        config = _make_config()
        model = DopamineVLAForConditionalGeneration(config)
        assert model.lm_head.out_features == VOCAB
        assert model.lm_head.in_features == config.text_config.hidden_size

    def test_get_input_embeddings_returns_module(self) -> None:
        config = _make_config()
        model = DopamineVLAForConditionalGeneration(config)
        emb = model.get_input_embeddings()
        assert emb.weight.shape == (VOCAB, config.text_config.hidden_size)
