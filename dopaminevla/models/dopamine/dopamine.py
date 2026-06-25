import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationConfig
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.smolvlm.configuration_smolvlm import SmolVLMConfig, SmolVLMVisionConfig
from transformers.models.smolvlm.image_processing_pil_smolvlm import SmolVLMImageProcessorPil
from transformers.models.smolvlm.image_processing_smolvlm import SmolVLMImageProcessor
from transformers.models.smolvlm.modeling_smolvlm import (
    SmolVLMBaseModelOutputWithPast,
    SmolVLMCausalLMOutputWithPast,
    SmolVLMForConditionalGeneration,
    SmolVLMModel,
    SmolVLMPreTrainedModel,
    SmolVLMVisionTransformer,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check

logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="HuggingFaceTB/SmolVLM2-2.2B-Instruct")  # todo: Make a hub entry for DopamineVLA
@strict
class DopamineVLAVisionConfig(SmolVLMVisionConfig):
    r"""
    Example:

    ```python
    >>> from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionTransformer
    >>> from transformers.models.smolvlm.configuration_smolvlm import SmolVLMVisionConfig

    >>> # Initializing a SmolVLMVisionConfig with google/siglip-so400m-patch14-384 style configuration
    >>> configuration = SmolVLMVisionConfig()

    >>> # Initializing a SmolVLMVisionTransformer (with random weights) from the google/siglip-so400m-patch14-384 style configuration
    >>> model = SmolVLMVisionTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "dopaminevla_vision"


class DopamineVLAPreTrainedModel(SmolVLMPreTrainedModel):
    pass


class DopamineVLAVisionTransformer(SmolVLMVisionTransformer):
    pass


@auto_docstring(checkpoint="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
@strict
class DopamineVLAConfig(SmolVLMConfig):
    r"""
    scale_factor (`int`, *optional*, defaults to 2):
        The scale factor for the image encoder.

    Example:
    ```python
    >>> from transformers import SmolVLMModel, SmolVLMConfig
    >>> # Initializing configuration
    >>> configuration = SmolVLMConfig()
    >>> # Initializing a model from the configuration
    >>> model = SmolVLMModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "dopaminevla"


class DopamineVLAImageProcessor(SmolVLMImageProcessor):
    pass


class DopamineVLAImageProcessorPil(SmolVLMImageProcessorPil):
    pass


class DopamineVLABaseModelOutputWithPast(SmolVLMBaseModelOutputWithPast):
    pass


class DopamineVLAModel(SmolVLMModel):
    """
    A subclass of SmolVLMModel. We do *not* remove or block the call to inputs_merger
    in forward. Instead, we override inputs_merger here with custom logic.
    """

    def inputs_merger(self, input_ids: torch.LongTensor | None, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor) -> torch.Tensor:
        _, patch_size, _ = image_hidden_states.shape

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device))
            image_mask = image_mask[..., 0]  # slice off the hidden dim
        else:
            image_mask = input_ids == self.config.image_token_id

        num_image_tokens = image_mask.sum(dim=1)
        torch_compilable_check(
            torch.all(num_image_tokens % patch_size == 0),
            "At least one sample has <image> tokens not divisible by patch_size.",
        )
        blocks_per_sample = num_image_tokens // patch_size

        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx

        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]

        merged_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)
        return merged_embeds

    @can_return_tuple
    @auto_docstring(custom_intro="Encodes images into continuous embeddings that can be forwarded to the language model.")
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, ...] | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        pixel_attention_mask (`torch.LongTensor`, *optional*):
            The attention mask indicating padded regions in the image.
        """
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

        # If no images, leave one empty image.
        real_images_inds[0] |= ~torch.any(real_images_inds)

        pixel_values = pixel_values[real_images_inds].contiguous()
        # Handle the vision attention mask
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=[pixel_values.shape[i] for i in (0, 2, 3)],
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            # Remove padding images from the mask
            pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()
        patch_size = self.config.vision_config.patch_size  # pyrefly: ignore[missing-attribute]  # TODO: PretrainedConfig type is too broad
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Get sequence from the vision encoder
        image_outputs = self.vision_model(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask, return_dict=True, **kwargs)
        image_hidden_states = image_outputs.last_hidden_state

        # Modality projection & resampling
        image_features = self.connector(image_hidden_states)
        image_outputs.pooler_output = image_features

        return image_outputs

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """
    )
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        image_hidden_states: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, ...] | DopamineVLABaseModelOutputWithPast:
        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")

        if pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask, return_dict=True).pooler_output
            image_hidden_states = image_hidden_states.to(inputs_embeds.device)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=inputs_embeds.device)

        if image_hidden_states is not None:
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        return DopamineVLABaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,  # pyrefly: ignore[bad-argument-type]  # TODO: upstream type mismatch (expects tuple)
        )


class DopamineVLAForConditionalGeneration(SmolVLMForConditionalGeneration):
    _tied_weights_keys = {"lm_head.weight": "model.text_model.embed_tokens.weight"}

    def __init__(self, config: DopamineVLAConfig) -> None:
        super().__init__(config)
        self.model = DopamineVLAModel(config)
        self.model.text_model.generation_config = GenerationConfig.from_model_config(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)  # pyrefly: ignore[missing-attribute]  # TODO: PretrainedConfig type too broad
        self.post_init()

    def forward(self, **super_kwargs: Unpack[TransformersKwargs]) -> tuple[torch.Tensor, ...] | SmolVLMCausalLMOutputWithPast:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> import httpx
        >>> from io import BytesIO
        >>> import torch
        >>> from PIL import Image
        >>> from io import BytesIO

        >>> from transformers import AutoProcessor, AutoModelForImageTextToText
        >>> from transformers.image_utils import load_image

        >>> # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
        >>> image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
        >>> image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
        >>> image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

        >>> processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        >>> model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", dtype=torch.bfloat16, device_map="auto")

        >>> # Create inputs
        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "video", "path": path/to/video},
        ...             {"type": "text", "text": "What is happening in this video?"},
        ...         ]
        ...     }
        ... ]

        >>> inputs = processor.apply_chat_template([messages], add_generation_prompt=True)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=256)
        >>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        >>> print(generated_texts)
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "DopamineVLAVisionConfig",
    "DopamineVLAConfig",
    "DopamineVLAImageProcessor",
    "DopamineVLAImageProcessorPil",
    "DopamineVLAForConditionalGeneration",
    "DopamineVLAPreTrainedModel",
    "DopamineVLAModel",
    "DopamineVLAVisionTransformer",
]
