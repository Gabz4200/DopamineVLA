"""DopamineVLA inference example.

Initializes the model from config (no pre-trained weights) and runs
generation with real image inputs. Output will be random — this is
infrastructure preparation for when weights become available.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer
from transformers.image_utils import load_image

from dopaminevla.models.dopamine import (
    DopamineVLAConfig,
    DopamineVLAForConditionalGeneration,
)
from dopaminevla.models.siglino.siglino.image_processor import IMAGE_MEAN, IMAGE_STD, smart_resize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

PATCH_SIZE = 16
MIN_PIXELS = 128 * 128
MAX_PIXELS = 256 * 256


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize + rescale + normalize a PIL image to ``(3, H, W)``.

    ``H``, ``W`` are divisible by ``PATCH_SIZE`` (16).
    Normalization: ``(pixel / 255 - mean) / std`` with ``mean = std = 0.5``.
    """
    image = image.convert("RGB")
    w, h = image.size
    new_h, new_w = smart_resize(
        h, w, factor=PATCH_SIZE, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    arr = np.array(image).astype(np.float32) / 255.0
    mean = np.array(IMAGE_MEAN, dtype=np.float32)
    std = np.array(IMAGE_STD, dtype=np.float32)
    arr = (arr - mean) / std

    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)


# ---------------------------------------------------------------------------
# Load images
# ---------------------------------------------------------------------------

print("Loading images...")
image1 = load_image(
    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
)
image2 = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")

raw_images = [image1, image2]
imgs = [preprocess_image(img) for img in raw_images]

# Pad to common spatial size so they can be batched
max_h = max(img.shape[1] for img in imgs)
max_w = max(img.shape[2] for img in imgs)
pixel_values_list = []
for img in imgs:
    h, w = img.shape[1:]
    pad_h, pad_w = max_h - h, max_w - w
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h))
    pixel_values_list.append(img)

pixel_values = torch.stack(pixel_values_list).unsqueeze(0)  # (1, N, 3, H, W)
print(f"  pixel_values: {tuple(pixel_values.shape)}")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Xenova/llama2-tokenizer")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Add image token so image_token_id is within the embedding table
image_token = "<image>"
tokenizer.add_tokens([image_token], special_tokens=True)
image_token_id = tokenizer.convert_tokens_to_ids(image_token)
print(f"  image_token_id: {image_token_id}")
print(f"  vocab_size: {len(tokenizer)}")

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

config = DopamineVLAConfig(
    vision_config={
        "hidden_size": 512,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "head_dim": 64,
        "spatial_patch_size": PATCH_SIZE,
    },
    text_config={
        "model_type": "llama",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "vocab_size": len(tokenizer),
    },
    image_token_id=image_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

print("Creating model...")
model = DopamineVLAForConditionalGeneration._from_config(config)
model.to(DEVICE)
model.eval()
print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------------------
# Prepare inputs
# ---------------------------------------------------------------------------

print("Preparing inputs...")
n_latents = config.vision_connector_n_latents  # 64 per image
prompt = "Can you describe the two images?"

# Insert image_token_id placeholders at the start — n_latents per image
image_placeholder = torch.full((1, n_latents * 2), image_token_id, dtype=torch.long)
text_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
input_ids = torch.cat([image_placeholder, text_ids], dim=1)
attention_mask = torch.ones_like(input_ids)

input_ids = input_ids.to(DEVICE)
attention_mask = attention_mask.to(DEVICE)
pixel_values = pixel_values.to(DEVICE)

print(f"  input_ids: {tuple(input_ids.shape)}")
image_token_count = (input_ids == image_token_id).sum().item()
print(f"  image_token count: {image_token_count}  (expected: {2 * n_latents})")

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

print("\nGenerating (random weights — output is garbage)...")
with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_new_tokens=50,
        do_sample=False,
        use_cache=True,
    )

generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
print(f"\nGenerated token IDs:\n  {generated_ids[0].tolist()}")
print(f"\nDecoded:\n  {generated_text[0]}")
