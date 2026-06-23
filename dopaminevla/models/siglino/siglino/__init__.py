# Copyright (c) 2025 TII (Technology Innovation Institute)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Falcon Vision - Standalone Vision Encoder from Multi-Teacher Distillation
# A pure vision model distilled from DINOv3 and SigLIP2 teachers

from .configs import SigLinoArgs, siglino_configs
from .image_processor import SigLinoImageProcessor
from .model import SigLino
from .utils import load_siglino_model

__all__ = [
    "SigLino",
    "SigLinoArgs",
    "siglino_configs",
    "SigLinoImageProcessor",
    "load_siglino_model",
]
