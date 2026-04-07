from __future__ import annotations

import torch
from torch import nn
from typing import Optional

class ObfuscatedLayer(nn.Module):
    """Base class for all obfuscated layers."""
    def __init__(self, recorder: Optional[object] = None, record_name: Optional[str] = None):
        super().__init__()
        self.recorder = recorder
        self.record_name = record_name

    def record(self, value: torch.Tensor):
        if self.recorder is not None and self.record_name is not None:
            self.recorder.record(self.record_name, value)
