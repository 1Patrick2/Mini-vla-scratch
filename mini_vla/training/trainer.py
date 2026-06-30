"""Training orchestration for Mini VLA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Trainer:
    """Stage 0 trainer shell.

    The real behavior cloning loop is intentionally deferred to Stage 3.
    """

    config: dict[str, Any]

    def describe(self) -> str:
        model_name = self.config["model"]["name"]
        dataset_type = self.config["data"]["dataset_type"]
        epochs = self.config["train"]["epochs"]
        return f"Trainer(model={model_name}, data={dataset_type}, epochs={epochs})"
