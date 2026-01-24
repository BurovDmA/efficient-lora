from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lightning import Callback, LightningModule, Trainer


@dataclass
class SavePeftAdapterCallback(Callback):
    """Save PEFT (LoRA) adapters during/after training.

    This avoids saving huge full-model checkpoints. It calls `model.save_pretrained(...)` on
    the underlying PEFT-wrapped model.
    """

    dirpath: str
    every_n_epochs: int = 1
    save_on_train_end: bool = True

    def _save(self, trainer: Trainer, pl_module: LightningModule, tag: str) -> None:
        out_dir = Path(self.dirpath) / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        model = getattr(pl_module, "model", None)
        if model is None:
            return

        # PEFT models implement save_pretrained (adapter-only)
        save_fn = getattr(model, "save_pretrained", None)
        if callable(save_fn):
            save_fn(str(out_dir))

        # Save tokenizer too if present
        tok = getattr(trainer.datamodule, "_tokenizer", None)
        if tok is not None and callable(getattr(tok, "save_pretrained", None)):
            tok.save_pretrained(str(out_dir))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.every_n_epochs <= 0:
            return
        epoch = int(getattr(trainer, "current_epoch", 0))
        if (epoch + 1) % int(self.every_n_epochs) != 0:
            return
        self._save(trainer, pl_module, tag=f"epoch_{epoch:03d}")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.save_on_train_end:
            return
        self._save(trainer, pl_module, tag="final")
