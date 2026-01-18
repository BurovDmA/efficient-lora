from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from lightning import LightningModule


class CausalLMLoRALitModule(LightningModule):
    """LightningModule for HF CausalLM fine-tuning with LoRA adapters (PEFT).

    Notes:
    - By default saves only LoRA weights via `peft` (use callback) to avoid huge checkpoints.
    - Uses standard `loss` from `AutoModelForCausalLM` (labels are expected in batch).
    """

    def __init__(
        self,
        model_name_or_path: str,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        compile: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: Optional[str] = None,  # "auto" | "bf16" | "fp16" | "fp32"
        gradient_checkpointing: bool = True,
        use_cache: bool = False,
        # LoRA
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list[str]] = None,
        lora_bias: str = "none",
        # Optional: load existing adapter for continued training
        lora_adapter_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = None

    def _resolve_dtype(self) -> Optional[torch.dtype]:
        s = self.hparams.torch_dtype
        if s is None:
            return None
        if s == "auto":
            return None
        if s in ("bf16", "bfloat16"):
            return torch.bfloat16
        if s in ("fp16", "float16"):
            return torch.float16
        if s in ("fp32", "float32"):
            return torch.float32
        raise ValueError(f"Unknown torch_dtype={s}")

    def setup(self, stage: str) -> None:
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM

        dtype = self._resolve_dtype()
        model = AutoModelForCausalLM.from_pretrained(
            self.hparams.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=self.hparams.trust_remote_code,
        )

        if self.hparams.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = bool(self.hparams.use_cache)

        # LoRA via PEFT
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        target_modules = self.hparams.lora_target_modules
        if target_modules is None:
            # reasonable defaults for many decoder-only transformers (incl. Qwen/LLaMA-like)
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_cfg = LoraConfig(
            r=int(self.hparams.lora_r),
            lora_alpha=int(self.hparams.lora_alpha),
            lora_dropout=float(self.hparams.lora_dropout),
            bias=str(self.hparams.lora_bias),
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(target_modules),
        )

        model = get_peft_model(model, lora_cfg)

        if self.hparams.lora_adapter_path:
            # Load weights into current PEFT model
            model = PeftModel.from_pretrained(model, self.hparams.lora_adapter_path, is_trainable=True)

        # log trainable params (rank0 only happens via lightning log settings)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

        self.model = model

        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def forward(self, **batch: torch.Tensor) -> Any:
        return self.model(**batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.model(**batch)
        loss = out.loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        try:
            ppl = torch.exp(loss.detach().clamp(max=20))
            self.log("train/ppl", ppl, on_step=True, on_epoch=True, prog_bar=False)
        except Exception:
            pass
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model(**batch)
        loss = out.loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        try:
            ppl = torch.exp(loss.detach().clamp(max=20))
            self.log("val/ppl", ppl, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        except Exception:
            pass

    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer(params=trainable_params)

        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        # Support HF schedulers like get_linear_schedule_with_warmup via _partial_
        num_training_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        if num_training_steps <= 0:
            # fallback for some trainer configs
            num_training_steps = 1

        scheduler = self.hparams.scheduler(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


