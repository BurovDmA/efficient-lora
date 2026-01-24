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
        scheduler: Any | None = None,
        compile: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: str | None = None,  # "auto" | "bf16" | "fp16" | "fp32"
        gradient_checkpointing: bool = True,
        use_cache: bool = False,
        # LoRA
        adapter_type: str = "lora",  # "lora" | "l1ra"
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        lora_bias: str = "none",
        # Optional: load existing adapter for continued training
        lora_adapter_path: str | None = None,
        # L1RA-specific
        l1ra_lambda: float = 1e-3,
        l1ra_eta_c: float = 1e-2,
        l1ra_rank_update_ratio: float = 0.1,
        l1ra_prune_threshold: float = 1e-6,
        l1ra_reassign: bool = True,
        l1ra_exclude_pruned: bool = True,
        l1ra_warmup_steps: int = 0,
        # Adapter saving (module-level)
        save_adapter_dir: str | None = None,
        save_adapter_every_n_epochs: int = 1,
        save_adapter_on_train_end: bool = True,
        # Resource logging (module-level)
        log_resource_every_n_steps: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = None
        self._l1ra_real_step = 0

    def _resolve_dtype(self) -> torch.dtype | None:
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
            # Disable KV-cache for training (recommended with gradient checkpointing)
            if bool(self.hparams.gradient_checkpointing):
                model.config.use_cache = False
            else:
                model.config.use_cache = bool(self.hparams.use_cache)

        # LoRA/L1RA via PEFT
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        from src.l1ra.tuner.config import L1RAConfig

        target_modules = self.hparams.lora_target_modules
        if target_modules is None:
            # reasonable defaults for many decoder-only transformers (incl. Qwen/LLaMA-like)
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        if str(self.hparams.adapter_type).lower() == "l1ra":
            lora_cfg = L1RAConfig(
                r=int(self.hparams.lora_r),
                lora_alpha=int(self.hparams.lora_alpha),
                lora_dropout=float(self.hparams.lora_dropout),
                bias=str(self.hparams.lora_bias),
                task_type=TaskType.CAUSAL_LM,
                target_modules=list(target_modules),
                l1ra_lambda=float(self.hparams.l1ra_lambda),
                eta_c=float(self.hparams.l1ra_eta_c),
                rank_update_ratio=float(self.hparams.l1ra_rank_update_ratio),
                prune_threshold=float(self.hparams.l1ra_prune_threshold),
                reassign=bool(self.hparams.l1ra_reassign),
                exclude_pruned=bool(self.hparams.l1ra_exclude_pruned),
            )
        else:
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
            model = PeftModel.from_pretrained(
                model, self.hparams.lora_adapter_path, is_trainable=True
            )

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

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if str(self.hparams.adapter_type).lower() == "l1ra":
            self._maybe_update_l1ra_ranks()

        out = self.model(**batch)
        loss = out.loss
        if str(self.hparams.adapter_type).lower() == "l1ra":
            loss = loss + self._l1ra_l1_regularization()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        try:
            ppl = torch.exp(loss.detach().clamp(max=20))
            self.log("train/ppl", ppl, on_step=True, on_epoch=True, prog_bar=False)
        except Exception:
            pass

        self._log_resources_if_needed()
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model(**batch)
        loss = out.loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        try:
            ppl = torch.exp(loss.detach().clamp(max=20))
            self.log("val/ppl", ppl, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        except Exception:
            pass

    def on_train_start(self) -> None:
        # Ensure full model is in train mode (PEFT may keep some modules in eval)
        if self.model is not None:
            self.model.train()
        if str(self.hparams.adapter_type).lower() == "l1ra":
            try:
                lr = self._get_base_lr()
                if hasattr(self.model, "set_threshold"):
                    self.model.set_threshold(lr)
            except Exception:
                pass

    def on_train_epoch_start(self) -> None:
        # Keep train mode across epochs (Lightning can toggle for val)
        if self.model is not None:
            self.model.train()

    def on_validation_epoch_end(self) -> None:
        if not self.hparams.save_adapter_dir:
            return
        every = int(self.hparams.save_adapter_every_n_epochs)
        if every <= 0:
            return
        epoch = int(getattr(self.trainer, "current_epoch", 0))
        if (epoch + 1) % every != 0:
            return
        self._save_adapter(tag=f"epoch_{epoch:03d}")

    def on_train_end(self) -> None:
        if self.hparams.save_adapter_dir and self.hparams.save_adapter_on_train_end:
            self._save_adapter(tag="final")

    def _save_adapter(self, tag: str) -> None:
        if self.model is None:
            return
        try:
            from pathlib import Path

            out_dir = Path(self.hparams.save_adapter_dir) / tag
            out_dir.mkdir(parents=True, exist_ok=True)

            save_fn = getattr(self.model, "save_pretrained", None)
            if callable(save_fn):
                save_fn(str(out_dir))

            tok = getattr(getattr(self.trainer, "datamodule", None), "_tokenizer", None)
            if tok is not None and callable(getattr(tok, "save_pretrained", None)):
                tok.save_pretrained(str(out_dir))
        except Exception:
            pass

    def _log_resources_if_needed(self) -> None:
        every = int(self.hparams.log_resource_every_n_steps)
        if every <= 0:
            return
        step = int(getattr(self.trainer, "global_step", 0))
        if step == 0 or (step % every != 0):
            return

        # Process + system RAM
        try:
            import psutil

            proc = psutil.Process()
            mem = proc.memory_info()
            self.log(
                "sys/ram_rss_mb", mem.rss / (1024**2), on_step=True, on_epoch=False, logger=True
            )
            self.log(
                "sys/ram_vms_mb", mem.vms / (1024**2), on_step=True, on_epoch=False, logger=True
            )
            vm = psutil.virtual_memory()
            self.log(
                "sys/ram_used_mb", vm.used / (1024**2), on_step=True, on_epoch=False, logger=True
            )
            self.log(
                "sys/ram_percent", float(vm.percent), on_step=True, on_epoch=False, logger=True
            )
        except Exception:
            pass

        # CUDA memory
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                alloc = torch.cuda.memory_allocated(device) / (1024**2)
                reserved = torch.cuda.memory_reserved(device) / (1024**2)
                max_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
                max_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
                self.log("sys/cuda_mem_alloc_mb", alloc, on_step=True, on_epoch=False, logger=True)
                self.log(
                    "sys/cuda_mem_reserved_mb", reserved, on_step=True, on_epoch=False, logger=True
                )
                self.log(
                    "sys/cuda_max_mem_alloc_mb",
                    max_alloc,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                )
                self.log(
                    "sys/cuda_max_mem_reserved_mb",
                    max_reserved,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                )
        except Exception:
            pass

        # MPS memory (best-effort)
        try:
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                fn = getattr(torch.mps, "current_allocated_memory", None)
                if callable(fn):
                    alloc_bytes = fn()
                    self.log(
                        "sys/mps_allocated_mb",
                        alloc_bytes / (1024**2),
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                    )
        except Exception:
            pass

    def configure_optimizers(self) -> dict[str, Any]:
        if str(self.hparams.adapter_type).lower() == "l1ra":
            optimizer = self._create_l1ra_optimizer()
        else:
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

    def _create_l1ra_optimizer(self) -> torch.optim.Optimizer:
        c_vectors = []
        ab_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_c" in name:
                c_vectors.append(param)
            else:
                ab_params.append(param)

        param_groups = []
        if c_vectors:
            param_groups.append(
                {
                    "params": c_vectors,
                    "weight_decay": 0.0,
                    "lr": float(self.hparams.l1ra_eta_c),
                }
            )
        if ab_params:
            param_groups.append(
                {
                    "params": ab_params,
                    "weight_decay": float(
                        getattr(self.hparams.optimizer, "keywords", {}).get("weight_decay", 0.0)
                    ),
                }
            )

        return self.hparams.optimizer(params=param_groups)

    def _l1ra_l1_regularization(self) -> torch.Tensor:
        coef = float(self.hparams.l1ra_lambda)
        if coef <= 0:
            return torch.tensor(0.0, device=self.device)
        l1_sum = 0.0
        count = 0
        for name, param in self.model.named_parameters():
            if "lora_c" in name and param.requires_grad:
                l1_sum = l1_sum + param.abs().sum()
                count += param.numel()
        if count == 0:
            return torch.tensor(0.0, device=self.device)
        return coef * (l1_sum / count)

    def _maybe_update_l1ra_ranks(self) -> None:
        if not hasattr(self.model, "update_ranks"):
            return
        if not hasattr(self.model, "threshold"):
            try:
                lr = self._get_base_lr()
                if hasattr(self.model, "set_threshold"):
                    self.model.set_threshold(lr)
            except Exception:
                pass
        num_training_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        if num_training_steps <= 0:
            return
        num_warmup_steps = int(self.hparams.l1ra_warmup_steps)
        updated = self.model.update_ranks(
            self._l1ra_real_step, num_training_steps, num_warmup_steps
        )
        if updated:
            self._restart_optimizer()
        self._l1ra_real_step += 1

    def _restart_optimizer(self) -> None:
        try:
            new_optimizer = self._create_l1ra_optimizer()
            # Update trainer optimizers
            if hasattr(self.trainer, "optimizers"):
                self.trainer.optimizers = [new_optimizer]
            if hasattr(self.trainer, "strategy") and hasattr(self.trainer.strategy, "optimizers"):
                self.trainer.strategy.optimizers = [new_optimizer]
            # Update schedulers to point to new optimizer
            if hasattr(self.trainer, "lr_scheduler_configs"):
                for cfg in self.trainer.lr_scheduler_configs:
                    try:
                        cfg.scheduler.optimizer = new_optimizer
                    except Exception:
                        pass
        except Exception:
            pass
