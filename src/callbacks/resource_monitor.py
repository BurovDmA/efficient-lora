from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from lightning import Callback, LightningModule, Trainer


@dataclass
class ResourceMonitorCallback(Callback):
    """Log process/system memory usage to the active logger.

    Logs:
    - sys/ram_rss_mb: process RSS in MB
    - sys/ram_vms_mb: process VMS in MB
    - sys/ram_used_mb + sys/ram_percent: system memory usage
    - sys/cuda_*: CUDA allocator stats (if available)
    - sys/mps_allocated_mb: MPS allocated memory (if available)
    """

    log_every_n_steps: int = 50

    def _log(self, pl_module: LightningModule, name: str, value: Any) -> None:
        try:
            pl_module.log(name, value, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        except Exception:
            pass

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.log_every_n_steps <= 0:
            return

        step = int(getattr(trainer, "global_step", 0))
        if step == 0 or (step % int(self.log_every_n_steps) != 0):
            return

        # Process + system RAM
        try:
            import psutil

            proc = psutil.Process()
            mem = proc.memory_info()
            rss_mb = mem.rss / (1024**2)
            vms_mb = mem.vms / (1024**2)
            self._log(pl_module, "sys/ram_rss_mb", rss_mb)
            self._log(pl_module, "sys/ram_vms_mb", vms_mb)

            vm = psutil.virtual_memory()
            self._log(pl_module, "sys/ram_used_mb", vm.used / (1024**2))
            self._log(pl_module, "sys/ram_percent", float(vm.percent))
        except Exception:
            pass

        # CUDA memory
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                alloc = torch.cuda.memory_allocated(device) / (1024**2)
                reserved = torch.cuda.memory_reserved(device) / (1024**2)
                max_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
                max_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
                self._log(pl_module, "sys/cuda_mem_alloc_mb", alloc)
                self._log(pl_module, "sys/cuda_mem_reserved_mb", reserved)
                self._log(pl_module, "sys/cuda_max_mem_alloc_mb", max_alloc)
                self._log(pl_module, "sys/cuda_max_mem_reserved_mb", max_reserved)
        except Exception:
            pass

        # MPS memory (best-effort)
        try:
            import torch

            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                fn = getattr(torch.mps, "current_allocated_memory", None)
                if callable(fn):
                    alloc_bytes = fn()
                    self._log(pl_module, "sys/mps_allocated_mb", alloc_bytes / (1024**2))
        except Exception:
            pass


