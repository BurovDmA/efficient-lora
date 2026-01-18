from __future__ import annotations

from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class CausalLMCollator:
    """Pad to max length in batch and create causal LM labels (pad -> -100)."""

    def __init__(self, tokenizer, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        has_labels = "labels" in features[0]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        if has_labels:
            # If labels were precomputed (e.g., prompt-masked), pad them to match input length.
            padded_labels: List[List[int]] = []
            max_len = int(input_ids.shape[1])
            for f in features:
                lab = list(f["labels"])
                if len(lab) < max_len:
                    lab = lab + [self.label_pad_token_id] * (max_len - len(lab))
                else:
                    lab = lab[:max_len]
                padded_labels.append(lab)
            labels = torch.tensor(padded_labels, dtype=input_ids.dtype)
        else:
            labels = input_ids.clone()

        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask.eq(0), self.label_pad_token_id)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ParquetCausalLMDataModule(LightningDataModule):
    """LightningDataModule for local parquet datasets -> tokenized causal LM batches.

    Loads one or many parquet files via `datasets`, builds a unified `text` field from
    common column patterns (question/response, input/output, question/answer), tokenizes,
    and returns batches suitable for `AutoModelForCausalLM`.
    """

    def __init__(
        self,
        data_files: Union[str, Sequence[str]],
        tokenizer_name_or_path: str,
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        val_size: float = 0.01,
        max_length: int = 1024,
        add_eos: bool = True,
        trust_remote_code: bool = True,
        text_column: Optional[str] = None,
        prompt_template: Optional[str] = None,
        use_chat_template: bool = False,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.batch_size_per_device = batch_size
        self._tokenizer = None
        self._collator = None

    def _resolve_files(self) -> List[str]:
        data_files = self.hparams.data_files
        if isinstance(data_files, str):
            # allow glob patterns
            expanded = glob(data_files)
            return expanded if expanded else [data_files]
        out: List[str] = []
        for item in list(data_files):
            if isinstance(item, str):
                expanded = glob(item)
                out.extend(expanded if expanded else [item])
        return out

    def prepare_data(self) -> None:
        # no downloading; just verify deps import
        import datasets as _  # noqa: F401
        import transformers as _t  # noqa: F401

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train is not None and self.data_val is not None:
            return

        from datasets import load_dataset
        from transformers import AutoTokenizer

        files = self._resolve_files()
        if not files:
            raise FileNotFoundError(
                f"No parquet files resolved from data_files={self.hparams.data_files}"
            )

        ds = load_dataset("parquet", data_files=files, split="train")

        tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name_or_path,
            use_fast=True,
            trust_remote_code=self.hparams.trust_remote_code,
        )
        # Many causal LM tokenizers have no pad token. For batching, map it to eos.
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer
        self._collator = CausalLMCollator(tokenizer=tokenizer)

        text_column = self.hparams.text_column
        prompt_template = self.hparams.prompt_template

        def _row_to_text(ex: Dict[str, Any]) -> str:
            # explicit column wins
            if text_column and ex.get(text_column) is not None:
                return str(ex[text_column])

            q = ex.get("question")
            r = ex.get("response")
            a = ex.get("answer")
            inp = ex.get("input")
            out = ex.get("output")

            if self.hparams.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                messages: List[Dict[str, str]] = []
                if self.hparams.system_prompt:
                    messages.append({"role": "system", "content": str(self.hparams.system_prompt)})

                if inp is not None and out is not None:
                    messages.append({"role": "user", "content": str(inp)})
                    messages.append({"role": "assistant", "content": str(out)})
                elif q is not None and r is not None:
                    messages.append({"role": "user", "content": str(q)})
                    messages.append({"role": "assistant", "content": str(r)})
                elif q is not None and a is not None:
                    messages.append({"role": "user", "content": str(q)})
                    messages.append({"role": "assistant", "content": str(a)})

                if messages:
                    try:
                        return tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                    except Exception:
                        pass

            if prompt_template:
                # simple python format with available keys
                try:
                    return prompt_template.format(**ex)
                except Exception:
                    # fall through to heuristics
                    pass

            if q is not None and r is not None:
                return f"Question:\n{q}\n\nAnswer:\n{r}"
            if q is not None and a is not None:
                return f"Question:\n{q}\n\nAnswer:\n{a}"
            if inp is not None and out is not None:
                return f"Instruction:\n{inp}\n\nResponse:\n{out}"

            # fallback: concatenate string-ish fields
            parts: List[str] = []
            for k, v in ex.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    parts.append(f"{k}: {v}")
            return "\n".join(parts)

        def _make_text(batch: Dict[str, List[Any]]) -> Dict[str, List[str]]:
            texts: List[str] = []
            keys = list(batch.keys())
            n = len(batch[keys[0]]) if keys else 0
            for i in range(n):
                ex = {k: batch[k][i] for k in keys}
                text = _row_to_text(ex)
                if self.hparams.add_eos and tokenizer.eos_token is not None and not self.hparams.use_chat_template:
                    text = text + tokenizer.eos_token
                texts.append(text)
            return {"text": texts}

        ds = ds.map(_make_text, batched=True, desc="Building text")

        def _tokenize_plain(batch: Dict[str, List[str]]) -> Dict[str, Any]:
            tok = tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.hparams.max_length,
                padding=False,
            )
            # precompute labels so collator won't overwrite (keeps future flexibility)
            tok["labels"] = [ids[:] for ids in tok["input_ids"]]
            return tok

        # If we want assistant-only loss, we need row-wise access to original columns to build prompt/full.
        if self.hparams.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            def _tokenize_chat(ex: Dict[str, Any]) -> Dict[str, Any]:
                q = ex.get("question")
                r = ex.get("response")
                a = ex.get("answer")
                inp = ex.get("input")
                out = ex.get("output")

                messages: List[Dict[str, str]] = []
                if self.hparams.system_prompt:
                    messages.append({"role": "system", "content": str(self.hparams.system_prompt)})

                if inp is not None and out is not None:
                    messages.append({"role": "user", "content": str(inp)})
                    messages.append({"role": "assistant", "content": str(out)})
                elif q is not None and r is not None:
                    messages.append({"role": "user", "content": str(q)})
                    messages.append({"role": "assistant", "content": str(r)})
                elif q is not None and a is not None:
                    messages.append({"role": "user", "content": str(q)})
                    messages.append({"role": "assistant", "content": str(a)})
                else:
                    # fallback to rendered text, loss on all tokens
                    tok = tokenizer(
                        ex["text"],
                        truncation=True,
                        max_length=self.hparams.max_length,
                        padding=False,
                    )
                    tok["labels"] = tok["input_ids"][:]
                    return tok

                # prompt is everything up to the assistant answer
                prompt_text = tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                prompt_tok = tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=self.hparams.max_length,
                    padding=False,
                )
                full_tok = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.hparams.max_length,
                    padding=False,
                )

                input_ids = full_tok["input_ids"]
                attention_mask = full_tok.get("attention_mask", [1] * len(input_ids))

                prompt_len = min(len(prompt_tok["input_ids"]), len(input_ids))
                labels = input_ids[:]
                labels[:prompt_len] = [-100] * prompt_len
                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

            ds = ds.map(
                _tokenize_chat,
                desc="Tokenizing (chat SFT, assistant-only loss)",
                remove_columns=list(ds.column_names),
            )
        else:
            ds = ds.map(
                _tokenize_plain,
                batched=True,
                # drop raw columns (including `text`) -> keep only tokenized tensors
                remove_columns=list(ds.column_names),
                desc="Tokenizing",
            )

        # train/val split
        if not (0.0 < float(self.hparams.val_size) < 1.0):
            raise ValueError("val_size must be a float in (0, 1)")
        split = ds.train_test_split(test_size=float(self.hparams.val_size), seed=int(self.hparams.seed))
        self.data_train = split["train"].with_format("torch")
        self.data_val = split["test"].with_format("torch")

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self._collator,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._collator,
        )


