"""Stage 3: (Q)LoRA / (Q)DoRA SFT of the student on verified teacher transcripts.

Loss is on assistant turns only (TRL swaps in a training chat template with generation markers),
so tool results and user turns are context, not targets.
Multi-GPU: launch with `accelerate launch --config_file configs/accelerate/fsdp.yaml -m distillkit.cli train -c ...`.
"""

import hashlib
import json
import random
import shutil
from pathlib import Path

from .config import Config


def admitted_training_rows(cfg: Config, *, evidence_dir: Path | None = None) -> list[dict]:
    if cfg.train.input_manifest is not None:
        from .training_inputs import admitted_rows
        return admitted_rows(cfg, evidence_dir=evidence_dir)
    from .io import read_jsonl
    from .support import source_for_records
    rows = read_jsonl(cfg.run_dir / 'verified.jsonl')
    if getattr(cfg.seeds, 'registry', None) is not None:
        source_for_records(cfg, rows)
    return rows


def token_lengths(tok, examples: list[dict]) -> list[int]:
    """Tokens per example as the trainer renders it (system prompt, tools, chat template)."""
    lengths = []
    for ex in examples:
        tools = json.loads(ex["tools"]) if ex.get("tools") else None
        text = tok.apply_chat_template(ex["messages"], tools=tools, tokenize=False,
                                       **ex.get("chat_template_kwargs", {}))
        lengths.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    return lengths


def length_report(lengths: list[int], max_length: int) -> str:
    over = sum(n > max_length for n in lengths)
    s = sorted(lengths)
    return (f"[train] tokens per example: median {s[len(s) // 2]}, max {s[-1]}; "
            f"{over} of {len(s)} over max_length={max_length} (truncated, losing the answer's end)")


def inputs_fingerprint(cfg: Config, examples: list[dict]) -> str:
    """What a checkpoint was trained on; resuming with other data or settings would mix two runs."""
    train = cfg.train.model_dump(mode="json", exclude={"save_steps", "save_total_limit", "report_to"})
    blob = json.dumps({"examples": examples, "student": cfg.student.model, "train": train},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def resume_checkpoint(ckpt_dir: Path, fingerprint: str) -> str | None:
    """Newest checkpoint-<step> in ckpt_dir to resume from, after checking it belongs to these inputs."""
    stamp = ckpt_dir / "inputs.sha256"
    found = sorted((d for d in ckpt_dir.glob("checkpoint-*") if d.name.split("-")[-1].isdigit()),
                   key=lambda d: int(d.name.split("-")[-1]))
    if not found:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        stamp.write_text(fingerprint + "\n")
        return None
    if not stamp.exists() or stamp.read_text().strip() != fingerprint:
        raise ValueError(f"{ckpt_dir} holds checkpoints from other training data or settings; "
                         "use a new run_dir or delete the checkpoints")
    return str(found[-1])


def split_by_document(rows: list[dict], examples: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """(train, validation) examples; whole source documents go to validation until it holds >= fraction."""
    if not fraction:
        return examples, []
    doc = [r.get("doc_id") or r["id"] for r in rows]
    order = sorted(set(doc))
    random.Random(seed).shuffle(order)
    held, n = set(), 0
    for d in order:
        if n >= fraction * len(rows):
            break
        held.add(d)
        n += doc.count(d)
    train = [e for d, e in zip(doc, examples) if d not in held]
    val = [e for d, e in zip(doc, examples) if d in held]
    if not train:
        raise ValueError("val_fraction leaves no training examples")
    return train, val


ADAPTER_FILES_SKIPPED = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"}


def keep_adapter(checkpoint: Path, run_dir: Path, step: int) -> Path:
    """Copy a checkpoint's adapter to run_dir/adapters/step_<N>/adapter, the layout `answer`/`evaluate`
    read, so it survives save_total_limit pruning and can be evaluated later."""
    dst = run_dir / "adapters" / f"step_{step}" / "adapter"
    dst.mkdir(parents=True, exist_ok=True)
    for f in checkpoint.iterdir():
        if f.is_file() and f.name not in ADAPTER_FILES_SKIPPED:
            shutil.copy2(f, dst / f.name)
    return dst


def run(cfg: Config, *, before_train=None, callbacks=()) -> Path:
    """Train through the production path, optionally inspecting it before optimization.

    The hooks support bounded runtime checks without maintaining a second trainer.
    They do not change configuration, labels, losses or model selection.
    """
    # Admission is checked before tokenizer/model loading, including direct train
    # invocations that reuse an older verified file without rerunning verification.
    rows = admitted_training_rows(cfg, evidence_dir=cfg.run_dir)
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from .records import training_example

    scfg, tcfg, run_dir = cfg.student, cfg.train, cfg.run_dir
    out_dir = run_dir / "adapter"

    tok = AutoTokenizer.from_pretrained(scfg.model)
    examples = [training_example(r, scfg.system_prompt) for r in rows]
    train_ex, val_ex = split_by_document(rows, examples, tcfg.val_fraction, tcfg.seed)
    ds = Dataset.from_list(train_ex)
    val_ds = Dataset.from_list(val_ex) if val_ex else None
    print(f"[train] {len(ds)} examples (+{len(val_ex)} validation), student={scfg.model}, method={tcfg.method}, 4bit={tcfg.load_in_4bit}")
    print(length_report(token_lengths(tok, examples), tcfg.max_length))
    ckpt_dir = run_dir / "checkpoints"
    resume = resume_checkpoint(ckpt_dir, inputs_fingerprint(cfg, examples)) if tcfg.save_steps else None
    if resume:
        print(f"[train] resuming from {resume}")

    # fp16 AMP needs fp32 master weights (GPUs without bf16, e.g. RTX 20xx)
    load_dtype = torch.float32 if tcfg.fp16 else torch.bfloat16
    model_kwargs: dict = {"dtype": load_dtype}
    if tcfg.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, tcfg.dtype),
            # FSDP shards all params of one dtype together, so packed 4-bit weights must be stored in it
            bnb_4bit_quant_storage=load_dtype,
        )

    args = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=tcfg.epochs,
        per_device_train_batch_size=tcfg.batch_size,
        gradient_accumulation_steps=tcfg.grad_accum,
        learning_rate=tcfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # float in [0, 1) = ratio of total steps (transformers 5 dropped warmup_ratio)
        max_length=tcfg.max_length,
        assistant_only_loss=True,
        fp16=tcfg.fp16,
        bf16=tcfg.bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # required under FSDP, fine without
        logging_steps=1,
        save_strategy="steps" if tcfg.save_steps else "no",
        save_steps=tcfg.save_steps or 500,
        save_total_limit=tcfg.save_total_limit,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=tcfg.save_steps or 100,
        per_device_eval_batch_size=tcfg.batch_size,
        seed=tcfg.seed,
        report_to=tcfg.report_to or "none",
        run_name=str(run_dir),  # the `track` stage finds this run by name
        model_init_kwargs=model_kwargs,
    )
    peft_cfg = LoraConfig(
        r=tcfg.lora_r,
        lora_alpha=tcfg.lora_alpha,
        lora_dropout=tcfg.lora_dropout,
        use_dora=tcfg.method == "dora",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=scfg.model,
        args=args,
        train_dataset=ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    if tcfg.load_in_4bit and tcfg.fp16:
        # TRL casts QLoRA adapters to bf16, which fp16 AMP's grad scaler can't unscale: keep them fp32
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    if tcfg.save_steps:
        from transformers import TrainerCallback

        class KeepAdapters(TrainerCallback):
            def on_save(self, args, state, control, **kw):
                if state.is_world_process_zero:
                    keep_adapter(Path(args.output_dir) / f"checkpoint-{state.global_step}", run_dir, state.global_step)

        trainer.add_callback(KeepAdapters())
    for callback in callbacks:
        trainer.add_callback(callback)
    if before_train is not None:
        before_train(trainer)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[train] adapter saved -> {out_dir}")
    return out_dir
