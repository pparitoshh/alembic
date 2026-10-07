"""Stage 3: (Q)LoRA / (Q)DoRA SFT of the student on verified teacher transcripts.

Loss is on assistant turns only (TRL swaps in a training chat template with generation markers),
so tool results and user turns are context, not targets.
Multi-GPU: launch with `accelerate launch --config_file configs/accelerate/fsdp.yaml -m distillkit.cli train -c ...`.
"""

from pathlib import Path

from .config import Config


def run(cfg: Config, *, before_train=None, callbacks=()) -> Path:
    """Train through the production path, optionally inspecting it before optimization.

    The hooks support bounded runtime checks without maintaining a second trainer.
    They do not change configuration, labels, losses or model selection.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from .io import read_jsonl
    from .records import training_example

    scfg, tcfg, run_dir = cfg.student, cfg.train, cfg.run_dir
    out_dir = run_dir / "adapter"

    tok = AutoTokenizer.from_pretrained(scfg.model)
    rows = read_jsonl(run_dir / "verified.jsonl")
    ds = Dataset.from_list([training_example(r, scfg.system_prompt) for r in rows])
    print(f"[train] {len(ds)} examples, student={scfg.model}, method={tcfg.method}, 4bit={tcfg.load_in_4bit}")

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
        output_dir=str(run_dir / "checkpoints"),
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
        save_strategy="no",
        seed=tcfg.seed,
        report_to="none",
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
        processing_class=tok,
        peft_config=peft_cfg,
    )
    if tcfg.load_in_4bit and tcfg.fp16:
        # TRL casts QLoRA adapters to bf16, which fp16 AMP's grad scaler can't unscale: keep them fp32
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    for callback in callbacks:
        trainer.add_callback(callback)
    if before_train is not None:
        before_train(trainer)
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[train] adapter saved -> {out_dir}")
    return out_dir
