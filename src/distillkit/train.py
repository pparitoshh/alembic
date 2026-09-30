"""Stage 3: LoRA SFT of the student on verified teacher answers (loss on the completion only)."""

from pathlib import Path


def run(cfg: dict) -> Path:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from .io import read_jsonl
    from .prompting import render_prompt

    scfg, tcfg = cfg["student"], cfg["train"]
    run_dir = Path(cfg["run_dir"])
    out_dir = run_dir / "adapter"

    tok = AutoTokenizer.from_pretrained(scfg["model"])
    rows = read_jsonl(run_dir / "verified.jsonl")
    ds = Dataset.from_list(
        [
            {
                "prompt": render_prompt(tok, scfg["system_prompt"], r["question"]),
                "completion": r["answer"] + tok.eos_token,
            }
            for r in rows
        ]
    )
    print(f"[train] {len(ds)} examples, student={scfg['model']}")

    args = SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # float in [0, 1) = ratio of total steps (transformers 5 dropped warmup_ratio)
        max_length=tcfg["max_length"],
        completion_only_loss=True,
        fp16=tcfg["fp16"],
        bf16=tcfg["bf16"],
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        # fp16 AMP needs fp32 master weights (GPUs without bf16, e.g. RTX 20xx)
        model_init_kwargs={"dtype": "float32" if tcfg["fp16"] else "bfloat16"},
    )
    lora = LoraConfig(
        r=tcfg["lora_r"],
        lora_alpha=tcfg["lora_alpha"],
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=scfg["model"],
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[train] adapter saved -> {out_dir}")
    return out_dir
