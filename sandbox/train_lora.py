import os
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

BASE_ID = "meta-llama/Llama-3.2-1B-Instruct"
TRAIN_FP = os.path.expanduser("~/Documents/Advanced_AI/AdvancedAI_IndividualProject/train.jsonl")
VAL_FP   = os.path.expanduser("~/Documents/Advanced_AI/AdvancedAI_IndividualProject/val.jsonl")
OUT_DIR  = os.path.expanduser("~/Documents/Advanced_AI/AdvancedAI_IndividualProject/recipes-adapter-1B")
os.makedirs(OUT_DIR, exist_ok=True)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SEQ_LEN        = int(os.environ.get("SEQ_LEN", "192"))
TOK_NUM_PROC   = int(os.environ.get("TOK_NUM_PROC", "2"))
DL_WORKERS     = int(os.environ.get("DL_WORKERS", "1"))
BATCH          = int(os.environ.get("BATCH", "6"))
GRAD_ACC       = int(os.environ.get("GRAD_ACC", "1"))
MAX_SAMPLES    = int(os.environ.get("MAX_SAMPLES", "0"))
TRAIN_FRACTION = float(os.environ.get("TRAIN_FRACTION", "0.05"))

def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
    tok.padding_side = "right"
    tok.truncation_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def tokenize_file(tokenizer, data_fp, apply_subsample=False, fraction=None):
    ds = load_dataset("json", data_files=data_fp, split="train")
    if fraction is not None and 0 < fraction < 1.0:
        n = max(1, int(len(ds) * fraction))
        ds = ds.select(range(n))
    def _to_text(ex):
        t = ex.get("text", "")
        return {"text": t if isinstance(t, str) else str(t)}
    ds = ds.map(_to_text, remove_columns=[c for c in ds.column_names if c != "text"])
    def tok_batch(batch):
        texts = [x + tokenizer.eos_token for x in batch["text"]]
        enc = tokenizer(
            texts,
            max_length=SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        return enc
    tok_ds = ds.map(
        tok_batch,
        batched=True,
        num_proc=max(1, min(TOK_NUM_PROC, (os.cpu_count() or 4))),
        remove_columns=["text"],
    )
    if apply_subsample and MAX_SAMPLES and 0 < MAX_SAMPLES < len(tok_ds):
        tok_ds = tok_ds.shuffle(seed=42).select(range(MAX_SAMPLES))
    tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tok_ds

def build_model():
    use_mps = torch.backends.mps.is_available()
    torch_dtype = torch.float16 if use_mps else torch.float32
    return AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )

def main():
    tokenizer = build_tokenizer()
    train_ds = tokenize_file(tokenizer, TRAIN_FP, apply_subsample=True, fraction=TRAIN_FRACTION)
    val_ds   = tokenize_file(tokenizer, VAL_FP, apply_subsample=False, fraction=None)

    model = build_model()

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    sft_cfg = SFTConfig(
        output_dir=OUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        dataloader_num_workers=DL_WORKERS,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        save_steps=0,
        save_total_limit=1,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
    )

    trainer.train()
    metrics = trainer.evaluate()
    try:
        print("Eval loss:", metrics["eval_loss"], " | Eval ppl:", math.exp(metrics["eval_loss"]))
    except Exception:
        pass
    trainer.model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("\n✅ Finished. Trained on first {:.1f}% of train.jsonl and evaluated on val.jsonl. LoRA adapter saved to: {}".format(TRAIN_FRACTION*100, OUT_DIR))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()