"""
Fine-tune with GRPO so every reply
reads like **The Irish Penny Journal** (1840).

Key settings
────────────
Hardware      : 1 × 48 GB GPU
Optimiser     : AdamW-8bit
Precision     : bf16 (if supported) else fp16   + TF32 matmuls
Group size    : 8 generations / prompt  →  batch size = 8
Dataset       : WizardLM-70 K (first 70 000 rows)
Lengths       : prompt ≤ 512 tok, completion ≤ 128 tok
vLLM cache    : 60 % GPU RAM ⇒ ≈ 9 k tokens  (no frequent resets)
Logging       : TensorBoard
"""

from __future__ import annotations
import os, random, torch, logging
from typing import List
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from trl import GRPOConfig, GRPOTrainer

torch.manual_seed(0)
random.seed(0)
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

# Reward
REWARD_MODEL_ID = "./ipj_classifier"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class IPJReward:
    """Return P(label==1) — higher ⇒ more IPJ-like."""

    def __init__(self, model_id: str = REWARD_MODEL_ID):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=DTYPE, device_map={"": DEVICE}
        ).eval()
        self.__name__ = "IPJReward"

    @torch.no_grad()
    def __call__(self, prompts, completions, **__):
        enc = self.tok(
            completions,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)
        return torch.softmax(self.model(**enc).logits, -1)[:, 1].cpu().tolist()


reward_fn = IPJReward()

# Model
BASE_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

tok = AutoTokenizer.from_pretrained(BASE_ID, padding_side="left")
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_ID, torch_dtype=DTYPE, device_map="auto"
)

# Prompt
SYSTEM_IPJ = (
    "You are a learned assistant whose every utterance is penned in the "
    "elegant prose of *The Irish Penny Journal* (circa 1840). "
    "Be descriptive, poetic, and rich in historical colour."
)


def build_prompt(user_q: str) -> str:
    return (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_IPJ}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_q}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


# Dataset
# This could be almost anything...
# Just need some prompts to fill
wizard_full = load_dataset(
    "WizardLM/WizardLM_evol_instruct_V2_196k",
    split="train",
)

# 10k I stopped at 6800 steps
# This may be more or less depending on training parameters
wizard = wizard_full.select(range(10_000))


def fmt(ex):
    for turn in ex["conversations"]:
        if turn["from"] in ("human", "user"):
            prompt = turn["value"]
            break
    else:
        prompt = ex["conversations"][0]["value"]
    return {"prompt": build_prompt(prompt)}


train_ds = wizard.shuffle(seed=42).map(fmt, remove_columns=wizard.column_names)
eval_ds = train_ds.select(range(1000))

dataset = {"train": train_ds, "test": eval_ds}

# GRPO
NUM_GENS = 12
BATCH_SIZE = 12

train_cfg = GRPOConfig(
    output_dir="llama3_ipj_grpo",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=5e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    logging_dir="runs/ipj_tensorboard",
    report_to="tensorboard",
    logging_steps=10,
    save_strategy="steps",
    save_steps=400,
    save_total_limit=2,
    # If you want to do a holdout eval
    # do_eval=False,
    # eval_strategy="steps",
    # eval_steps=200,
    # —— sequence lengths ——
    max_prompt_length=512,
    max_completion_length=512,
    # —— GRPO specifics ——
    num_generations=NUM_GENS,
    disable_dropout=True,
    sync_ref_model=True,
    ref_model_sync_steps=32,
    ref_model_mixup_alpha=0.8,
    # precision / optimiser
    bf16=(DTYPE is torch.bfloat16),
    fp16=(DTYPE is torch.float16),
    tf32=True,
    optim="adamw_8bit",
    # gradient_checkpointing = True,
    # —— vLLM settings ——
    use_vllm=False,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.50,
    #
    cache_implementation="static",
    mask_truncated_completions=True,
    loss_type="dr_grpo",
    use_liger_kernel=True,
    use_liger_loss=True,
)

# vLLM environment vars if needed
# os.environ.setdefault("RANK", "0")
# os.environ.setdefault("WORLD_SIZE", "1")
# os.environ.setdefault("LOCAL_RANK", "0")
# os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
# os.environ.setdefault("MASTER_PORT", "29500")

# # silence noisy “reset prefix cache” logs
# logging.getLogger("block_pool").setLevel(logging.ERROR)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=train_cfg,
    train_dataset=dataset["train"],
    # eval_dataset     = dataset["test"],
    processing_class=tok,
)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    trainer.train()
    trainer.save_model(train_cfg.output_dir)
    tok.save_pretrained(train_cfg.output_dir)
