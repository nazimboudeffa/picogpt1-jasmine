"""Entraîne un adaptateur LoRA sur un très petit dataset français.

Usage :
    python train_small.py

Différence avec train.py (la baseline v0.1.0-jasmine) :
  - v1  : GPT-2 de 6,86 M params entraîné de zéro sur 500 exemples Alpaca.
  - ici : SmolLM2-135M pré-entraîné + LoRA sur 40 exemples écrits à la main.

La connaissance vient donc du pré-entraînement ; les 40 exemples ne servent
qu'à apprendre le format « question -> réponse courte en français ».
"""

import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
DATA_PATH = Path("dataset_small.jsonl")
OUTPUT_DIR = Path("PicoGPT1-Jasmine-Tea")

MAX_LEN = 256
MAX_EXAMPLES = 40
EPOCHS = 60
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
EVAL_FRACTION = 0.1
SEED = 42


def load_examples():
    """Lit dataset_small.jsonl : une paire prompt/réponse par ligne JSON."""
    examples = []
    with DATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append((row["prompt"].strip(), row["response"].strip()))

    if len(examples) != MAX_EXAMPLES:
        print(f"Attention : {len(examples)} exemples lus, {MAX_EXAMPLES} attendus.")

    return examples[:MAX_EXAMPLES]


def build_example(tokenizer, prompt, completion):
    """Tokenise une paire et masque le prompt dans les labels."""
    text = f"{prompt}\n{completion}{tokenizer.eos_token}"
    tokenized = tokenizer(text, truncation=True, max_length=MAX_LEN)
    input_ids = tokenized["input_ids"]

    prefix_ids = tokenizer(f"{prompt}\n", truncation=True, max_length=MAX_LEN)["input_ids"]
    labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids):]

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


def main():
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_LEN

    examples = load_examples()
    dataset = Dataset.from_list(
        [build_example(tokenizer, prompt, response) for prompt, response in examples]
    )
    split = dataset.train_test_split(test_size=EVAL_FRACTION, seed=SEED)

    print(f"Modèle de base   : {BASE_MODEL}")
    print(f"Exemples         : {len(dataset)} ({len(split['train'])} train / {len(split['test'])} eval)")
    print(f"Pad / EOS token  : {tokenizer.pad_token!r} / {tokenizer.eos_token!r}")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        seed=SEED,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=collator,
    )
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"PicoGPT1-Jasmine-Tea sauvegardé dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
