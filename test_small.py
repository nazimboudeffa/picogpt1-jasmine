"""Teste l'adaptateur LoRA entraîné sur dataset_small.jsonl.

Usage :
    python test_small.py
    python test_small.py --prompt "Quelle est la capitale du Canada ?"
    python test_small.py --eval
    python test_small.py --heldout
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset

from small_utils import ADAPTER_DIR, generate, load_model
from train_small import EVAL_FRACTION, SEED

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = Path("dataset_small.jsonl")


def normalize(text):
    """Comparaison souple : casse, espaces et ponctuation finale ignorés."""
    return " ".join(text.strip().lower().replace("«", "").replace("»", "").split()).rstrip(".")


def evaluate(tokenizer, model):
    """Mesure le nombre de réponses exactes sur les 40 exemples du dataset."""
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    correct = 0

    for row in rows:
        answer = generate(tokenizer, model, row["prompt"])
        ok = normalize(answer) == normalize(row["response"])
        correct += int(ok)
        print(f"[{'OK ' if ok else 'KO '}] {row['prompt']} -> {answer}")

    print(f"\nRéponses exactes : {correct}/{len(rows)}")


def evaluate_heldout(tokenizer, model):
    """Mesure les réponses exactes sur les seuls exemples tenus dehors (graine 42)."""
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = Dataset.from_list(
        [{"prompt": row["prompt"], "response": row["response"]} for row in rows]
    )
    heldout = dataset.train_test_split(test_size=EVAL_FRACTION, seed=SEED)["test"]
    correct = 0

    for row in heldout:
        answer = generate(tokenizer, model, row["prompt"])
        ok = normalize(answer) == normalize(row["response"])
        correct += int(ok)
        print(f"[{'OK ' if ok else 'KO '}] {row['prompt']} -> {answer}")

    print(f"\nRéponses exactes hors-jeu : {correct}/{len(heldout)}")


def main():
    parser = argparse.ArgumentParser(description="Génération avec le modèle LoRA 40 exemples.")
    parser.add_argument("--model-dir", default=ADAPTER_DIR, help="Dossier de l'adaptateur LoRA")
    parser.add_argument("--prompt", default="Quelle est la capitale du Canada ?", help="Question à poser")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--eval", action="store_true", help="Tester les 40 exemples et afficher le score")
    parser.add_argument("--heldout", action="store_true", help="Tester uniquement les exemples tenus dehors (graine 42)")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)

    if args.eval:
        evaluate(tokenizer, model)
        return

    if args.heldout:
        evaluate_heldout(tokenizer, model)
        return

    print(generate(
        tokenizer,
        model,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    ))


if __name__ == "__main__":
    main()
