"""Fusionne l'adaptateur LoRA de Jasmine-Tea dans le modèle de base.

Usage :
    python merge_tea.py

Produit un dossier `PicoGPT1-Jasmine-Tea-Merged/` directement chargeable avec
`AutoModelForCausalLM.from_pretrained()` et publiable sur Hugging Face.
"""

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
ADAPTER_DIR = Path("PicoGPT1-Jasmine-Tea")
OUTPUT_DIR = Path("PicoGPT1-Jasmine-Tea-Merged")


def main() -> None:
    if not (ADAPTER_DIR / "adapter_config.json").is_file():
        raise SystemExit(f"Adaptateur introuvable dans {ADAPTER_DIR}. Lancez d'abord `python train_small.py`.")

    print(f"Fusion de {ADAPTER_DIR} dans {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model = model.merge_and_unload()
    model.config.use_cache = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Modèle fusionné sauvegardé dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
