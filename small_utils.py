"""Chargement et génération pour l'expérience LoRA (dataset de 40 exemples).

Les poids de base ne sont pas versionnés : le modèle Hugging Face est téléchargé
au premier lancement, seul l'adaptateur LoRA est sauvegardé localement.
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
ADAPTER_DIR = "PicoGPT1-Jasmine-Tea"
MAX_LEN = 256


def load_model(adapter_dir=ADAPTER_DIR, base_model=BASE_MODEL):
    """Charge le modèle de base, l'adaptateur LoRA et le tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    return tokenizer, model


def generate(
    tokenizer,
    model,
    prompt,
    max_new_tokens=40,
    do_sample=False,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1,
):
    """Génère une réponse et ne renvoie que le texte nouveau (sans le prompt)."""
    inputs = tokenizer(f"{prompt}\n", return_tensors="pt", truncation=True, max_length=MAX_LEN)
    prompt_length = inputs["input_ids"].shape[-1]

    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update(temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)

    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)

    new_tokens = output[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
