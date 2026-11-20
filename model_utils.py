"""Fonctions partagées pour charger le modèle et générer une réponse."""

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "PicoGPT1-Jasmine"
MAX_LEN = 256  # doit rester identique à MAX_LEN dans train.py
DEFAULT_MAX_NEW_TOKENS = 50


def load_model(model_dir=MODEL_DIR):
    """Charge le tokenizer et le modèle, avec un message clair si les poids manquent."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir)
    except OSError as exc:
        raise SystemExit(
            f"Poids introuvables dans {model_dir!r}. Lancez d'abord `python train.py`."
        ) from exc

    model.eval()
    # Le generation_config sauvegardé contient max_length=20, ce qui entre en conflit
    # avec max_new_tokens. On le neutralise pour garder un seul réglage.
    model.generation_config.max_length = None
    return tokenizer, model


def generate(
    tokenizer,
    model,
    prompt,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    do_sample=False,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1,
):
    """Génère une réponse et renvoie le texte complet (prompt + réponse)."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs.update(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    output_ids = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
