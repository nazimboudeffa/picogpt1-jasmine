# 8. Publication Hugging Face et llama.app

## Dépôt publié

https://huggingface.co/nazimboudeffa/PicoGPT1-Jasmine

Le dépôt est public et contient :

- les poids Transformers et le tokenizer ;
- `PicoGPT1-Jasmine.gguf` ;
- le `README.md` (fiche modèle) généré depuis `MODEL_CARD.md`.

Hugging Face reconnaît automatiquement les tags `gguf`, `llama.cpp`, `llama.app` et affiche les commandes correspondantes dans « Use this model ».

## Publication

Prérequis : être connecté via `huggingface-cli login` (ou avoir `HF_TOKEN`), et avoir généré le GGUF.

```bash
python export_gguf.py
python publish_hf.py
```

`publish_hf.py` :

1. Crée le dépôt s'il n'existe pas (`private=False`, `exist_ok=True`).
2. Upload le contenu de `PicoGPT1-Jasmine/` en ignorant `checkpoint-*`.
3. Upload `MODEL_CARD.md` comme `README.md`.
4. Upload `PicoGPT1-Jasmine.gguf`.

Le script configure une session HTTP avec retries pour absorber les erreurs TLS intermittentes observées sur certains réseaux.

## Tester avec le CLI llama.app

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Jasmine -hff PicoGPT1-Jasmine.gguf
```

`-hff` force le fichier GGUF à utiliser. Comme le dépôt ne contient qu'un seul GGUF, la forme courte fonctionne aussi :

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Jasmine
```

## Interface web de chat

```bash
llama serve -hf nazimboudeffa/PicoGPT1-Jasmine -hff PicoGPT1-Jasmine.gguf
# http://localhost:9931
```

Le port par défaut du serveur llama.app est `9931`.

## Tester avec Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "nazimboudeffa/PicoGPT1-Jasmine"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Quelle est la capitale de la France ?\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=48, do_sample=True, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
