---
library_name: transformers
tags:
  - llama
  - smollm2
  - lora
  - gguf
  - french
  - instruction
---

# PicoGPT1-Jasmine-Tea

Variante de `PicoGPT1-Jasmine` qui change de paradigme : au lieu d'entraîner un GPT-2 de zéro sur 500 exemples, elle part de **SmolLM2-135M** pré-entraîné et n'apprend qu'un adaptateur LoRA sur **40 exemples français écrits à la main**.

Le pré-entraînement fournit la langue ; les 40 exemples n'apprennent que le format `question -> réponse courte en français`.

- Base : `HuggingFaceTB/SmolLM2-135M` (135 M paramètres, architecture Llama)
- Adaptation : LoRA `r=16`, toutes les projections attention et MLP (460 800 paramètres entraînés)
- Données : 40 paires prompt / réponse, dont 4 tenues de côté pour l'évaluation
- Contexte : 256 tokens
- Poids publiés : modèle fusionné (base + LoRA), directement utilisable sans PEFT

> ⚠️ Modèle expérimental. Il produit des phrases françaises fluides et respecte le format appris, mais il **invente les faits absents des 40 exemples**. Ce n'est pas un assistant généraliste.

## Utilisation avec Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "nazimboudeffa/PicoGPT1-Jasmine-Tea"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Quelle est la capitale de la France ?\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=40,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Le séparateur `\n` après la question reproduit le format vu à l'entraînement.

## Utilisation avec llama.cpp / llama.app

Si un fichier `PicoGPT1-Jasmine-Tea.gguf` est présent dans ce dépôt :

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Jasmine-Tea
```

```bash
llama serve -hf nazimboudeffa/PicoGPT1-Jasmine-Tea
# puis http://localhost:9931
```

## Résultats mesurés

- Paramètres entraînés : **460 800** sur 134 975 808 (0,34 %).
- `eval_loss` : 2,57 en début d'entraînement → 1,56 vers 20 epochs, puis 2,13 à 60 epochs (sur-apprentissage).
- **36 / 40** réponses exactes sur le jeu complet, dont 36 exemples vus à l'entraînement.
- **0 / 4** sur le split de validation (graine 42 : Joconde, `15 - 8`, pattes de l'araignée, photosynthèse). `python test_small.py --heldout` reproduit ce score.
- **0 / 2** sur deux sondes manuelles hors dataset : « capitale du Canada » → Toronto, « 12 + 5 » → 27.

> Le correctif du sur-apprentissage (`load_best_model_at_end`, ~20 epochs) est prévu dans la release suivante, **PicoGPT1-Jasmine-Mint** ; les poids de Tea restent ceux qui ont été publiés.

## Entraînement et mises à jour

Le code source se trouve dans `main` du dépôt
[picogpt1-jasmine](https://github.com/nazimboudeffa/picogpt1-jasmine).

Pour republier une version mise à jour :

```bash
python train_small.py       # réentraîne l'adaptateur sur dataset_small.jsonl
python merge_tea.py         # fusionne base + adaptateur
python export_gguf_tea.py   # export GGUF pour llama.app
python publish_tea.py       # met à jour ce dépôt Hugging Face
```

Chaque appel écrase les fichiers du dépôt Hugging Face avec la version courante.
