# 1. Présentation

## Objectif

`PicoGPT1-Jasmine` est un mini modèle GPT-2 entraîné à partir de zéro sur un petit jeu d'instructions/réponses. Le projet utilise des **scripts Python directs** : Jupyter n'est pas nécessaire pour entraîner ou tester le modèle.

Les scripts locaux n'utilisent **ni llama.cpp, ni Ollama**. Une variante GGUF est toutefois exportée pour permettre de tester avec [llama.app](https://llama.app).

## État actuel

| Élément | État |
| --- | --- |
| Entraînement | Terminé (10 epochs, 1 250 steps, loss finale ≈ 4,95) |
| Poids v0.1 | Non présents dans le dépôt ; baseline disponible via le tag `v0.1.0-jasmine` (`v1.0.0-jasmine` conservé comme tag historique) |
| Publication | https://huggingface.co/nazimboudeffa/PicoGPT1-Jasmine |
| Compatibilité llama.app | Confirmée via `llama cli -hf ...` |
| Branche Git locale | `main` |

> Le modèle est expérimental et très petit : les réponses générées sont souvent incohérentes. C'est un projet d'apprentissage, pas un assistant généraliste.

## Arborescence

```
.
├── PicoGPT1-Jasmine-Tea/    # adaptateur LoRA Tea (versionné)
├── dataset_small.jsonl      # 40 exemples français
├── doc/                     # cette documentation, dont 12-jasmine-tea.md
├── export_gguf.py           # scripts v1 (baseline taguée)
├── model_utils.py
├── prepare_dataset.py
├── publish_hf.py
├── serve.py
├── test.py
├── train.py
├── merge_tea.py             # scripts Tea
├── export_gguf_tea.py
├── publish_tea.py
├── small_utils.py
├── test_small.py
└── README.md                # vue d'ensemble
```

Le dossier `PicoGPT1-Jasmine/` n'est plus versionné : `train.py` le recrée si tu relances la baseline, sinon le tag `v0.1.0-jasmine` donne accès à l'état d'origine (`v1.0.0-jasmine` reste disponible comme tag historique).

## Versions utilisées lors du développement

- Python 3.13 sur Windows 11
- `torch` (CPU)
- `transformers` 4.52.4
- `datasets`
- `huggingface_hub` 0.33.0
- CLI llama.cpp / llama.app build b12375
