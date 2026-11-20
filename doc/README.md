# Documentation — PicoGPT1-Jasmine

Cette documentation décrit le dépôt local, l'entraînement, la génération, le serveur HTTP, l'export GGUF et la publication sur Hugging Face.

## Sommaire

1. [Présentation](01-presentation.md)
2. [Installation](02-installation.md)
3. [Dataset](03-dataset.md)
4. [Entraînement](04-entrainement.md)
5. [Génération avec `test.py`](05-generation.md)
6. [Serveur local avec `serve.py`](06-serveur-local.md)
7. [Export GGUF](07-gguf.md)
8. [Publication Hugging Face et llama.app](08-huggingface.md)
9. [Architecture du modèle](09-architecture.md)
10. [Dépannage](10-depannage.md)
11. [Limites et pistes d'amélioration](11-limites.md)

## Démarrage rapide

```bash
python -m pip install torch transformers datasets huggingface_hub
python prepare_dataset.py
python train.py
python test.py --prompt "What is 2+2?\n" --do-sample
```

Serveur local :

```bash
python serve.py
```

Export et publication (nécessite `huggingface-cli login`) :

```bash
python export_gguf.py
python publish_hf.py
```

## Liens

- Dépôt GitHub : https://github.com/nazimboudeffa/picogpt1-jasmine
- Modèle Hugging Face : https://huggingface.co/nazimboudeffa/PicoGPT1-Jasmine
- llama.app : https://llama.app
