# 2. Installation

## Prérequis

- Python 3.10 ou plus récent (testé avec Python 3.13)
- Git
- Environnement Windows, macOS ou Linux

## Cloner le dépôt

```bash
git clone https://github.com/nazimboudeffa/picogpt1-jasmine.git
cd picogpt1-jasmine
```

## Dépendances Python

```bash
python -m pip install torch transformers datasets huggingface_hub
```

`tokenizers`, `accelerate` et `safetensors` sont installés automatiquement avec `transformers`.

Vérification rapide :

```bash
python -m py_compile train.py test.py serve.py model_utils.py prepare_dataset.py export_gguf.py publish_hf.py
```

Aucune sortie signifie que la syntaxe est correcte.

## Connexion Hugging Face

La connexion n'est nécessaire **que pour publier** le modèle :

```bash
huggingface-cli login
```

Le token est saisi dans le formulaire interactif. Il ne doit jamais être collé dans un chat ou commité dans le dépôt.

## llama.app / CLI llama.cpp

Pour tester le modèle publié avec llama.app :

- Windows : installer l'application llama.app, ou `winget install llama.cpp`
- macOS/Linux : `curl -LsSf https://llama.app/install.sh | sh`

Vérifier l'installation :

```bash
llama --version
```

## Poids du modèle

Les poids (`model.safetensors`) et le GGUF sont ignorés par Git. Deux options :

1. Entraîner le modèle localement : `python train.py`
2. Télécharger le modèle publié : `llama cli -hf nazimboudeffa/PicoGPT1-Jasmine -hff PicoGPT1-Jasmine.gguf`
