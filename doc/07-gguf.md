# 7. Export GGUF

## Pourquoi un GGUF ?

Les scripts Python utilisent directement les poids Transformers (`model.safetensors`). llama.cpp / llama.app attendent un fichier **GGUF**. Le script `export_gguf.py` convertit le modèle local en `PicoGPT1-Jasmine.gguf`.

## Commande

```bash
python export_gguf.py
```

Le fichier généré est ignoré par Git (`*.gguf` dans `.gitignore`).

## Fonctionnement

1. Télécharge le convertisseur `convert_hf_to_gguf.py` depuis une version épinglée de llama.cpp (`b4000`).
2. Installe le paquet Python `gguf` dans un dossier temporaire s'il n'est pas déjà disponible.
3. Lance la conversion avec `--outtype f16`.
4. Supprime les fichiers temporaires à la fin.

Le modèle GPT-2 est pris en charge par le convertisseur. La taille du GGUF généré est d'environ **26,7 Mo** pour 29 tenseurs.

## Vérification locale

```bash
llama cli -m PicoGPT1-Jasmine.gguf -st -p "What is 2+2?" -n 20
```

Le GGUF contient le `chat_template` défini dans `tokenizer_config.json`, ce qui permet à llama.cpp d'utiliser le mode chat.

## Relance après réentraînement

Après chaque `python train.py`, relancez :

```bash
python export_gguf.py
python publish_hf.py
```

Les deux fichiers doivent rester synchronisés : le GGUF est généré à partir des poids présents dans `PicoGPT1-Jasmine/`.
