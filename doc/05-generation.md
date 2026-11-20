# 5. Génération avec `test.py`

## Commande simple

```bash
python test.py
```

Par défaut, le prompt est :

```
What is the capital of France?
```

Le script affiche le texte complet : prompt + réponse générée.

## Options

```bash
python test.py --prompt "Explique la psychanalyse.\n" --do-sample
```

| Option | Défaut | Description |
| --- | --- | --- |
| `--model-dir` | `PicoGPT1-Jasmine` | Dossier du modèle |
| `--prompt` | `What is the capital of France?\n` | Texte à compléter |
| `--max-new-tokens` | `50` | Nombre maximum de tokens générés |
| `--do-sample` | désactivé | Active l'échantillonnage |
| `--temperature` | `0.8` | Température si `--do-sample` |
| `--top-p` | `0.9` | Nucleus sampling si `--do-sample` |
| `--repetition-penalty` | `1.1` | Pénalité de répétition si `--do-sample` |

Le décodage glouton (par défaut) a tendance à répéter ; `--do-sample` donne des résultats plus variés.

## Fonctions partagées

`model_utils.py` contient deux fonctions utilisées par `test.py` et `serve.py` :

- `load_model(model_dir)` : charge tokenizer + modèle, passe en mode évaluation et neutralise `generation_config.max_length`.
- `generate(tokenizer, model, prompt, ...)` : tokenise le prompt (tronqué à 256 tokens), appelle `model.generate` et renvoie le texte décodé.

Erreur attendue si les poids sont absents :

```
Poids introuvables dans 'PicoGPT1-Jasmine'. Lancez d'abord `python train.py`.
```

## Conseil sur le prompt

Le modèle a été entraîné avec un saut de ligne entre le prompt et la réponse. Ajoutez donc `\n` à la fin de votre prompt pour rester proche du format d'entraînement.

Exemple PowerShell :

```powershell
python test.py --prompt "What is 2+2?`n" --do-sample
```
