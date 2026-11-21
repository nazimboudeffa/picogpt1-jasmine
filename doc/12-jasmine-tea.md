# PicoGPT1-Jasmine-Tea

Variante expérimentale de `v1.0.0-jasmine`, développée sur la branche `experiment/tiny-dataset`.

## Principe

`v1.0.0-jasmine` entraîne un GPT-2 de 6,86 M paramètres **de zéro** sur 500 exemples Alpaca. Avec si peu de données, le modèle doit apprendre en même temps la langue, les connaissances et le format : le résultat reste incohérent (loss ≈ 4,95).

`PicoGPT1-Jasmine-Tea` change de source de connaissance :

| | v1.0.0-jasmine | Jasmine-Tea |
| --- | --- | --- |
| Modèle de base | GPT-2 initialisé au hasard | SmolLM2-135M pré-entraîné |
| Entraînement | Tous les poids | Adaptateur LoRA (0,34 % des poids) |
| Données | 500 couples Alpaca en anglais | 40 couples écrits à la main, en français |
| Rôle des données | Apprendre la langue **et** la tâche | Apprendre uniquement le format question → réponse courte |
| Résultat | Réponses approximatives | Français fluide, format respecté |

La connaissance vient donc du pré-entraînement ; les 40 exemples ne servent qu'à orienter le style et le format.

## Fichiers ajoutés

| Fichier | Rôle |
| --- | --- |
| `dataset_small.jsonl` | Les 40 paires prompt / réponse (versionné, contrairement à `train.txt`) |
| `train_small.py` | Entraînement LoRA et sauvegarde dans `PicoGPT1-Jasmine-Tea/` |
| `small_utils.py` | Chargement base + adaptateur et génération |
| `test_small.py` | Test CLI, avec option `--eval` sur les 40 exemples |

## Utilisation

Dépendance supplémentaire :

```bash
python -m pip install peft
```

Entraînement (~2 min 30 sur CPU, une fois le modèle de base téléchargé) :

```bash
python train_small.py
```

Génération :

```bash
python test_small.py
python test_small.py --prompt "Quelle est la capitale du Canada ?"
python test_small.py --do-sample --temperature 0.7
```

Évaluation sur les 40 exemples :

```bash
python test_small.py --eval
```

## Configuration

| Constante | Valeur |
| --- | --- |
| `BASE_MODEL` | `HuggingFaceTB/SmolLM2-135M` |
| `DATA_PATH` | `dataset_small.jsonl` |
| `OUTPUT_DIR` | `PicoGPT1-Jasmine-Tea` |
| `MAX_LEN` | 256 |
| `EPOCHS` | 60 |
| `BATCH_SIZE` | 4 |
| `LEARNING_RATE` | `3e-4` |
| LoRA | `r=16`, `alpha=32`, dropout `0,05` |
| Modules ciblés | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Split | 36 entraînement / 4 évaluation (graine 42) |

Les poids de base ne sont pas versionnés : ils sont téléchargés depuis Hugging Face au premier lancement. L'adaptateur est sauvegardé localement dans `PicoGPT1-Jasmine-Tea/`, dossier ignoré par Git. Le tokenizer est rechargé depuis le modèle de base, donc seul l'adaptateur est nécessaire pour générer.

## Résultats observés

- Paramètres entraînés : **460 800** sur 134 975 808 (0,34 %).
- `eval_loss` : 2,57 en début d'entraînement → 1,56 autour de 20 epochs, puis remontée à 2,13 à 60 epochs (sur-apprentissage sur 36 exemples).
- **36 / 40 réponses exactes** avec le dernier adaptateur : les 36 exemples vus sont restitués correctement, les 4 exemples tenus dehors ne le sont pas.

Exemples hors dataset :

```
Quelle est la capitale du Canada ?  -> Toronto est la capitale du Canada.   (faux : Ottawa)
Combien font 12 + 5 ?               -> 12 + 5 font 27.                       (faux : 17)
```

## Limites

Le gain porte sur la **forme**, pas sur le fond. Le modèle produit des phrases françaises bien construites et respecte le format appris, mais il invente les faits absents des 40 exemples. Une augmentation du nombre d'exemples ou le passage à un modèle de base plus grand sont nécessaires pour améliorer la fiabilité.

Le sur-apprentissage est rapide : au-delà d'une vingtaine d'epochs, la loss d'évaluation augmente alors que la loss d'entraînement s'effondre. Configurer `load_best_model_at_end` avec `metric_for_best_model="eval_loss"` permettrait de récupérer automatiquement le meilleur point.
