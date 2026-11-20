# 4. Entraînement

## Commande

```bash
python train.py
```

Le script nécessite `train.txt` (voir [dataset](03-dataset.md)).

## Constantes configurables

En haut de `train.py` :

| Constante | Valeur par défaut | Description |
| --- | --- | --- |
| `DATA_PATH` | `train.txt` | Fichier d'exemples |
| `OUTPUT_DIR` | `PicoGPT1-Jasmine` | Dossier de sauvegarde |
| `MAX_LEN` | `256` | Longueur maximale du contexte |
| `MAX_EXAMPLES` | `500` | Nombre d'exemples utilisés |
| `EPOCHS` | `10` | Nombre d'epochs |
| `BATCH_SIZE` | `4` | Taille du batch |
| `LEARNING_RATE` | `5e-4` | Taux d'apprentissage |

`MAX_LEN` est aussi utilisé dans `model_utils.py` pour tronquer les prompts à la génération : les deux valeurs doivent rester identiques.

## Ce que fait le script

1. Charge le tokenizer GPT-2 (`gpt2`) et utilise `<|endoftext|>` comme token de padding.
2. Définit un `chat_template` correspondant au format appris : `prompt\nreponse<|endoftext|>`.
3. Lit `train.txt` et tokenise chaque exemple.
4. Masque le prompt dans les labels (`-100`) : la loss ne porte que sur la réponse.
5. Construit un `GPT2LMHeadModel` avec la configuration décrite dans [architecture](09-architecture.md).
6. Entraîne avec `Trainer` (`logging_steps=50`, `save_steps=1000`).
7. Sauvegarde le modèle et le tokenizer dans `PicoGPT1-Jasmine/`.

## Sorties

Fichiers générés dans `PicoGPT1-Jasmine/` :

- `config.json`
- `generation_config.json`
- `model.safetensors` (ignoré par Git)
- `merges.txt`, `vocab.json`, `tokenizer.json`
- `special_tokens_map.json`, `tokenizer_config.json`

Un dossier `checkpoint-1250/` peut aussi apparaître ; il est ignoré par Git et peut être supprimé sans risque.

## Durée et métriques

Sur CPU, l'exécution complète prend environ **2 min 40**. Dernier entraînement observé :

```
train_runtime : 160 s
train_loss    : 5,90
loss finale   : 4,95
```

Le loss reste élevé : le modèle est volontairement minuscule et entraîné sur peu de données.
