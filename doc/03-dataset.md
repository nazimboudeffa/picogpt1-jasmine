# 3. Dataset

## Générer `train.txt`

```bash
python prepare_dataset.py
```

Le script télécharge les 500 premiers exemples de `tatsu-lab/alpaca` et écrit `train.txt`. Ce fichier est ignoré par Git.

## Format attendu

`train.py` lit le fichier par blocs séparés par une ligne vide. Chaque bloc doit contenir **exactement deux lignes non vides** :

```
<prompt>
<réponse>

<prompt>
<réponse>
```

Exemple :

```
What is the capital of France?
Paris

Explain photosynthesis in one sentence.
Plants convert light into chemical energy.
```

Si l'instruction Alpaca contient un champ `input`, `prepare_dataset.py` l'intègre sur la même ligne :

```
Write a sentence about X Input: X
X is an example.
```

Cela garantit que le prompt reste sur une seule ligne et que la réponse est bien la deuxième ligne du bloc.

## Constantes

Dans `prepare_dataset.py` :

| Constante | Valeur | Rôle |
| --- | --- | --- |
| `MAX_EXAMPLES` | `500` | Nombre d'exemples téléchargés/écrits |
| `OUTPUT_FILE` | `train.txt` | Fichier de sortie |

Dans `train.py` :

| Constante | Valeur | Rôle |
| --- | --- | --- |
| `DATA_PATH` | `train.txt` | Fichier lu à l'entraînement |
| `MAX_EXAMPLES` | `500` | Nombre maximum d'exemples utilisés |

Les deux valeurs `MAX_EXAMPLES` doivent rester cohérentes si vous changez la taille du jeu de données.

## Dataset personnalisé

Vous pouvez remplacer `train.txt` par vos propres couples prompt/réponse, à condition de respecter le format ci-dessus. Lancez ensuite `python train.py`.

Pour un format multi-lignes, il faut d'abord adapter le parseur dans `train.py` (`load_dataset_from_file`).
