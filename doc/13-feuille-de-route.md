# 13. Feuille de route des versions `0.x`

Ce document décrit la lignée expérimentale **Jasmine** de la famille **PicoGPT1**. La convention de nommage est celle de `picogpt-kit` : `PicoGPT<majeure>-<Fleur>`, avec un tag `v<version>-<fleur>`.

## Pourquoi `0.x`

Dans `picogpt-kit`, Jasmine est décrite comme le **prototype historique, conservé tel quel** : `0.x`, GPT-2 6,86 M paramètres entraîné de zéro sur 500 exemples Alpaca. Les releases `1.0.0` (Daisy) et `1.1.0` (Iris) sont, elles, dans le dépôt `picogpt-kit`.

Ce dépôt-ci ne « monte » donc pas en `1.x` : il documente les variantes greffées sur le prototype Jasmine. Le `1` de `PicoGPT1` reste le numéro de **famille**, pas le numéro de version de la release.

> L'ancien tag `v1.0.0-jasmine` a été supprimé après création du tag conforme `v0.1.0-jasmine`, qui pointe sur le même commit (`d059b54`). Aucun contenu n'a été perdu. Dans un clone existant, nettoyer avec `git fetch --prune --tags` puis `git tag -d v1.0.0-jasmine`.

## Convention de nommage

| Élément | Règle |
| --- | --- |
| Nom du modèle | `PicoGPT1-Jasmine-<CodeName>` |
| Nom de la baseline | `PicoGPT1-Jasmine` |
| Tag de la baseline | `v0.N.0-jasmine` |
| Tag d'une variante | `v0.N.0-jasmine-<codename>` |
| Branche de travail | `experiment/<code-name>` |
| Dataset versionné | `data/<codename>.jsonl` |
| Fiche modèle | `MODEL_CARD_<CODE>.md` |
| Dépôt Hugging Face | `nazimboudeffa/PicoGPT1-Jasmine-<CodeName>` — **un repo par codename** |

Le suffixe `-jasmine-<codename>` du tag est une extension locale de la règle `v<version>-<fleur>` du kit : il garde visible la lignée dont vient l'expérience. Les codenames ne sont pas des fleurs, pour ne pas entrer en collision avec le registre de `picogpt-kit`.

## Règles d'or

1. **Une seule variable par expérience.** Si deux choses changent, la comparaison n'a plus de sens.
2. **Budget minimal** : base ≤ 135 M paramètres, ≤ 200 exemples écrits à la main, ≤ ~5 min d'entraînement CPU, adaptateur versionné ≤ ~20 Mo, GGUF optionnel.
3. **Évaluation figée** : split 36/4 (graine 42) pour Tea, puis 8 exemples tenus dehors à partir de Matcha. Le script `test_small.py --heldout` mesure uniquement ces exemples ; `--eval` mesure le jeu complet.
4. **Résultat négatif = résultat quand même.** Une variante qui échoue garde sa fiche avec ses chiffres et son statut `écarté`.
5. **Rien ne change sur Hugging Face sans décision explicite.** Les identifiants `REPO_ID` et les noms de GGUF sont figés ; les tags Git n'ont aucun effet sur les dépôts Hub.

## Trajectoire proposée

| Version | Codename | Une seule variable changée | Hypothèse | Statut |
| --- | --- | --- | --- | --- |
| `0.1.0` | Jasmine | — | Baseline GPT-2 6,86 M, 500 exemples Alpaca | publié (alias `v0.1.0-jasmine`) |
| `0.2.0` | Tea | base pré-entraînée + LoRA, 40 exemples | La langue vient du pré-entraînement, les exemples n'apprennent que le format | publié, `v0.2.0-jasmine-tea` |
| `0.3.0` | Mint | `load_best_model_at_end` + early stopping (~20 epochs) | Réduire le sur-apprentissage observé à 60 epochs | à venir |
| `0.4.0` | Matcha | 40 → 80 exemples | Couvrir un peu plus de faits sans grossir le modèle | proposé |
| `0.5.0` | Chamomile | LoRA `r=8`, cibles `q_proj` / `v_proj` seulement | Moins de capacité ⇒ moins de mémorisation | proposé |
| `0.6.0` | Rooibos | base `SmolLM2-135M-Instruct` | Même taille, meilleure tenue du style hors dataset | proposé |
| `0.7.0` | Infusion | export GGUF `Q4_K_M` uniquement | Réduire la taille sans réentraîner | proposé |

Le sur-apprentissage de Tea (`eval_loss` 2,57 → 1,56 → 2,13) n'est **pas** corrigé dans Tea : les poids fusionnés sont déjà publiés. Il est traité dans Mint, qui ne change que le nombre d'epochs effectives.

## Procédure à chaque release

```bash
# 1. entraînement / fusion / export dans une branche experiment/<code-name>
python train_small.py
python merge_tea.py
python export_gguf_tea.py

# 2. mesure
python test_small.py --heldout

# 3. publication Hub (nouveau repo par codename)
python publish_tea.py

# 4. tag git
git tag -a v0.N.0-jasmine-<codename> -m "Release PicoGPT1-Jasmine-<CodeName>"
git push origin v0.N.0-jasmine-<codename>
```

Le passage à une famille `PicoGPT2` relève de `picogpt-kit` (prévu à `2.0.0`) et n'est pas dans le périmètre de cette feuille de route.
