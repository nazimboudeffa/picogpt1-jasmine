# 11. Limites et pistes d'amélioration

## Limites actuelles

- Jeu de données réduit à 500 exemples Alpaca.
- Architecture très petite : 2 couches, 128 dimensions.
- Contexte limité à 256 tokens.
- Aucun jeu de validation : le loss affiché est uniquement le loss d'entraînement.
- Décodage glouton par défaut, avec tendance à la répétition.
- GGUF exporté en F16, sans quantification.

## Améliorations possibles

1. **Plus de données** : augmenter `MAX_EXAMPLES` dans `prepare_dataset.py` et `train.py`.
2. **Modèle plus grand** : passer à 4 couches / 256 dimensions, ou partir d'un GPT-2 pré-entraîné.
3. **Évaluation** : réserver 10 % des exemples pour un jeu de validation et suivre `eval_loss`.
4. **Génération** : régler `temperature`, `top_p` et `repetition_penalty` selon l'usage.
5. **GGUF quantifié** : exporter en Q4_K_M pour réduire la taille et accélérer l'inférence.
6. **Historique git** : pousser les commits locaux sur GitHub si le dépôt doit être partagé.

## Rappel

Ce projet sert avant tout à comprendre le cycle complet : donnée → entraînement → scripts Python → GGUF → publication Hugging Face → test dans llama.app.
