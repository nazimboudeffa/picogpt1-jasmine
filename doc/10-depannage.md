# 10. Dépannage

## `Poids introuvables dans 'PicoGPT1-Jasmine'`

Les poids ne sont pas versionnés. Lancez :

```bash
python train.py
```

Puis, si vous voulez un GGUF :

```bash
python export_gguf.py
```

## `Token is required ... no token found`

La publication nécessite une connexion Hugging Face :

```bash
huggingface-cli login
```

Ne collez jamais le token dans le chat ni dans un fichier versionné.

## Erreur SSL lors de la publication

Certains réseaux présentent une chaîne de certificats auto-signée de manière intermittente. `publish_hf.py` configure déjà des retries HTTP ; relancez simplement la commande :

```bash
python publish_hf.py
```

## Le modèle ne génère rien avec llama.cpp

Vérifiez que le GGUF a été régénéré après le dernier entraînement et qu'il contient le chat template :

```bash
python export_gguf.py
llama cli -hf nazimboudeffa/PicoGPT1-Jasmine -hff PicoGPT1-Jasmine.gguf
```

## Les réponses sont incohérentes

C'est attendu : 500 exemples et 6,86 M paramètres sont très peu pour un assistant. Essayez :

```bash
python test.py --prompt "Votre texte\n" --do-sample
```

Pour de meilleurs résultats, augmentez le nombre d'exemples et la taille du modèle.

## Port 8000 déjà utilisé

Changez le port du serveur local :

```bash
python serve.py --port 9000
```

## Avertissements Git sur CRLF

Sous Windows, Git peut afficher :

```
LF will be replaced by CRLF the next time Git touches it
```

Ces avertissements sont sans conséquence.

## Dossier `checkpoint-1250/`

Il est créé par `Trainer` et ignoré par Git. Vous pouvez le supprimer ; les poids finals sont dans `model.safetensors`.
