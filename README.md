# PicoGPT1 Scratch — Entraînement et utilisation d'un Mini GPT-2

Ce projet montre comment entraîner un petit modèle GPT-2 sur un dataset d'instructions/réponses, puis l'utiliser directement en Python. Les scripts Python ne nécessitent ni llama.cpp ni Ollama ; une variante GGUF est aussi publiée pour llama.app.

Dépôt : https://github.com/nazimboudeffa/picogpt1-jasmine

Documentation détaillée : [doc/README.md](doc/README.md)

## 1. Préparation du dataset

Le script `prepare_dataset.py` télécharge un sous-ensemble du dataset [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) et le convertit au format texte attendu :

```
<prompt>
<réponse>

<prompt>
<réponse>
...etc
```

Exécutez :

```bash
python prepare_dataset.py
```

Cela crée le fichier `train.txt`.

## 2. Entraînement du modèle

L'entraînement se fait directement avec le script Python `train.py` :

```bash
python train.py
```

Les constantes en haut du fichier (`MAX_LEN`, `MAX_EXAMPLES`, `EPOCHS`, etc.) permettent de modifier la configuration sans éditer le reste du code.

Étapes effectuées par le script :
- Chargement du tokenizer GPT-2
- Chargement et formatage du dataset
- Définition d'une petite architecture GPT-2 (2 couches, 128 dimensions)
- Entraînement avec HuggingFace Transformers (`Trainer`)
- Sauvegarde du modèle et du tokenizer dans le dossier `PicoGPT1-Jasmine/`

Les notebooks `train.ipynb` et `test.ipynb` sont conservés comme référence, mais ne sont plus nécessaires.

## 3. Tester le modèle

Le script `test.py` charge le modèle entraîné et génère une réponse. Lancez d'abord `train.py` (les poids ne sont pas versionnés dans le dépôt) :

```bash
python train.py
python test.py
```

Pour tester un autre prompt, passez-le en argument :

```bash
python test.py --prompt "Quelle est la capitale de la France ?"
```

L'option `--do-sample` active l'échantillonnage (`--temperature`, `--top-p`, `--repetition-penalty`) pour limiter les répétitions du décodage glouton :

```bash
python test.py --prompt "Explique la psychanalyse." --do-sample
```

## 4. Servir le modèle en local (sans Ollama)

Le script `serve.py` démarre un petit serveur HTTP basé uniquement sur la bibliothèque standard de Python. Il garde le modèle en mémoire et expose une seule route : `POST /generate`.

```bash
python serve.py
```

Interrogez-le depuis un autre terminal.

Avec PowerShell :

```powershell
$body = @{ prompt = "What is psychoanalysis?`n"; max_new_tokens = 50 } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/generate -Method Post -Body $body -ContentType "application/json"
```

Avec curl (bash) :

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is psychoanalysis?"}'
```

Le champ `do_sample` peut aussi être envoyé dans le JSON pour activer l'échantillonnage. Les options `--host`, `--port` et `--model-dir` permettent de configurer le serveur :

```bash
python serve.py --port 9000 --model-dir PicoGPT1-Jasmine
```

## 5. Publier sur Hugging Face et tester avec llama.app

Le modèle est publié ici : https://huggingface.co/nazimboudeffa/PicoGPT1-Jasmine

Après vous être connecté avec `huggingface-cli login`, générez le GGUF puis publiez les poids, la fiche modèle et le GGUF :

```bash
python export_gguf.py
python publish_hf.py
```

Le dépôt contient les poids Transformers et `PicoGPT1-Jasmine.gguf`. Avec le CLI installé par [llama.app](https://llama.app) :

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Jasmine -hff PicoGPT1-Jasmine.gguf
```

Pour ouvrir l'interface de chat dans le navigateur :

```bash
llama serve -hf nazimboudeffa/PicoGPT1-Jasmine -hff PicoGPT1-Jasmine.gguf
# http://localhost:9931
```

Le modèle est très petit (26 Mo) : les réponses sont attendues comme approximatives.

## 6. Références

- [Vidéo originale](https://www.youtube.com/watch?v=MpHj4UD4kiw)
- [Repo original](https://github.com/hassanhabib/LLMfromNothing)
