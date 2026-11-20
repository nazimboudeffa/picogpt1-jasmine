# PicoGPT1-Jasmine

**Un mini GPT-2 entraîné de zéro, du dataset jusqu'au chat dans le navigateur.**

`PicoGPT1-Jasmine` est un projet pédagogique qui montre le cycle complet d'un petit modèle de langage : préparation d'un jeu de données d'instructions, entraînement d'un GPT-2 minuscule en Python, génération en ligne de commande ou via un serveur HTTP local, export au format GGUF, publication sur Hugging Face et test avec [llama.app](https://llama.app).

Tous les scripts Python fonctionnent **sans llama.cpp ni Ollama** : ils utilisent directement les poids Transformers. Le format GGUF n'est produit que pour une utilisation optionnelle avec llama.cpp / llama.app.

> ⚠️ **Modèle expérimental** : 500 exemples d'entraînement et 6,86 M paramètres. Les réponses sont souvent approximatives, voire incohérentes. L'objectif est de comprendre et de faire fonctionner la chaîne complète, pas de fournir un assistant généraliste.

---

## Sommaire

1. [Présentation](#1-présentation)
2. [Structure du dépôt](#2-structure-du-dépôt)
3. [Prérequis](#3-prérequis)
4. [Installation](#4-installation)
5. [Démarrage rapide](#5-démarrage-rapide)
6. [Préparer le dataset](#6-préparer-le-dataset)
7. [Entraîner le modèle](#7-entraîner-le-modèle)
8. [Générer une réponse avec `test.py`](#8-générer-une-réponse-avec-testpy)
9. [Serveur HTTP local avec `serve.py`](#9-serveur-http-local-avec-servepy)
10. [Exporter en GGUF](#10-exporter-en-gguf)
11. [Publier sur Hugging Face](#11-publier-sur-hugging-face)
12. [Utiliser le modèle publié](#12-utiliser-le-modèle-publié)
13. [Architecture du modèle](#13-architecture-du-modèle)
14. [Dépannage](#14-dépannage)
15. [Limites et pistes d'amélioration](#15-limites-et-pistes-damélioration)
16. [Liens et références](#16-liens-et-références)

---

## 1. Présentation

### Ce que fait le projet

| Étape | Script | Résultat |
| --- | --- | --- |
| Données | `prepare_dataset.py` | `train.txt` (500 couples prompt / réponse issus d'Alpaca) |
| Entraînement | `train.py` | Poids Transformers dans `PicoGPT1-Jasmine/` |
| Test CLI | `test.py` | Affichage d'une réponse générée |
| API locale | `serve.py` | `POST /generate` sur `http://127.0.0.1:8000` |
| Export | `export_gguf.py` | `PicoGPT1-Jasmine.gguf` (~26,7 Mo) |
| Publication | `publish_hf.py` | Dépôt Hugging Face public |

### État actuel

| Élément | Détail |
| --- | --- |
| Entraînement de référence | 10 epochs, 1 250 steps, loss finale ≈ 4,95 |
| Poids locaux | `PicoGPT1-Jasmine/model.safetensors` (**non versionné**) |
| Modèle publié | [nazimboudeffa/PicoGPT1-Jasmine](https://huggingface.co/nazimboudeffa/PicoGPT1-Jasmine) |
| Compatibilité llama.app | Confirmée via `llama cli -hf ...` |
| Versions utilisées | Python 3.13 / Windows 11, `transformers` 4.52.4, `huggingface_hub` 0.33.0, `torch` CPU |

---

## 2. Structure du dépôt

```
.
├── PicoGPT1-Jasmine/          # modèle entraîné (config, tokenizer, poids)
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors      # généré, ignoré par Git
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── checkpoint-1250/       # checkpoint Trainer, ignoré par Git
├── doc/                       # documentation détaillée, chapitre par chapitre
├── export_gguf.py             # Transformers → GGUF
├── model_utils.py             # chargement du modèle + génération (partagé)
├── prepare_dataset.py         # téléchargement Alpaca → train.txt
├── publish_hf.py              # publication du modèle sur Hugging Face
├── serve.py                   # serveur HTTP local
├── test.py                    # génération en ligne de commande
├── train.py                   # entraînement
├── train.txt                  # dataset généré, ignoré par Git
├── PicoGPT1-Jasmine.gguf      # export GGUF, ignoré par Git
├── MODEL_CARD.md              # fiche modèle publiée sur le Hub
└── README.md                  # ce fichier
```

**Rôle des fichiers Python**

| Fichier | Responsabilité |
| --- | --- |
| `prepare_dataset.py` | Télécharge un sous-ensemble de `tatsu-lab/alpaca` et écrit `train.txt` |
| `train.py` | Construit le GPT-2, entraîne sur `train.txt`, sauvegarde dans `PicoGPT1-Jasmine/` |
| `model_utils.py` | `load_model()` et `generate()`, utilisés par `test.py` et `serve.py` |
| `test.py` | Interface CLI pour générer une réponse |
| `serve.py` | Serveur HTTP local basé sur la bibliothèque standard |
| `export_gguf.py` | Télécharge le convertisseur llama.cpp épinglé et produit le `.gguf` |
| `publish_hf.py` | Crée le dépôt Hub et upload poids, fiche modèle et GGUF |

Pour approfondir chaque partie, consultez [doc/README.md](doc/README.md).

---

## 3. Prérequis

- **Python 3.10 ou supérieur** (testé avec Python 3.13)
- **Git**
- Windows, macOS ou Linux
- Environnement **CPU suffisant** : l'entraînement complet prend ≈ 2 min 40 sur CPU
- Accès Internet pour télécharger le dataset, les dépendances et (si besoin) le modèle publié

Optionnels :

- Un compte [Hugging Face](https://huggingface.co) avec un token d'écriture, uniquement pour publier
- Le CLI [llama.app](https://llama.app) / llama.cpp, uniquement pour tester le GGUF

---

## 4. Installation

### 4.1 Cloner le dépôt

```bash
git clone https://github.com/nazimboudeffa/picogpt1-jasmine.git
cd picogpt1-jasmine
```

### 4.2 Créer un environnement virtuel (recommandé)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4.3 Installer les dépendances

```bash
python -m pip install --upgrade pip
python -m pip install torch transformers datasets huggingface_hub
```

`tokenizers`, `accelerate`, `safetensors`, `requests` et `urllib3` sont installés automatiquement comme dépendances. Aucun `requirements.txt` n'est nécessaire.

### 4.4 Vérifier l'installation

```bash
python -m py_compile train.py test.py serve.py model_utils.py prepare_dataset.py export_gguf.py publish_hf.py
```

Aucune sortie signifie que la syntaxe est correcte. Vous pouvez aussi vérifier les versions :

```bash
python -c "import torch, transformers, datasets, huggingface_hub; print(torch.__version__, transformers.__version__, datasets.__version__, huggingface_hub.__version__)"
```

### 4.5 Connexion Hugging Face (facultatif)

Nécessaire **uniquement** pour `publish_hf.py` :

```bash
huggingface-cli login
```

Le token est saisi dans l'invite interactive. Ne le collez jamais dans un chat, un script ou un fichier versionné. La variable d'environnement `HF_TOKEN` est également acceptée.

### 4.6 Installer le CLI llama.app (facultatif)

```bash
# Windows
winget install llama.cpp

# macOS / Linux
curl -LsSf https://llama.app/install.sh | sh

# Vérification
llama --version
```

Sous Windows, l'application [llama.app](https://llama.app) peut aussi être installée directement.

### 4.7 Attention : les poids ne sont pas dans Git

Le fichier `.gitignore` exclut `*.safetensors`, `*.gguf`, `*.pt`, `train.txt` et les checkpoints. Après un clonage, **il n'y a donc pas de modèle utilisable**. Deux solutions :

1. **Entraîner localement** : `python prepare_dataset.py` puis `python train.py` (recommandé pour comprendre le projet).
2. **Télécharger le modèle publié** via llama.app (voir [section 12](#12-utiliser-le-modèle-publié)).

---

## 5. Démarrage rapide

Enchaînement complet, depuis le dépôt cloné :

```bash
python -m pip install torch transformers datasets huggingface_hub
python prepare_dataset.py       # crée train.txt
python train.py                 # entraîne et sauvegarde dans PicoGPT1-Jasmine/
python test.py --prompt "What is 2+2?\n" --do-sample
python serve.py                 # API locale sur le port 8000
```

Publication et export (nécessitent `huggingface-cli login`) :

```bash
python export_gguf.py
python publish_hf.py
```

> Le `\n` à la fin du prompt est important : il reproduit le séparateur utilisé pendant l'entraînement. Sous PowerShell, écrivez `` `n `` à la place : `--prompt "What is 2+2?`n"`.

---

## 6. Préparer le dataset

```bash
python prepare_dataset.py
```

Le script télécharge le split `train` de [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca), garde les **500 premiers** exemples et écrit `train.txt`.

### Format attendu

`train.py` lit le fichier par blocs séparés par une ligne vide. Chaque bloc doit contenir **exactement deux lignes non vides** : le prompt puis la réponse.

```
<prompt>
<réponse>

<prompt>
<réponse>
```

Exemple concret :

```
What is the capital of France?
Paris

Explain photosynthesis in one sentence.
Plants convert light into chemical energy.
```

Si l'exemple Alpaca possède un champ `input`, il est ajouté sur la même ligne que l'instruction (`Write a sentence about X Input: X`) afin que le prompt reste sur une seule ligne.

### Constantes

| Fichier | Constante | Défaut | Rôle |
| --- | --- | --- | --- |
| `prepare_dataset.py` | `MAX_EXAMPLES` | `500` | Nombre d'exemples téléchargés et écrits |
| `prepare_dataset.py` | `OUTPUT_FILE` | `train.txt` | Fichier de sortie |
| `train.py` | `DATA_PATH` | `train.txt` | Fichier lu à l'entraînement |
| `train.py` | `MAX_EXAMPLES` | `500` | Nombre maximum d'exemples utilisés |

Les deux `MAX_EXAMPLES` doivent rester cohérents si vous changez la taille du jeu de données.

### Dataset personnalisé

Remplacez `train.txt` par vos propres couples prompt / réponse en respectant le format ci-dessus, puis relancez `python train.py`. Pour des réponses sur plusieurs lignes, il faut d'abord adapter le parseur `load_dataset_from_file()` dans `train.py`.

---

## 7. Entraîner le modèle

```bash
python train.py
```

`train.txt` doit exister (voir section précédente).

### Constantes configurables

En tête de `train.py` :

| Constante | Défaut | Description |
| --- | --- | --- |
| `DATA_PATH` | `train.txt` | Fichier d'exemples |
| `OUTPUT_DIR` | `PicoGPT1-Jasmine` | Dossier de sauvegarde |
| `MAX_LEN` | `256` | Longueur maximale du contexte |
| `MAX_EXAMPLES` | `500` | Nombre d'exemples utilisés |
| `EPOCHS` | `10` | Nombre d'epochs |
| `BATCH_SIZE` | `4` | Taille du batch |
| `LEARNING_RATE` | `5e-4` | Taux d'apprentissage |

> `MAX_LEN` est également défini dans `model_utils.py` pour tronquer les prompts à la génération. Les deux valeurs doivent rester identiques.

### Ce que fait le script

1. Charge le tokenizer GPT-2 (`gpt2`) et utilise `<|endoftext|>` comme token EOS et PAD.
2. Définit un `chat_template` correspondant au format appris `prompt\nreponse<|endoftext|>`.
3. Lit `train.txt` et tokenise chaque exemple (tronqué à 256 tokens).
4. Masque les tokens du prompt dans les labels (`-100`) : la loss ne porte que sur la réponse.
5. Construit un `GPT2LMHeadModel` (2 couches, 128 dimensions, 2 têtes).
6. Entraîne avec le `Trainer` de Transformers (`logging_steps=50`, `save_steps=1000`).
7. Sauvegarde le modèle et le tokenizer dans `PicoGPT1-Jasmine/`.

### Durée et sorties

Sur CPU, comptez environ **2 min 40** pour la configuration par défaut. Le loss reste élevé (≈ 4,95 en fin d'entraînement), ce qui est attendu avec un modèle aussi petit.

Fichiers générés dans `PicoGPT1-Jasmine/` : `config.json`, `generation_config.json`, `model.safetensors`, `merges.txt`, `vocab.json`, `tokenizer.json`, `special_tokens_map.json`, `tokenizer_config.json`.

Un dossier `checkpoint-1250/` peut apparaître : il est ignoré par Git et peut être supprimé sans risque, les poids finaux sont dans `model.safetensors`.

---

## 8. Générer une réponse avec `test.py`

```bash
python test.py
```

Sans argument, le prompt est `What is the capital of France?\n` et le script affiche le texte complet (prompt + réponse).

### Options

```bash
python test.py --prompt "Explique la psychanalyse.\n" --do-sample
```

| Option | Défaut | Description |
| --- | --- | --- |
| `--model-dir` | `PicoGPT1-Jasmine` | Dossier du modèle entraîné |
| `--prompt` | `What is the capital of France?\n` | Texte à compléter |
| `--max-new-tokens` | `50` | Nombre maximum de tokens générés |
| `--do-sample` | désactivé | Active l'échantillonnage au lieu du décodage glouton |
| `--temperature` | `0.8` | Température (si `--do-sample`) |
| `--top-p` | `0.9` | Nucleus sampling (si `--do-sample`) |
| `--repetition-penalty` | `1.1` | Pénalité de répétition (si `--do-sample`) |

Le décodage glouton a tendance à répéter ; `--do-sample` donne des résultats plus variés. Terminez votre prompt par `\n` pour rester proche du format d'entraînement.

Exemples :

```bash
# bash
python test.py --prompt "What is the capital of France?\n" --do-sample

# PowerShell
python test.py --prompt "What is 2+2?`n" --do-sample --max-new-tokens 80
```

### Depuis votre propre code Python

Toute la logique est dans `model_utils.py` :

```python
from model_utils import load_model, generate

tokenizer, model = load_model("PicoGPT1-Jasmine")
print(generate(tokenizer, model, "What is 2+2?\n", do_sample=True))
```

`load_model()` charge tokenizer + modèle, passe en mode évaluation et neutralise `generation_config.max_length` pour éviter un conflit avec `max_new_tokens`. `generate()` tronque le prompt à 256 tokens et renvoie le texte décodé.

Si les poids sont absents, le message suivant s'affiche :

```
Poids introuvables dans 'PicoGPT1-Jasmine'. Lancez d'abord `python train.py`.
```

---

## 9. Serveur HTTP local avec `serve.py`

Le modèle peut être exposé par une petite API locale, sans framework web ni Ollama : seule la bibliothèque standard de Python est utilisée. Le modèle reste chargé en mémoire pendant toute la durée du serveur.

### Démarrage

```bash
python serve.py
```

Sortie attendue :

```
Serveur prêt : http://127.0.0.1:8000/generate
```

### Route `POST /generate`

Champs du corps JSON :

| Champ | Type | Défaut | Description |
| --- | --- | --- | --- |
| `prompt` | string | obligatoire | Texte à compléter |
| `max_new_tokens` | int | `50` | Nombre maximum de tokens générés |
| `do_sample` | bool | `false` | Active l'échantillonnage |

Réponses :

```json
{ "generated_text": "prompt + réponse" }
```

```json
{ "error": "message d'erreur" }
```

Toute autre route renvoie `404`.

### Exemples d'appel

**curl (bash / macOS / Linux)**

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is psychoanalysis?\n","do_sample":true}'
```

**PowerShell**

```powershell
$body = @{
    prompt = "What is psychoanalysis?`n"
    max_new_tokens = 50
    do_sample = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/generate -Method Post -Body $body -ContentType "application/json"
```

**Python**

```python
import requests

r = requests.post(
    "http://127.0.0.1:8000/generate",
    json={"prompt": "What is 2+2?\n", "do_sample": True},
)
print(r.json()["generated_text"])
```

### Options CLI

```bash
python serve.py --host 127.0.0.1 --port 9000 --model-dir PicoGPT1-Jasmine
```

| Option | Défaut | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Adresse d'écoute |
| `--port` | `8000` | Port d'écoute |
| `--model-dir` | `PicoGPT1-Jasmine` | Dossier du modèle |

Arrêtez le serveur avec `Ctrl+C`.

---

## 10. Exporter en GGUF

Les scripts Python utilisent `model.safetensors`. llama.cpp / llama.app attendent un fichier **GGUF** : `export_gguf.py` fait la conversion.

```bash
python export_gguf.py
```

Prérequis : `PicoGPT1-Jasmine/model.safetensors` doit exister (`python train.py`).

### Fonctionnement

1. Télécharge `convert_hf_to_gguf.py` depuis une version épinglée de llama.cpp (`b4000`).
2. Installe le paquet Python `gguf` dans un dossier temporaire s'il manque.
3. Lance la conversion avec `--outtype f16`.
4. Supprime les fichiers temporaires.

Résultat : `PicoGPT1-Jasmine.gguf` (~26,7 Mo, 29 tenseurs), ignoré par Git. Le `chat_template` présent dans `tokenizer_config.json` est conservé dans le GGUF.

### Vérification locale

```bash
llama cli -m PicoGPT1-Jasmine.gguf -st -p "What is 2+2?" -n 20
```

Après chaque réentraînement, relancez l'export puis la publication pour garder poids Transformers et GGUF synchronisés :

```bash
python export_gguf.py
python publish_hf.py
```

---

## 11. Publier sur Hugging Face

Le modèle est publié publiquement ici : **[nazimboudeffa/PicoGPT1-Jasmine](https://huggingface.co/nazimboudeffa/PicoGPT1-Jasmine)**.

### Prérequis

- `huggingface-cli login` effectué (ou `HF_TOKEN` défini)
- Poids Transformers présents : `python train.py`
- GGUF présent : `python export_gguf.py`
- `MODEL_CARD.md` présent à la racine

### Commande

```bash
python publish_hf.py
```

Le script :

1. Crée le dépôt s'il n'existe pas (`private=False`, `exist_ok=True`).
2. Upload le contenu de `PicoGPT1-Jasmine/` en ignorant `checkpoint-*`.
3. Upload `MODEL_CARD.md` comme `README.md` du dépôt Hub.
4. Upload `PicoGPT1-Jasmine.gguf`.

Une session HTTP avec retries est configurée pour absorber les erreurs TLS intermittentes de certains réseaux. En cas d'échec, relancez simplement la commande.

---

## 12. Utiliser le modèle publié

### Avec le CLI llama.app

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Jasmine -hff PicoGPT1-Jasmine.gguf
```

`-hff` force le fichier GGUF à utiliser. Comme le dépôt ne contient qu'un seul GGUF, la forme courte fonctionne aussi :

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Jasmine
```

### Interface web de chat

```bash
llama serve -hf nazimboudeffa/PicoGPT1-Jasmine -hff PicoGPT1-Jasmine.gguf
# puis ouvrir http://localhost:9931
```

### Avec Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "nazimboudeffa/PicoGPT1-Jasmine"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Quelle est la capitale de la France ?\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=48, do_sample=True, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Avec le serveur local

Si vous avez entraîné vos propres poids, `python serve.py` expose directement `PicoGPT1-Jasmine/`. Pour un autre dossier :

```bash
python serve.py --model-dir chemin/vers/mon/modele --port 9000
```

---

## 13. Architecture du modèle

Configuration définie dans `train.py` et sauvegardée dans `PicoGPT1-Jasmine/config.json` :

| Paramètre | Valeur |
| --- | --- |
| `model_type` | `gpt2` |
| `architectures` | `GPT2LMHeadModel` |
| `n_layer` | `2` |
| `n_embd` | `128` |
| `n_head` | `2` |
| `n_positions` / `n_ctx` | `256` |
| `vocab_size` | `50 257` |
| `activation_function` | `gelu_new` |
| `torch_dtype` | `float32` |

Soit environ **6,86 millions de paramètres**.

### Tokenizer et format appris

- Tokenizer GPT-2 (`GPT2TokenizerFast`), vocabulaire de 50 257 tokens.
- `<|endoftext|>` (ID `50256`) sert de token EOS, BOS, UNK et PAD.
- Chaque exemple est tokenisé comme `{prompt}\n{reponse}<|endoftext|>`.
- Les tokens du prompt, y compris le `\n` final, ont un label `-100` : la loss ne porte que sur la réponse.

### Chat template

Sauvegardé dans `tokenizer_config.json` et repris dans le GGUF :

```jinja
{% for message in messages %}{{ message['content'] }}{% if not loop.last %}{{ '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\n' }}{% endif %}
```

Les messages sont séparés par des sauts de ligne, et un saut de ligne est ajouté lorsque llama.cpp demande le début de la réponse.

### Génération

`model_utils.generate()` utilise `max_new_tokens` (50 par défaut), `pad_token_id = eos_token_id = 50256`, le décodage glouton par défaut et un échantillonnage optionnel (`temperature`, `top_p`, `repetition_penalty`).

---

## 14. Dépannage

| Symptôme | Solution |
| --- | --- |
| `Poids introuvables dans 'PicoGPT1-Jasmine'` | Les poids ne sont pas versionnés : lancez `python train.py` (puis `python export_gguf.py` pour le GGUF). |
| `Token is required ... no token found` | Connectez-vous avec `huggingface-cli login` ou définissez `HF_TOKEN`. |
| Erreur SSL pendant la publication | Les retries sont déjà configurés ; relancez `python publish_hf.py`. |
| Le modèle ne génère rien avec llama.cpp | Régénérez le GGUF après le dernier entraînement : `python export_gguf.py`, puis testez avec `llama cli ...`. |
| Réponses incohérentes | C'est attendu avec 500 exemples et 6,86 M paramètres. Essayez `--do-sample` et augmentez données / taille du modèle. |
| Port 8000 déjà utilisé | `python serve.py --port 9000`. |
| `LF will be replaced by CRLF...` sous Windows | Avertissement Git sans conséquence. |
| Dossier `checkpoint-1250/` | Créé par `Trainer`, ignoré par Git : supprimable sans risque. |

---

## 15. Limites et pistes d'amélioration

### Limites actuelles

- Jeu de données réduit à 500 exemples Alpaca.
- Architecture très petite (2 couches, 128 dimensions) et contexte de 256 tokens.
- Aucun jeu de validation : le loss affiché est uniquement le loss d'entraînement.
- Décodage glouton par défaut, avec tendance à la répétition.
- GGUF exporté en F16, sans quantification.

### Améliorations possibles

1. Augmenter `MAX_EXAMPLES` dans `prepare_dataset.py` **et** `train.py`.
2. Passer à 4 couches / 256 dimensions, ou partir d'un GPT-2 pré-entraîné.
3. Réserver 10 % des exemples pour un jeu de validation et suivre `eval_loss`.
4. Ajuster `temperature`, `top_p` et `repetition_penalty` selon l'usage.
5. Exporter le GGUF en `Q4_K_M` pour réduire sa taille et accélérer l'inférence.
6. Pousser les commits locaux sur GitHub si le dépôt doit être partagé.

---

## 16. Liens et références

- Dépôt GitHub : <https://github.com/nazimboudeffa/picogpt1-jasmine>
- Modèle Hugging Face : <https://huggingface.co/nazimboudeffa/PicoGPT1-Jasmine>
- Dataset : <https://huggingface.co/datasets/tatsu-lab/alpaca>
- Documentation détaillée : [doc/README.md](doc/README.md)
- Fiche modèle : [MODEL_CARD.md](MODEL_CARD.md)
- llama.app : <https://llama.app>
