# 6. Serveur local avec `serve.py`

`serve.py` expose le modèle sur HTTP avec la bibliothèque standard de Python. Aucun framework web ni Ollama n'est nécessaire.

## Démarrage

```bash
python serve.py
```

Sortie attendue :

```
Serveur prêt : http://127.0.0.1:8000/generate
```

Le modèle reste chargé en mémoire pendant toute la durée du serveur.

## Route

`POST /generate`

Corps JSON :

| Champ | Type | Défaut | Description |
| --- | --- | --- | --- |
| `prompt` | string | obligatoire | Texte à compléter |
| `max_new_tokens` | int | `50` | Nombre maximum de tokens |
| `do_sample` | bool | `false` | Active l'échantillonnage |

Réponse succès (`200`) :

```json
{ "generated_text": "prompt + réponse" }
```

Erreur (`400`) :

```json
{ "error": "message d'erreur" }
```

Toute autre route renvoie `404`.

## Exemples

PowerShell :

```powershell
$body = @{
    prompt = "What is psychoanalysis?`n"
    max_new_tokens = 50
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/generate -Method Post -Body $body -ContentType "application/json"
```

curl :

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is psychoanalysis?\n","do_sample":true}'
```

## Options CLI

```bash
python serve.py --host 127.0.0.1 --port 9000 --model-dir PicoGPT1-Jasmine
```

| Option | Défaut | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Adresse d'écoute |
| `--port` | `8000` | Port d'écoute |
| `--model-dir` | `PicoGPT1-Jasmine` | Dossier du modèle |

Arrêter le serveur avec `Ctrl+C`.
