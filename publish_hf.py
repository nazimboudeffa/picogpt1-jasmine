"""Publie PicoGPT1-Jasmine (poids Transformers + GGUF) sur Hugging Face.

Prérequis : être connecté via `huggingface-cli login` ou avoir HF_TOKEN.
"""

from pathlib import Path

import requests
from huggingface_hub import HfApi, configure_http_backend
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPO_ID = "nazimboudeffa/PicoGPT1-Jasmine"
MODEL_DIR = Path("PicoGPT1-Jasmine")
GGUF_FILE = Path("PicoGPT1-Jasmine.gguf")
MODEL_CARD = Path("MODEL_CARD.md")


def _session_factory() -> requests.Session:
    """Session qui retente les erreurs TLS intermittentes du réseau."""
    session = requests.Session()
    retries = Retry(
        total=10,
        connect=10,
        read=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=None,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def main() -> None:
    configure_http_backend(_session_factory)

    if not (MODEL_DIR / "model.safetensors").is_file():
        raise SystemExit("Poids introuvables. Lancez d'abord `python train.py`.")
    if not GGUF_FILE.is_file():
        raise SystemExit(f"{GGUF_FILE} introuvable. Convertissez le modèle en GGUF avant publication.")
    if not MODEL_CARD.is_file():
        raise SystemExit(f"{MODEL_CARD} introuvable.")

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)

    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Poids et tokenizer PicoGPT1-Jasmine",
        ignore_patterns=["checkpoint-*"],
    )
    api.upload_file(
        path_or_fileobj=str(MODEL_CARD),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Fiche modèle et instructions llama.app",
    )
    api.upload_file(
        path_or_fileobj=str(GGUF_FILE),
        path_in_repo=GGUF_FILE.name,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Ajout du GGUF pour llama.cpp / llama.app",
    )

    print(f"Modèle publié : https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
