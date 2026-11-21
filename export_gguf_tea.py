"""Convertit le modèle fusionné Jasmine-Tea en GGUF pour llama.app / llama.cpp.

Le convertisseur est téléchargé depuis une version épinglée de llama.cpp ;
le paquet Python `gguf` est installé dans un dossier temporaire si besoin.
"""

import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MODEL_DIR = Path("PicoGPT1-Jasmine-Tea-Merged")
GGUF_FILE = Path("PicoGPT1-Jasmine-Tea.gguf")
CONVERTER_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/b4500/convert_hf_to_gguf.py"


def _ensure_deps(target: Path) -> str | None:
    """Installe les paquets Python manquants dans un dossier temporaire."""
    import importlib.util

    required = ("gguf", "sentencepiece")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if not missing:
        return None

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", str(target), *missing],
        check=True,
    )
    return str(target)


def main() -> None:
    if not (MODEL_DIR / "model.safetensors").is_file():
        raise SystemExit("Modèle fusionné introuvable. Lancez d'abord `python merge_tea.py`.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        converter = tmp_dir / "convert_hf_to_gguf.py"
        print("Téléchargement du convertisseur GGUF...")
        urllib.request.urlretrieve(CONVERTER_URL, converter)

        gguf_path = _ensure_deps(tmp_dir / "gguf_pkg")
        env = os.environ.copy()
        env["NO_LOCAL_GGUF"] = "1"
        if gguf_path:
            env["PYTHONPATH"] = gguf_path + os.pathsep + env.get("PYTHONPATH", "")

        subprocess.run(
            [
                sys.executable,
                str(converter),
                str(MODEL_DIR),
                "--outfile",
                str(GGUF_FILE),
                "--outtype",
                "f16",
            ],
            check=True,
            env=env,
        )

    print(f"GGUF exporté : {GGUF_FILE}")


if __name__ == "__main__":
    main()
