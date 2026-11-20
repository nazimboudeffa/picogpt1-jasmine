"""Convertit PicoGPT1-Jasmine en GGUF pour llama.app / llama.cpp.

Le convertisseur est téléchargé depuis une version épinglée de llama.cpp ;
le paquet Python `gguf` est installé dans un dossier temporaire si besoin.
"""

import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MODEL_DIR = Path("PicoGPT1-Jasmine")
GGUF_FILE = Path("PicoGPT1-Jasmine.gguf")
CONVERTER_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/b4000/convert_hf_to_gguf.py"


def _ensure_gguf(target: Path) -> str | None:
    try:
        import gguf  # noqa: F401

        return None
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", str(target), "gguf"],
            check=True,
        )
        return str(target)


def main() -> None:
    if not (MODEL_DIR / "model.safetensors").is_file():
        raise SystemExit("Poids introuvables. Lancez d'abord `python train.py`.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        converter = tmp_dir / "convert_hf_to_gguf.py"
        print("Téléchargement du convertisseur GGUF...")
        urllib.request.urlretrieve(CONVERTER_URL, converter)

        gguf_path = _ensure_gguf(tmp_dir / "gguf_pkg")
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
