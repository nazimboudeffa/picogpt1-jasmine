"""Teste le modèle entraîné en générant une réponse à partir d'un prompt.

Usage :
    python test.py
    python test.py --prompt "Quelle est la capitale de la France ?" --do-sample
"""

import argparse

from model_utils import DEFAULT_MAX_NEW_TOKENS, MODEL_DIR, generate, load_model


def main():
    parser = argparse.ArgumentParser(description="Génère une réponse avec PicoGPT1-Jasmine")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="dossier du modèle entraîné")
    parser.add_argument("--prompt", default="What is the capital of France?\n", help="prompt à compléter")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="nombre maximum de tokens générés",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="échantillonnage au lieu du décodage glouton (répétitions moindres)",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)
    print(
        generate(
            tokenizer,
            model,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    )


if __name__ == "__main__":
    main()
