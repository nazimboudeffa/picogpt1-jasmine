---
library_name: transformers
tags:
  - gpt2
  - gguf
  - llama.cpp
  - llama.app
  - alpaca
  - instruction
---

# PicoGPT1-Jasmine

Petit modèle GPT-2 entraîné sur un sous-ensemble de 500 instructions/réponses Alpaca.

- Architecture : 2 couches, 128 dimensions, 2 têtes d'attention, contexte 256 tokens
- Tokenizer : GPT-2 (50 257 tokens)
- Format appris : `prompt\nreponse<|endoftext|>`
- Fichiers : poids Transformers (`model.safetensors`) et variante `PicoGPT1-Jasmine.gguf`

> Modèle expérimental de très petite taille : les réponses peuvent être incohérentes.

## Utiliser avec llama.app / llama.cpp

Le CLI fourni par [llama.app](https://llama.app) télécharge le GGUF depuis ce dépôt :

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Jasmine -hff PicoGPT1-Jasmine.gguf
```

Pour ouvrir l'interface web de chat :

```bash
llama serve -hf nazimboudeffa/PicoGPT1-Jasmine -hff PicoGPT1-Jasmine.gguf
# puis http://localhost:9931
```

## Utiliser avec Transformers

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
