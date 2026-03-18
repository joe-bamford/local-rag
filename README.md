# RAG

## Overview
A basic locally-running RAG setup using Ollama and LangChain. Requires a plain text context file (.txt, .md, ...) which is embedded and stored in a local ChromaDB for retrieval.

Example context file (canterbury.txt) is the General Prologue of the Canterbury tales by Chaucer (roughly 8500 tokens).

## Setup

First, [install Ollama](https://ollama.com/download).

Install requirements, then pull your LM and embedding model of choice. Here I use [qwen3-embedding:4b](https://ollama.com/library/qwen3-embedding:4b) and [qwen3.5:9b](https://ollama.com/library/qwen3.5:9b):

```
ollama pull qwen3-embedding:4b
ollama pull qwen3.5:9b
```

> NOTE: The 9B parameter model runs comfortably on my GPU (12GB VRAM), but a rough rule of thumb for required RAM in GB is $\sim 2 * N_{params}$. Ollama auto-detects GPUs and will utilise them as far as possible when performing inference.

## Execution
Ensure `model_name` variable in the script matches the LM tag you just pulled. Supply a query then run script.

Adjust the prompt and text splitting parameters to see the effect this has on the model's ability to answer context-based questions.