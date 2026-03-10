# RAG
A basic locally-running RAG setup using Ollama and LangChain. Requires a plain text context file (.txt, .md, ...) which is embedded and stored in a local ChromaDB for retrieval.

## Setup
Install requirements, then pull your LM and embedding model of choice. Here I use [nomic-embed-text](https://ollama.com/library/nomic-embed-text) and [qwen3:8b](https://ollama.com/library/qwen3:8b):

```
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

## Execution
Ensure `model_name` variable in the script matches the LM tag you just pulled. Then run script.

Adjust the prompt and text splitting parameters to see the effect this has on the model's ability to answer context-based questions.
