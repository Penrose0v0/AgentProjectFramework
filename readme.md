# Agent Project Framework

## Getting started
Run `pip install -r requirement.txt`\
Edit `config.yaml`\
Add your api key in `key`\
Prepare your prompts in `prompts`

Finally, run `python .\agent_project\main.py`

## To do
- RAG
  - Retrieval
  - Prompt augmentation
- Process \<think> tag
- Web UI

## Memo
**06-15**\
Init rag component\
Using Chroma as the vector database

**06-14**\
Updated the whole framework with LangChain

**04-02**\
Basically, if model is local, use `ollama`;\
if model is online, use `openai`

**04-01**\
Able to call deepseek via SiliconFlow