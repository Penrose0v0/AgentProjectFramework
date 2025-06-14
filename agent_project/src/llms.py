from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
import json


def make_llm(config: dict) -> BaseChatModel:
    # Default
    provider = config.get("provider", "openai")
    model = config.get("model", "Pro/deepseek-ai/DeepSeek-V3")
    temperature = config.get("temperature", 0.7)
    streaming = config.get("streaming", True)

    api_key_path = config.get("api_key_path")
    if api_key_path:
        with open(api_key_path, 'r', encoding='utf-8') as file: 
            api_key = file.read().strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            openai_api_key=api_key,
            openai_api_base=config.get("base_url")
        )
    
    elif provider == "claude":
        from langchain.chat_models import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            streaming=streaming,
            anthropic_api_key=api_key
        )

    elif provider == "gemini":
        from langchain.chat_models import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            google_api_key=api_key
        )
    
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=config.get("base_url", "http://localhost:11434")
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


class LLMClient:
    def __init__(self, config: dict):
        self.llm = make_llm(config)
        self.memory = InMemoryChatMessageHistory()

    def call(self, messages, stream=False):
        if stream:
            return self._stream_call(messages)
        else:
            return self._normal_call(messages)
    
    def _normal_call(self, messages):
        full_input = self.memory.messages + messages
        response = self.llm.invoke(full_input)

        self.memory.add_message(messages[-1])
        self.memory.add_ai_message(response.content)

        return response.content
    
    def _stream_call(self, messages):
        full_input = self.memory.messages + messages
        content = ""

        for chunk in self.llm.stream(full_input):
            delta = chunk.content
            content += delta
            yield delta

        self.memory.add_message(messages[-1])
        self.memory.add_ai_message(content)
        

    def clear_memory(self):
        self.memory.clear()

    def export_memory(self, export_file):
        memory = [
            {"role": msg.type, "content": msg.content}
            for msg in self.memory.messages
        ]
        with open(export_file, 'w', encoding='utf-8') as file:
            json.dump(memory, file, ensure_ascii=False, indent=4)

        return memory
