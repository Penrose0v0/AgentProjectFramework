from openai import OpenAI
import ollama
import json

from .utils import logger


class LLMBase: 
    def __init__(self): 
        self.system_prompt = [{"role": "", "content": ""}]
        self.memory = []

    def clear_memory(self): 
        self.memory = []

    def export_memory(self, export_file, include_system_prompt=True):
        memory = self.system_prompt + self.memory if include_system_prompt else self.memory
        with open(export_file, 'w', encoding='utf-8') as file:
            json.dump(memory, file, ensure_ascii=False, indent=4)

    def set_system_prompt(self, system_prompt): 
        self.system_prompt[0]["content"] = system_prompt

    def response(self, prompt, check_response=None): 
        pass


class GPT(LLMBase): 
    def __init__(self, **kwargs):
        super().__init__()
        api_key_path = kwargs["api_key_path"]
        with open(api_key_path, 'r', encoding='utf-8') as file: 
            api_key = file.read().strip()
        client = OpenAI(api_key=api_key)

        self.system_prompt[0]["role"] = "developer"
        self.send_to_model = lambda messages: client.chat.completions.create(
            model="gpt-4o",
            messages=messages, 
            temperature=kwargs["temperature"],
            top_p=kwargs["top_p"]
        )

    def response(self, prompt, check_response=None): 
        self.memory.append({"role": "user", "content": prompt})
        messages = self.system_prompt + self.memory
        completion = self.send_to_model(messages)
        response = completion.choices[0].message.content

        if check_response is None or check_response(response): 
            self.memory.append({"role": "assistant", "content": response})
        else: 
            logger.warning("Response is not valid")

        return response

class DeepSeek(LLMBase): 
    def __init__(self, **kwargs): 
        super().__init__()
        self.system_prompt[0]["role"] = "system"
        self.send_to_model = lambda messages: ollama.chat(
            model="deepseek-r1:14b", 
            messages=messages, 
            options={
                "temperature": kwargs["temperature"], 
                "top_p": kwargs["top_p"]
            }
        )

    def response(self, prompt, check_response=None): 
        self.memory.append({"role": "user", "content": prompt})
        messages = self.system_prompt + self.memory
        completion = self.send_to_model(messages)
        think, response = map(lambda x: x.strip(), completion.message.content.split("</think>"))
        logger.debug(f"Think: {think}")

        if check_response is None or check_response(response): 
            self.memory.append({"role": "assistant", "content": response})
        else: 
            logger.warning("Response is not valid")

        return response

class QWQ(LLMBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.system_prompt[0]["role"] = "system"
        self.send_to_model = lambda messages: ollama.chat(
            model="qwq:latest",
            messages=messages,
            options={
                "temperature": kwargs["temperature"],
                "top_p": kwargs["top_p"]
            }
        )

    def response(self, prompt, check_response=None): 
        self.memory.append({"role": "user", "content": prompt})
        messages = self.system_prompt + self.memory
        completion = self.send_to_model(messages)
        think, response = map(lambda x: x.strip(), completion.message.content.split("</think>"))
        logger.debug(f"Think: {think}")

        if check_response is None or check_response(response): 
            self.memory.append({"role": "assistant", "content": response})
        else: 
            logger.warning("Response is not valid")

        return response


def make_llm(config: dict) -> LLMBase: 
    if config["llm"] == "gpt": 
        llm_cls = GPT
    elif config["llm"] == "deepseek": 
        llm_cls = DeepSeek
    elif config["llm"] == "qwq":
        llm_cls = QWQ
    else: 
        raise ValueError(f"Unknown LLM: {config['llm']}")
    
    api_key_path = config["api_key_path"] if "api_key_path" in config else None
    temperature = config["temperature"] if "temperature" in config else None
    top_p = config["top_p"] if "top_p" in config else None

    return llm_cls(
        api_key_path=api_key_path, 
        temperature=temperature, 
        top_p=top_p
        )