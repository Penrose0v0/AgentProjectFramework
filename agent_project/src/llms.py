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
        self.memory.append({"role": "user", "content": prompt})
        messages = self.system_prompt + self.memory
        completion = self.send_to_model(messages)
        content = completion.message.content if self.local else completion.choices[0].message.content
        if "</think>" in content:
            think, response = map(lambda x: x.strip(), content.split("</think>"))
            logger.debug(f"Think: {think}")
        else:
            response = content

        if check_response is None or check_response(response): 
            self.memory.append({"role": "assistant", "content": response})
        else: 
            logger.warning("Response is not valid")

        return response


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


class DeepSeek(LLMBase): 
    def __init__(self, **kwargs): 
        super().__init__()
        self.system_prompt[0]["role"] = "system"
        self.local = kwargs["local"]
        if kwargs["local"]:
            self.send_to_model = lambda messages: ollama.chat(
                model="deepseek-r1:14b", 
                messages=messages, 
                options={
                    "temperature": kwargs["temperature"], 
                    "top_p": kwargs["top_p"]
                }
            )

        else:
            base_url = "https://api.siliconflow.cn/v1"
            api_key_path = kwargs["api_key_path"]
            with open(api_key_path, 'r', encoding='utf-8') as file: 
                api_key = file.read().strip()
            client = OpenAI(api_key=api_key, base_url=base_url)

            self.send_to_model = lambda messages: client.chat.completions.create(
            model="Pro/deepseek-ai/DeepSeek-V3",
            messages=messages, 
            temperature=kwargs["temperature"],
            top_p=kwargs["top_p"]
        )


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


def make_llm(config: dict) -> LLMBase: 
    llm_map = {
        "gpt": GPT,
        "deepseek": DeepSeek,
        "qwq": QWQ,
    }

    llm_name = config["llm"].lower()
    if llm_name in llm_map:
        llm_cls = llm_map[llm_name]
    else:
        raise ValueError(f"Unknown LLM: {config['llm']}")
    
    local = config["local"] if "local" in config else False
    api_key_path = config["api_key_path"] if "api_key_path" in config else None
    temperature = config["temperature"] if "temperature" in config else None
    top_p = config["top_p"] if "top_p" in config else None

    return llm_cls(
        local=local,
        api_key_path=api_key_path, 
        temperature=temperature, 
        top_p=top_p
        )