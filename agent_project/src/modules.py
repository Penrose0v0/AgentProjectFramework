import os
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from .llms import LLMClient


class PromptRegistry:
    def __init__(self, base_dir="prompts"):
        self.base_dir = base_dir

    def load(self, name):
        path = os.path.join(self.base_dir, name + ".txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


class AgentBase:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

        self.registry = PromptRegistry()
        self.system_prompt = SystemMessagePromptTemplate.from_template(self.registry.load("agent"))

    def run(self, user_template: str, stream=False, **kwargs):
        user_prompt = HumanMessagePromptTemplate.from_template(user_template)
        prompt = ChatPromptTemplate.from_messages([self.system_prompt, user_prompt])
        messages = prompt.format_messages(**kwargs)
        
        return self.llm.call(messages, stream=stream)
    

class TestAgent(AgentBase): 
    def __init__(self, config: dict): 
        super().__init__(LLMClient(config=config))
        self.prompts = {
            "chat": self.registry.load("chat")
        }

    def chat(self, contents, stream=False): 
        return self.run(self.prompts["chat"], stream=stream, contents=contents)
    