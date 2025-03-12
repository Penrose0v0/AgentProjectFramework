import os

from .llms import make_llm
from .utils import logger

def load_prompt(prompt: str) -> str:
    prompt_path = "prompts"
    prompt_file = prompt + ".txt"
    with open(os.path.join(prompt_path, prompt_file), 'r', encoding='utf-8') as file: 
        content = file.read() 
    return content

    
class Agent: 
    def __init__(self, config: dict): 
        self.llm = make_llm(config=config)

        system_prompt = load_prompt("agent")
        self.llm.set_system_prompt(system_prompt=system_prompt)
    
    
    def chat(self, contents: str):
        """
        Input: User contents
        Output: Agent response
        """
        prompt_template = load_prompt("chat")
        prompt = prompt_template.format(contents=contents)

        response = self.llm.response(prompt)
        logger.debug(f"Response of `chat`: \n{response}")

        return response
