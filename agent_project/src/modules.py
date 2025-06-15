import os
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage

from .llms import LLMClient
from .rag import RAGComponent


class PromptRegistry:
    def __init__(self, base_dir="prompts"):
        self.base_dir = base_dir

    def load(self, name):
        path = os.path.join(self.base_dir, name + ".txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


class AgentBase:
    def __init__(self, llm_client: LLMClient, rag_component: RAGComponent = None):
        self.llm = llm_client
        self.rag = rag_component

        self.registry = PromptRegistry()
        self.system_prompt = SystemMessagePromptTemplate.from_template(self.registry.load("agent"))

    def run(self, user_template: str, rag_template: str = None, top_k: int = 5, stream: bool = False, **kwargs):
        if self.rag:
            return self._run_with_rag(user_template, rag_template, top_k, stream=stream, **kwargs)
        else:
            return self._run_without_rag(user_template, stream=stream, **kwargs)
    
    def _run_with_rag(self, user_template: str, rag_template: str, top_k: int, stream: bool = False, **kwargs): 
        user_prompt = HumanMessagePromptTemplate.from_template(user_template)

        query = None
        for k, v in kwargs.items():
            if isinstance(v, str):
                query = v
                break

        if query is None:
            raise ValueError("RAG 启用时未能从参数中提取有效的查询字段。")

        docs = self.rag.search(query, top_k)
        context = "\n".join([doc.page_content for doc in docs])
        rag_template = rag_template if rag_template else "参考资料：\n{context}"
        retrievals = HumanMessage(content=rag_template.format(context=context))

        messages = ChatPromptTemplate.from_messages(
            [self.system_prompt, retrievals, user_prompt]
        ).format_messages(**kwargs)

        return self.llm.call(messages, stream=stream)

    def _run_without_rag(self, user_template: str, stream=False, **kwargs): 
        user_prompt = HumanMessagePromptTemplate.from_template(user_template)
        prompt = ChatPromptTemplate.from_messages([self.system_prompt, user_prompt])
        messages = prompt.format_messages(**kwargs)

        return self.llm.call(messages, stream=stream)
    

class TestAgent(AgentBase): 
    def __init__(self, llm_config: dict, rag_config: dict): 
        super().__init__(LLMClient(config=llm_config), RAGComponent(config=rag_config))
        self.prompts = {
            "chat": self.registry.load("chat")
        }

    def chat(self, contents, stream=False): 
        self.rag.sync_documents()
        return self.run(self.prompts["chat"], stream=stream, contents=contents)
    