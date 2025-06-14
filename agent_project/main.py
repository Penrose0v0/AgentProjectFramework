import os
import yaml

from src.modules import TestAgent
from src.utils import get_next_output_folder


def main(args): 
    print("程序启动")

    # Read config
    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    output_folder = get_next_output_folder()

    # Init modules
    agent = TestAgent(config=config)
    print("模组初始化成功")

    # Start main loop
    print("开始进行对话\n")
    while True: 
        user_input = input("User: ")
        if user_input == "q": 
            break

        # Streaming output
        print("Agent: ", end="", flush=True)
        for chunk in agent.chat(contents=user_input, stream=True):
            print(chunk, end="", flush=True)
        print()

    print("\n对话结束")

    # Export memory
    export_file = os.path.join(output_folder, "memory.json")
    agent.llm.export_memory(export_file)
    print("对话记录导出成功")

    print("程序结束")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent CLI 对话程序")
    parser.add_argument("--config", type=str, default="./configs/qwq_ollama.yaml", help="LLM config path")
    args = parser.parse_args()

    main(args)
